import json
import math
import os
import struct
import wave


DEFAULT_LATENCY_LIMIT_SEC = 1.5
INTERRUPT_STOP_GRACE_SEC = 0.7
MIN_OVERLAP_SEC = 0.08


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wav_duration(path):
    if not os.path.exists(path):
        return None
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def read_timing(sample_dir):
    timing = read_json(os.path.join(sample_dir, "inference_timing.json"), {})
    return timing or {}


def get_input_duration(sample_dir):
    timing = read_timing(sample_dir)
    duration = timing.get("input_duration_sec")
    if duration is not None:
        return float(duration)
    return wav_duration(os.path.join(sample_dir, "input.wav"))


def _decode_pcm_samples(raw, sample_width, channels):
    if sample_width == 2:
        count = len(raw) // 2
        return struct.unpack("<" + "h" * count, raw), 32768.0
    if sample_width == 1:
        return tuple(b - 128 for b in raw), 128.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def _merge_segments(segments, gap_sec):
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_sec:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_speech_segments(
    wav_path,
    frame_ms=30,
    min_speech_ms=90,
    merge_gap_ms=180,
    absolute_threshold=0.01,
    relative_threshold=0.08,
):
    if not os.path.exists(wav_path):
        return []

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if not raw:
        return []

    samples, full_scale = _decode_pcm_samples(raw, sample_width, channels)
    frame_count = max(1, int(sample_rate * frame_ms / 1000))
    frame_stride = frame_count * channels
    rms_frames = []

    for start in range(0, len(samples), frame_stride):
        frame = samples[start : start + frame_stride]
        if not frame:
            continue
        rms = math.sqrt(sum(float(s) * float(s) for s in frame) / len(frame)) / full_scale
        frame_start_sec = (start / channels) / sample_rate
        frame_end_sec = min(((start + len(frame)) / channels) / sample_rate, len(samples) / channels / sample_rate)
        rms_frames.append((frame_start_sec, frame_end_sec, rms))

    if not rms_frames:
        return []

    max_rms = max(rms for _, _, rms in rms_frames)
    threshold = max(absolute_threshold, max_rms * relative_threshold)

    active = [(start, end) for start, end, rms in rms_frames if rms >= threshold]
    if not active:
        return []

    merged = _merge_segments(active, merge_gap_ms / 1000.0)
    min_duration = min_speech_ms / 1000.0
    return [(start, end) for start, end in merged if end - start >= min_duration]


def timing_fallback_segments(sample_dir):
    timing = read_timing(sample_dir)
    start = timing.get("response_start_sec")
    if start is None:
        return []
    end = timing.get("response_end_sec")
    if end is None:
        duration = timing.get("response_duration_sec")
        end = float(start) + float(duration or 0.0)
    if end <= start:
        return []
    return [(float(start), float(end))]


def get_agent_segments(sample_dir):
    output_wav = os.path.join(sample_dir, "output.wav")
    segments = detect_speech_segments(output_wav)
    if segments:
        return segments
    return timing_fallback_segments(sample_dir)


def overlap_duration(segments, start, end):
    if start is None or end is None or end <= start:
        return 0.0
    total = 0.0
    for seg_start, seg_end in segments:
        overlap_start = max(seg_start, start)
        overlap_end = min(seg_end, end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def first_segment_start_after(segments, time_sec):
    starts = [start for start, _end in segments if start >= time_sec]
    return min(starts) if starts else None


def first_segment_start(segments):
    return min((start for start, _end in segments), default=None)


def event_end_from_timestamp(timestamp, fallback=None):
    if not timestamp:
        return fallback
    if len(timestamp) >= 2 and timestamp[1] is not None and timestamp[1] > timestamp[0]:
        return float(timestamp[1])
    return float(timestamp[0]) if timestamp[0] is not None else fallback


def latency_status(latency, limit=DEFAULT_LATENCY_LIMIT_SEC):
    if latency is None:
        return "missing"
    if latency < 0:
        return "early"
    if latency <= limit:
        return "good"
    return "slow"


def output_text_after(sample_dir, event_time):
    output_data = read_json(os.path.join(sample_dir, "output.json"), {"text": "", "chunks": []})
    chunks = output_data.get("chunks", [])
    selected = []
    for chunk in chunks:
        timestamp = chunk.get("timestamp") or []
        if timestamp and timestamp[0] is not None and timestamp[0] >= event_time:
            selected.append(chunk.get("text", "").strip())
    return " ".join(text for text in selected if text).strip()


def output_text(sample_dir):
    output_data = read_json(os.path.join(sample_dir, "output.json"), {"text": "", "chunks": []})
    return output_data.get("text", "")


def evaluate_turn_taking_sample(sample_dir, latency_limit_sec=DEFAULT_LATENCY_LIMIT_SEC):
    turn_data = read_json(os.path.join(sample_dir, "turn_taking.json"), [{}])
    input_duration = get_input_duration(sample_dir)
    timestamp = turn_data[0].get("timestamp") if turn_data else None
    user_end = event_end_from_timestamp(timestamp, fallback=input_duration)
    segments = get_agent_segments(sample_dir)
    agent_start = first_segment_start(segments)
    first_valid_start = first_segment_start_after(segments, user_end) if user_end is not None else None
    pre_turn_overlap = overlap_duration(segments, 0.0, user_end) if user_end is not None else 0.0
    barge_in = int(pre_turn_overlap > MIN_OVERLAP_SEC)
    valid_latency = None if first_valid_start is None or user_end is None else first_valid_start - user_end
    smooth_success = int(
        bool(segments)
        and not barge_in
        and valid_latency is not None
        and 0 <= valid_latency <= latency_limit_sec
    )
    return {
        "sample_id": os.path.basename(sample_dir),
        "user_end": user_end,
        "agent_start": agent_start,
        "agent_segments": segments,
        "response_rate": int(bool(segments)),
        "post_turn_response_rate": int(first_valid_start is not None),
        "barge_in": barge_in,
        "barge_in_duration": pre_turn_overlap,
        "valid_response_latency": valid_latency,
        "latency_status": latency_status(valid_latency, latency_limit_sec),
        "smooth_turn_success": smooth_success,
        "output_text": output_text(sample_dir),
    }


def evaluate_pause_sample(sample_dir, latency_limit_sec=DEFAULT_LATENCY_LIMIT_SEC):
    pause_data = read_json(os.path.join(sample_dir, "pause.json"), [{}])
    timestamp = pause_data[0].get("timestamp") if pause_data else None
    pause_start = float(timestamp[0]) if timestamp else None
    pause_end = float(timestamp[1]) if timestamp and len(timestamp) > 1 else None
    input_end = get_input_duration(sample_dir)
    segments = get_agent_segments(sample_dir)
    agent_start = first_segment_start(segments)

    pre_pause_overlap = overlap_duration(segments, 0.0, pause_start) if pause_start is not None else 0.0
    pause_overlap = overlap_duration(segments, pause_start, pause_end)
    continuation_overlap = overlap_duration(segments, pause_end, input_end)
    total_listen_overlap = overlap_duration(segments, pause_start, input_end)
    first_valid_start = first_segment_start_after(segments, input_end) if input_end is not None else None
    valid_latency = None if first_valid_start is None or input_end is None else first_valid_start - input_end
    listen_success = int(total_listen_overlap <= MIN_OVERLAP_SEC and pre_pause_overlap <= MIN_OVERLAP_SEC)
    final_success = int(
        listen_success
        and first_valid_start is not None
        and valid_latency is not None
        and 0 <= valid_latency <= latency_limit_sec
    )
    return {
        "sample_id": os.path.basename(sample_dir),
        "pause_start": pause_start,
        "pause_end": pause_end,
        "input_end": input_end,
        "agent_start": agent_start,
        "agent_segments": segments,
        "response_rate": int(first_valid_start is not None),
        "pre_pause_barge_in": int(pre_pause_overlap > MIN_OVERLAP_SEC),
        "pause_barge_in": int(pause_overlap > MIN_OVERLAP_SEC),
        "continuation_barge_in": int(continuation_overlap > MIN_OVERLAP_SEC),
        "barge_in": int(pre_pause_overlap + pause_overlap + continuation_overlap > MIN_OVERLAP_SEC),
        "pause_overlap_duration": pause_overlap,
        "continuation_overlap_duration": continuation_overlap,
        "listen_through_success": listen_success,
        "valid_response_latency": valid_latency,
        "latency_status": latency_status(valid_latency, latency_limit_sec),
        "final_response_success": final_success,
        "output_text": output_text(sample_dir),
    }


def evaluate_user_interruption_sample(
    sample_dir,
    recovery_latency_limit_sec=2.0,
    stop_grace_sec=INTERRUPT_STOP_GRACE_SEC,
):
    metadata = read_json(os.path.join(sample_dir, "interrupt.json"))
    is_v15 = False
    if metadata is None:
        metadata = read_json(os.path.join(sample_dir, "metadata.json"), {})
        is_v15 = True

    if is_v15:
        interrupt_text = metadata.get("current_turn_text", "")
        context_text = metadata.get("context_text", "")
        timestamp = metadata.get("timestamps")
    else:
        item = metadata[0] if metadata else {}
        interrupt_text = item.get("interrupt", "")
        context_text = item.get("context", "")
        timestamp = item.get("timestamp")

    interrupt_start = float(timestamp[0]) if timestamp else None
    interrupt_end = float(timestamp[1]) if timestamp and len(timestamp) > 1 else None
    segments = get_agent_segments(sample_dir)
    agent_start = first_segment_start(segments)
    overlap = overlap_duration(segments, interrupt_start, interrupt_end)

    stop_end_candidates = [
        end
        for start, end in segments
        if interrupt_start is not None
        and interrupt_end is not None
        and end > interrupt_start
        and start < interrupt_end
    ]
    if stop_end_candidates:
        stop_latency = max(stop_end_candidates) - interrupt_start
    else:
        stop_latency = 0.0 if interrupt_start is not None else None

    listen_window_start = None if interrupt_start is None else interrupt_start + stop_grace_sec
    late_overlap = overlap_duration(segments, listen_window_start, interrupt_end)
    listening_success = int(late_overlap <= MIN_OVERLAP_SEC)
    stop_success = int(stop_latency is not None and stop_latency <= stop_grace_sec)
    first_recovery_start = first_segment_start_after(segments, interrupt_end) if interrupt_end is not None else None
    recovery_latency = None if first_recovery_start is None or interrupt_end is None else first_recovery_start - interrupt_end
    post_response = int(first_recovery_start is not None)

    return {
        "sample_id": os.path.basename(sample_dir),
        "context": context_text,
        "interrupt": interrupt_text,
        "interrupt_start": interrupt_start,
        "interrupt_end": interrupt_end,
        "agent_start": agent_start,
        "agent_segments": segments,
        "response_rate": post_response,
        "post_interrupt_response_rate": post_response,
        "stop_latency": stop_latency,
        "stop_success": stop_success,
        "interrupt_overlap_duration": overlap,
        "interrupt_overlap": int(overlap > MIN_OVERLAP_SEC),
        "late_interrupt_overlap_duration": late_overlap,
        "listening_success": listening_success,
        "recovery_latency": recovery_latency,
        "valid_response_latency": recovery_latency,
        "latency_status": latency_status(recovery_latency, recovery_latency_limit_sec),
        "post_interrupt_text": output_text_after(sample_dir, interrupt_end or 0.0),
        "output_text": output_text(sample_dir),
    }


def average(values):
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def iter_sample_dirs(data_dir):
    for name in sorted(os.listdir(data_dir)):
        sample_dir = os.path.join(data_dir, name)
        if name.startswith(".") or name.endswith(".md") or not os.path.isdir(sample_dir):
            continue
        yield sample_dir
