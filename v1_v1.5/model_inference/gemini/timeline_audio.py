import os
import wave


def _silence_frames(frame_count, channels, sample_width):
    return b"\x00" * frame_count * channels * sample_width


def pad_timeline_wav_to_min_duration(
    wav_path,
    sample_rate,
    min_duration_sec,
    default_channels=1,
    default_sample_width=2,
):
    """Rewrite a timeline-aligned WAV so it is at least min_duration_sec long."""
    target_frames = max(0, int(round(min_duration_sec * sample_rate)))
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)

    if os.path.exists(wav_path):
        with wave.open(wav_path, "rb") as src:
            channels = src.getnchannels()
            sample_width = src.getsampwidth()
            frame_rate = src.getframerate()
            frame_count = src.getnframes()
            frames = src.readframes(frame_count)
    else:
        channels = default_channels
        sample_width = default_sample_width
        frame_rate = sample_rate
        frame_count = 0
        frames = b""

    if frame_rate != sample_rate:
        raise ValueError(f"Expected {sample_rate}Hz timeline audio, got {frame_rate}Hz")

    padding_frames = max(0, target_frames - frame_count)
    if padding_frames:
        with wave.open(wav_path, "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(sample_width)
            dst.setframerate(frame_rate)
            dst.writeframes(frames)
            dst.writeframes(_silence_frames(padding_frames, channels, sample_width))

    return (frame_count + padding_frames) / sample_rate


def align_response_stem_to_timeline(
    raw_response_wav,
    output_wav,
    response_start_sec,
    sample_rate,
    min_duration_sec=0.0,
):
    """Create an agent-only stem that starts at conversation time 0."""
    response_start_sec = response_start_sec or 0.0
    silence_frames = max(0, int(round(response_start_sec * sample_rate)))

    with wave.open(raw_response_wav, "rb") as src:
        channels = src.getnchannels()
        sample_width = src.getsampwidth()
        frame_rate = src.getframerate()
        frames = src.readframes(src.getnframes())

    if frame_rate != sample_rate:
        raise ValueError(f"Expected {sample_rate}Hz response audio, got {frame_rate}Hz")

    current_frames = silence_frames + (len(frames) // (channels * sample_width))
    min_frames = max(0, int(round(min_duration_sec * sample_rate)))
    trailing_silence_frames = max(0, min_frames - current_frames)

    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    with wave.open(output_wav, "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sample_width)
        dst.setframerate(frame_rate)
        if silence_frames:
            dst.writeframes(_silence_frames(silence_frames, channels, sample_width))
        dst.writeframes(frames)
        if trailing_silence_frames:
            dst.writeframes(_silence_frames(trailing_silence_frames, channels, sample_width))
