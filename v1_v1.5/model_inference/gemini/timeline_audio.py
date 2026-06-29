import os
import wave


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
            dst.writeframes(b"\x00" * silence_frames * channels * sample_width)
        dst.writeframes(frames)
        if trailing_silence_frames:
            dst.writeframes(b"\x00" * trailing_silence_frames * channels * sample_width)
