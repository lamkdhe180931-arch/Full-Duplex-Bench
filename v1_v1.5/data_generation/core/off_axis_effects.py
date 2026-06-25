from pathlib import Path

from pydub import AudioSegment


DEFAULT_FRAME_RATE = 16000
DEFAULT_SAMPLE_WIDTH = 2


def _ensure_benchmark_format(sound):
    return (
        sound.set_frame_rate(DEFAULT_FRAME_RATE)
        .set_channels(1)
        .set_sample_width(DEFAULT_SAMPLE_WIDTH)
    )


def _apply_rir_convolution(sound, rir_sound):
    import numpy as np

    clean = _ensure_benchmark_format(sound)
    rir = _ensure_benchmark_format(rir_sound)

    clean_samples = np.array(clean.get_array_of_samples()).astype(np.float32)
    rir_samples = np.array(rir.get_array_of_samples()).astype(np.float32)

    if len(rir_samples) == 0 or np.max(np.abs(rir_samples)) == 0:
        return clean

    clean_scale = float(2 ** (8 * clean.sample_width - 1))
    rir_scale = float(2 ** (8 * rir.sample_width - 1))
    clean_float = clean_samples / clean_scale
    rir_float = rir_samples / rir_scale
    rir_float = rir_float / max(float(np.max(np.abs(rir_float))), 1e-8)

    conv_len = len(clean_float) + len(rir_float) - 1
    fft_len = 1 << (conv_len - 1).bit_length()
    convolved = np.fft.irfft(
        np.fft.rfft(clean_float, fft_len) * np.fft.rfft(rir_float, fft_len),
        fft_len,
    )[: len(clean_float)]

    peak = max(float(np.max(np.abs(convolved))), 1e-8)
    source_peak = max(float(np.max(np.abs(clean_float))), 1e-8)
    convolved = convolved / peak * min(source_peak, 0.95)
    output = (convolved * (clean_scale - 1)).astype("<i2")

    return AudioSegment(
        output.tobytes(),
        frame_rate=clean.frame_rate,
        sample_width=DEFAULT_SAMPLE_WIDTH,
        channels=1,
    )


def _apply_dsp_off_axis(sound, angle_deg, distance_m):
    processed = _ensure_benchmark_format(sound)

    attenuation_db = -6.0
    if angle_deg >= 135:
        attenuation_db = -12.0
    elif angle_deg >= 90:
        attenuation_db = -9.0
    elif angle_deg >= 45:
        attenuation_db = -6.0

    distance_loss_db = max(distance_m - 1.0, 0.0) * -2.0
    processed = processed.apply_gain(attenuation_db + distance_loss_db)
    processed = processed.high_pass_filter(120)
    processed = processed.low_pass_filter(3600 if angle_deg >= 90 else 4800)

    # A tiny room smear gives the side-addressed speech a less direct close-mic feel.
    early_reflection = processed.apply_gain(-13)
    late_reflection = processed.apply_gain(-18)
    processed = processed.overlay(early_reflection, position=35)
    processed = processed.overlay(late_reflection, position=80)

    return processed[: len(sound)]


def apply_off_axis_effect(sound, rir_path=None, angle_deg=90, distance_m=1.5):
    """
    Simulate speech addressed away from the microphone.

    If a real off-axis RIR is supplied, convolve speech with it. Otherwise, use a
    deterministic DSP fallback that attenuates level/high frequencies and adds a
    small room smear.
    """
    if rir_path:
        resolved_rir = Path(rir_path)
        if resolved_rir.exists():
            rir_sound = AudioSegment.from_file(resolved_rir)
            processed = _apply_rir_convolution(sound, rir_sound)
            return processed[: len(sound)], {
                "backend": "rir_convolution",
                "rir_path": str(resolved_rir),
                "angle_deg": angle_deg,
                "distance_m": distance_m,
            }

    processed = _apply_dsp_off_axis(sound, angle_deg=angle_deg, distance_m=distance_m)
    return processed, {
        "backend": "dsp_fallback",
        "angle_deg": angle_deg,
        "distance_m": distance_m,
        "filters": {
            "high_pass_hz": 120,
            "low_pass_hz": 3600 if angle_deg >= 90 else 4800,
        },
    }
