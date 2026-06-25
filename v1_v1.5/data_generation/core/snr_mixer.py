import math

from pydub import AudioSegment


def _rms_db(sound):
    if sound.rms <= 0:
        return float("-inf")
    return 20.0 * math.log10(sound.rms)


def _fit_duration(sound, duration_ms):
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive.")
    if len(sound) == 0:
        raise ValueError("background audio must not be empty.")

    if len(sound) < duration_ms:
        repeats = math.ceil(duration_ms / len(sound))
        sound = sound * repeats
    return sound[:duration_ms]


def measure_snr_db(speech, background):
    if speech.rms <= 0:
        raise ValueError("speech reference is silent; cannot measure SNR.")
    if background.rms <= 0:
        raise ValueError("background audio is silent; cannot measure SNR.")
    return _rms_db(speech) - _rms_db(background)


def match_background_to_snr(speech_reference, background, target_snr_db):
    speech_reference = speech_reference.set_channels(1)
    background = (
        background
        .set_frame_rate(speech_reference.frame_rate)
        .set_channels(speech_reference.channels)
        .set_sample_width(speech_reference.sample_width)
    )

    current_snr_db = measure_snr_db(speech_reference, background)
    gain_db = current_snr_db - target_snr_db
    scaled_background = background.apply_gain(gain_db)
    actual_snr_db = measure_snr_db(speech_reference, scaled_background)

    return scaled_background, {
        "target_db": target_snr_db,
        "actual_db": actual_snr_db,
        "background_gain_db": gain_db,
        "speech_rms_db": _rms_db(speech_reference),
        "background_rms_db": _rms_db(scaled_background),
    }


def mix_background_at_snr(
    clean,
    background,
    position_ms,
    duration_ms=None,
    target_snr_db=10,
    reference_sound=None,
):
    duration_ms = duration_ms or len(background)
    fitted_background = _fit_duration(background, duration_ms)
    speech_reference = reference_sound or clean
    scaled_background, metadata = match_background_to_snr(
        speech_reference,
        fitted_background,
        target_snr_db=target_snr_db,
    )

    end_ms = position_ms + len(scaled_background)
    if end_ms > len(clean):
        clean = clean + AudioSegment.silent(
            duration=end_ms - len(clean),
            frame_rate=clean.frame_rate,
        )

    mixed = clean.overlay(scaled_background, position=position_ms)
    peak_reduction_db = 0.0
    if mixed.max_dBFS > -1.0:
        peak_reduction_db = -1.0 - mixed.max_dBFS
        mixed = mixed.apply_gain(peak_reduction_db)

    metadata.update(
        {
            "position_ms": position_ms,
            "duration_ms": len(scaled_background),
            "peak_reduction_db": peak_reduction_db,
        }
    )
    return mixed, metadata
