import sys
from pathlib import Path

from pydub.generators import Sine


DATA_GENERATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_GENERATION_DIR))

from core.snr_mixer import match_background_to_snr, mix_background_at_snr


def test_match_background_to_snr_scales_noise_to_target_ratio():
    speech = Sine(440).to_audio_segment(duration=1000).apply_gain(-8)
    background = Sine(1000).to_audio_segment(duration=1000).apply_gain(-4)

    scaled_background, metadata = match_background_to_snr(
        speech,
        background,
        target_snr_db=10,
    )

    assert metadata["target_db"] == 10
    assert abs(metadata["actual_db"] - 10) <= 0.5
    assert scaled_background.dBFS < background.dBFS


def test_mix_background_at_snr_extends_clean_audio_and_records_timing():
    clean = Sine(440).to_audio_segment(duration=1000).apply_gain(-8)
    background = Sine(1000).to_audio_segment(duration=400).apply_gain(-10)

    mixed, metadata = mix_background_at_snr(
        clean,
        background,
        position_ms=800,
        duration_ms=700,
        target_snr_db=6,
    )

    assert len(mixed) == 1500
    assert metadata["position_ms"] == 800
    assert metadata["duration_ms"] == 700
    assert abs(metadata["actual_db"] - 6) <= 0.75
