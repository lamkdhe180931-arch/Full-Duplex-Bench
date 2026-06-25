import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine


DATA_GENERATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_GENERATION_DIR))

from core.off_axis_effects import apply_off_axis_effect


def test_off_axis_dsp_fallback_keeps_duration_and_reduces_level():
    speech = Sine(1000).to_audio_segment(duration=1000).set_frame_rate(16000).set_channels(1)

    processed, metadata = apply_off_axis_effect(speech, angle_deg=90, distance_m=1.5)

    assert abs(len(processed) - len(speech)) <= 1
    assert processed.dBFS < speech.dBFS
    assert metadata["backend"] in {"dsp_fallback", "pyroomacoustics"}

    assert metadata["angle_deg"] == 90
    assert metadata["distance_m"] == 1.5


def test_off_axis_uses_rir_convolution_when_rir_path_exists(tmp_path):
    speech = Sine(440).to_audio_segment(duration=500).set_frame_rate(16000).set_channels(1)
    rir_path = tmp_path / "rir.wav"
    impulse = AudioSegment.silent(duration=80, frame_rate=16000)
    impulse = impulse.overlay(Sine(1000).to_audio_segment(duration=5).apply_gain(-3), position=0)
    impulse.export(rir_path, format="wav")

    processed, metadata = apply_off_axis_effect(speech, rir_path=rir_path, angle_deg=90)

    assert abs(len(processed) - len(speech)) <= 1
    assert metadata["backend"] == "rir_convolution"
    assert metadata["rir_path"] == str(rir_path)
