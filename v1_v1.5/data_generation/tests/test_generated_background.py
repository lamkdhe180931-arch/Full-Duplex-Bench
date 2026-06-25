import sys
from pathlib import Path

import pytest
from pydub import AudioSegment


DATA_GENERATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_GENERATION_DIR))

from core.generated_background import GeneratedBackgroundGenerator


def test_elevenlabs_sfx_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("XI_API_KEY", raising=False)
    generator = GeneratedBackgroundGenerator(provider="elevenlabs_sfx")

    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        generator.generate(
            "busy cafe with indistinct Vietnamese speech",
            tmp_path / "background.wav",
            duration_sec=1,
        )


def test_audiogen_cli_command_generates_benchmark_wav(monkeypatch, tmp_path):
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "background.wav"
    AudioSegment.silent(duration=500, frame_rate=44100).set_channels(2).export(
        source_path,
        format="wav",
    )
    monkeypatch.setenv("AUDIOGEN_CMD", f"cp {source_path} {{output}}")
    generator = GeneratedBackgroundGenerator(provider="audiogen_cli")

    metadata = generator.generate(
        "muffled TV news from another room",
        output_path,
        duration_sec=0.5,
        seed=42,
    )

    sound = AudioSegment.from_file(output_path)
    assert sound.frame_rate == 16000
    assert sound.channels == 1
    assert metadata["provider"] == "audiogen_cli"
    assert metadata["command_env"] == "AUDIOGEN_CMD"
