import json
import asyncio
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine


DATA_GENERATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_GENERATION_DIR))
sys.path.insert(0, str(DATA_GENERATION_DIR / "v1_5"))

from core.tts_generator import VietnameseTTSGenerator
from v1_5.generate_v1_5 import (
    ASSET_REQUIRED_SCENARIOS,
    BACKGROUND_SPEECH_SCENARIO,
    TALKING_TO_OTHER_SCENARIO,
    TTS_OVERLAP_SCENARIOS,
    generate_background_speech,
    generate_interruption_and_backchannel,
    generate_talking_to_other,
)


def test_edge_tts_profile_options_are_passed_to_provider(monkeypatch, tmp_path):
    generator = VietnameseTTSGenerator(provider="edge-tts", seed=7)
    captured = {}

    async def fake_edge_tts(text, output_path, voice, rate="+0%", volume="+0%", pitch="+0Hz"):
        captured.update(
            {
                "text": text,
                "voice": voice,
                "rate": rate,
                "volume": volume,
                "pitch": pitch,
            }
        )
        Path(output_path).write_bytes(b"placeholder")

    def fake_convert(input_path, output_path):
        Path(output_path).write_bytes(b"wav")

    monkeypatch.setattr(generator, "_generate_edge_tts", fake_edge_tts)
    monkeypatch.setattr(generator, "_convert_to_benchmark_wav", fake_convert)

    profile = {
        "voice": "vi-VN-NamMinhNeural",
        "rate": "+8%",
        "volume": "-3%",
        "pitch": "+12Hz",
    }

    generator.generate("Xin chao", tmp_path / "out.wav", profile=profile)

    assert captured == {
        "text": "Xin chao",
        "voice": "vi-VN-NamMinhNeural",
        "rate": "+8%",
        "volume": "-3%",
        "pitch": "+12Hz",
    }
    assert generator.last_synthesis["profile"] == profile


def test_edge_tts_retries_transient_no_audio_response(monkeypatch, tmp_path):
    class NoAudioReceived(Exception):
        pass

    generator = VietnameseTTSGenerator(provider="edge-tts", seed=7, max_retries=1)
    attempts = []

    async def flaky_edge_tts(text, output_path, voice, rate="+0%", volume="+0%", pitch="+0Hz"):
        attempts.append(output_path)
        if len(attempts) == 1:
            raise NoAudioReceived("No audio was received")
        Path(output_path).write_bytes(b"placeholder")

    def fake_convert(input_path, output_path):
        Path(output_path).write_bytes(b"wav")

    monkeypatch.setattr(generator, "_generate_edge_tts", flaky_edge_tts)
    monkeypatch.setattr(generator, "_convert_to_benchmark_wav", fake_convert)

    generator.generate("Xin chao", tmp_path / "out.wav")

    assert len(attempts) == 2


def test_edge_tts_generate_works_inside_running_event_loop(monkeypatch, tmp_path):
    generator = VietnameseTTSGenerator(provider="edge-tts", seed=7)
    captured = {}

    async def fake_edge_tts(text, output_path, voice, rate="+0%", volume="+0%", pitch="+0Hz"):
        captured["text"] = text
        Path(output_path).write_bytes(b"placeholder")

    def fake_convert(input_path, output_path):
        Path(output_path).write_bytes(b"wav")

    monkeypatch.setattr(generator, "_generate_edge_tts", fake_edge_tts)
    monkeypatch.setattr(generator, "_convert_to_benchmark_wav", fake_convert)

    async def run_from_notebook_like_loop():
        return generator.generate("Xin chao trong notebook", tmp_path / "out.wav")

    asyncio.run(run_from_notebook_like_loop())

    assert captured["text"] == "Xin chao trong notebook"
    assert (tmp_path / "out.wav").exists()


def test_tts_profile_selection_is_deterministic_by_seed():
    first = VietnameseTTSGenerator(provider="edge-tts", seed=123)
    second = VietnameseTTSGenerator(provider="edge-tts", seed=123)

    first_choices = [first.select_profile("background_speech") for _ in range(4)]
    second_choices = [second.select_profile("background_speech") for _ in range(4)]

    assert first_choices == second_choices
    assert len({choice["voice"] for choice in first_choices}) >= 1


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.last_synthesis = {}

    def generate(self, text, output_path, role="primary", profile=None, voice=None):
        self.calls.append({"text": text, "role": role, "profile": profile})
        selected_voice = voice or f"fake-{role}"
        self.last_synthesis = {
            "provider": "fake",
            "role": role,
            "profile": {
                "voice": selected_voice,
                "rate": "+0%",
                "volume": "+0%",
                "pitch": "+0Hz",
            },
        }
        Sine(440).to_audio_segment(duration=250).set_frame_rate(16000).set_channels(1).export(
            output_path,
            format="wav",
        )
        return output_path


class FakeMixer:
    @staticmethod
    def load_audio(path):
        return AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1)

    @staticmethod
    def save_audio(sound, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sound.export(output_path, format="wav", codec="pcm_s16le")


class FakeBackgroundGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, output_path, duration_sec, seed=None, provider=None):
        self.calls.append(
            {
                "prompt": prompt,
                "output_path": str(output_path),
                "duration_sec": duration_sec,
                "seed": seed,
                "provider": provider,
            }
        )
        tone = Sine(880).to_audio_segment(duration=int(duration_sec * 1000)).set_frame_rate(16000)
        tone = tone.set_channels(1).apply_gain(-12)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tone.export(output_path, format="wav", codec="pcm_s16le")
        return {
            "provider": provider or "fake_text_to_audio",
            "prompt": prompt,
            "duration_sec": duration_sec,
            "seed": seed,
            "model_id": "fake-sfx",
        }


def test_v1_5_generation_records_tts_profiles_in_metadata(tmp_path):
    templates = [
        {
            "id": "sample_001",
            "context_text": "Tư vấn giúp tôi một nhà hàng gần đây.",
            "current_turn_text": "À khoan, ưu tiên quán yên tĩnh nhé.",
            "delay_sec": 0.5,
        }
    ]

    generate_interruption_and_backchannel(
        FakeGenerator(),
        FakeMixer(),
        templates,
        "user_interruption",
        tmp_path,
    )

    metadata_path = tmp_path / "user_interruption" / "sample_001" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["tts_profiles"]["context"]["role"] == "primary"
    assert metadata["tts_profiles"]["overlap"]["role"] == "user_interruption"
    assert metadata["tts_profiles"]["overlap"]["profile"]["voice"] == "fake-primary"


def test_interruption_generation_keeps_same_user_voice(tmp_path):
    templates = [
        {
            "id": "sample_002",
            "context_text": "Tìm giúp tôi một khách sạn gần hồ.",
            "current_turn_text": "À thêm điều kiện có bữa sáng nhé.",
            "delay_sec": 0.5,
        }
    ]

    generate_interruption_and_backchannel(
        FakeGenerator(),
        FakeMixer(),
        templates,
        "user_interruption",
        tmp_path,
    )

    metadata_path = tmp_path / "user_interruption" / "sample_002" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tts_profiles = metadata["tts_profiles"]

    assert tts_profiles["context"]["profile"]["voice"] == tts_profiles["overlap"]["profile"]["voice"]
    assert tts_profiles["context"]["role"] == "primary"
    assert tts_profiles["overlap"]["role"] == "user_interruption"


def test_interruption_metadata_marks_clear_tts_insert_strategy(tmp_path):
    templates = [
        {
            "id": "sample_003",
            "context_text": "Tìm chuyến bay đi Đà Nẵng chiều nay.",
            "current_turn_text": "À khoan, chỉ tìm chuyến sau sáu giờ tối.",
            "delay_sec": 0.75,
        }
    ]

    generate_interruption_and_backchannel(
        FakeGenerator(),
        FakeMixer(),
        templates,
        "user_interruption",
        tmp_path,
    )

    metadata_path = tmp_path / "user_interruption" / "sample_003" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["simulation"]["strategy"] == "tts_delayed_same_speaker"
    assert metadata["simulation"]["same_speaker"] is True
    assert metadata["simulation"]["overlap_gain_db"] == 0
    assert metadata["simulation"]["agent_response_window_sec"] == 15


def test_backchannel_metadata_marks_quiet_short_tts_insert(tmp_path):
    templates = [
        {
            "id": "sample_004",
            "context_text": "Giải thích giúp tôi cách đăng ký thẻ online.",
            "current_turn_text": "Dạ vâng",
            "delay_sec": 0.5,
        }
    ]

    generate_interruption_and_backchannel(
        FakeGenerator(),
        FakeMixer(),
        templates,
        "user_backchannel",
        tmp_path,
    )

    metadata_path = tmp_path / "user_backchannel" / "sample_004" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tts_profiles = metadata["tts_profiles"]

    assert metadata["simulation"]["strategy"] == "tts_quiet_same_speaker_backchannel"
    assert metadata["simulation"]["same_speaker"] is True
    assert metadata["simulation"]["overlap_gain_db"] < 0
    assert tts_profiles["context"]["profile"]["voice"] == tts_profiles["overlap"]["profile"]["voice"]


def test_tts_overlap_and_specialized_scenarios_are_declared():
    assert set(TTS_OVERLAP_SCENARIOS) == {"user_interruption", "user_backchannel"}
    assert TALKING_TO_OTHER_SCENARIO["strategy"] == "tts_off_axis_simulation"
    assert TALKING_TO_OTHER_SCENARIO["addressed_to_agent"] is False
    assert BACKGROUND_SPEECH_SCENARIO["strategy"] == "generated_background_snr_mix"
    assert BACKGROUND_SPEECH_SCENARIO["default_provider"] in {
        "elevenlabs_sfx",
        "audiogen_cli",
        "audioldm2_cli",
    }
    assert ASSET_REQUIRED_SCENARIOS == {}


def test_talking_to_other_generation_marks_off_axis_metadata(tmp_path):
    templates = [
        {
            "id": "talk_other_001",
            "context_text": "",
            "current_turn_text": "Tìm chuyến bay từ Hà Nội đi Đà Nẵng chiều nay.",
            "overlap_text": "Lấy cho anh cốc nước lọc nhé em.",
            "delay_sec": 0.5,
        }
    ]

    generate_talking_to_other(
        FakeGenerator(),
        FakeMixer(),
        templates,
        tmp_path,
    )

    metadata_path = tmp_path / "talking_to_other" / "talk_other_001" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["simulation"]["strategy"] == "tts_off_axis_simulation"
    assert metadata["simulation"]["off_axis"] is True
    assert metadata["simulation"]["addressed_to_agent"] is False
    assert metadata["simulation"]["angle_deg"] == 90
    assert metadata["tts_profiles"]["clean_input"]["role"] == "primary"
    assert metadata["tts_profiles"]["overlap"]["role"] == "talking_to_other"
    assert metadata["audio_effect"]["backend"] in {"rir_convolution", "dsp_fallback"}


def test_background_speech_uses_generated_noise_and_snr_metadata(tmp_path):
    templates = [
        {
            "id": "bg_speech_001",
            "context_text": "",
            "current_turn_text": "Hôm nay giá vàng trong nước có biến động gì nhiều không bạn?",
            "background_prompt": "muffled Vietnamese TV news speech from another room",
            "delay_sec": 0.5,
            "background_duration_sec": 1.0,
            "target_snr_db": 8,
            "background_provider": "elevenlabs_sfx",
        }
    ]
    background_generator = FakeBackgroundGenerator()

    generate_background_speech(
        FakeGenerator(),
        FakeMixer(),
        templates,
        tmp_path,
        background_generator=background_generator,
    )

    metadata_path = tmp_path / "background_speech" / "bg_speech_001" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert background_generator.calls[0]["provider"] == "elevenlabs_sfx"
    assert metadata["simulation"]["strategy"] == "generated_background_snr_mix"
    assert metadata["background_generation"]["provider"] == "elevenlabs_sfx"
    assert metadata["background_generation"]["prompt"] == templates[0]["background_prompt"]
    assert metadata["snr"]["target_db"] == 8
    assert abs(metadata["snr"]["actual_db"] - 8) <= 1.0
    assert metadata["timestamps"] == [0.75, 1.75]
    assert (tmp_path / "background_speech" / "bg_speech_001" / "input.wav").exists()
