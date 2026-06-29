import importlib.util
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path


def load_gemini_module():
    module_path = Path(__file__).resolve().parents[1] / "timeline_audio.py"
    spec = importlib.util.spec_from_file_location("timeline_audio", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_constant_wav(path, sample_rate, seconds, value):
    frames = struct.pack("<" + "h" * (sample_rate * seconds), *([value] * (sample_rate * seconds)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


def read_wav_samples(path):
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    return sample_rate, samples


class TimelineAudioTest(unittest.TestCase):
    def test_align_response_stem_prepends_silence_to_match_conversation_timeline(self):
        module = load_gemini_module()
        sample_rate = 24000
        response_start_sec = 2.0

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_response = tmp_path / "raw_output.wav"
            aligned_response = tmp_path / "output.wav"

            write_constant_wav(raw_response, sample_rate, seconds=1, value=12000)

            module.align_response_stem_to_timeline(
                str(raw_response),
                str(aligned_response),
                response_start_sec=response_start_sec,
                sample_rate=sample_rate,
            )

            sr, aligned = read_wav_samples(aligned_response)
            self.assertEqual(sr, sample_rate)
            self.assertEqual(len(aligned), sample_rate * 3)
            self.assertEqual(max(abs(x) for x in aligned[: sample_rate * 2]), 0)
            self.assertGreater(max(abs(x) for x in aligned[sample_rate * 2 :]), 10000)

    def test_align_response_stem_keeps_at_least_input_duration(self):
        module = load_gemini_module()
        sample_rate = 24000

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_response = tmp_path / "raw_output.wav"
            aligned_response = tmp_path / "output.wav"

            write_constant_wav(raw_response, sample_rate, seconds=1, value=12000)

            module.align_response_stem_to_timeline(
                str(raw_response),
                str(aligned_response),
                response_start_sec=0.5,
                sample_rate=sample_rate,
                min_duration_sec=3.0,
            )

            sr, aligned = read_wav_samples(aligned_response)
            self.assertEqual(sr, sample_rate)
            self.assertEqual(len(aligned), sample_rate * 3)
            self.assertGreater(max(abs(x) for x in aligned[sample_rate // 2 : sample_rate]), 10000)
            self.assertEqual(max(abs(x) for x in aligned[sample_rate * 2 :]), 0)

    def test_pad_timeline_wav_extends_short_output_without_append_mode(self):
        module = load_gemini_module()
        sample_rate = 24000

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output.wav"
            write_constant_wav(output, sample_rate, seconds=1, value=12000)

            duration = module.pad_timeline_wav_to_min_duration(
                str(output),
                sample_rate=sample_rate,
                min_duration_sec=3.0,
            )

            sr, padded = read_wav_samples(output)
            self.assertEqual(sr, sample_rate)
            self.assertEqual(duration, 3.0)
            self.assertEqual(len(padded), sample_rate * 3)
            self.assertGreater(max(abs(x) for x in padded[:sample_rate]), 10000)
            self.assertEqual(max(abs(x) for x in padded[sample_rate:]), 0)


if __name__ == "__main__":
    unittest.main()
