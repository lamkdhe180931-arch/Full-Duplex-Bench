import json
import math
import os
import struct
import sys
import tempfile
import unittest
import wave


EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from v1_timeline_metrics import (  # noqa: E402
    evaluate_pause_sample,
    evaluate_turn_taking_sample,
    evaluate_user_interruption_sample,
)


SR = 16000


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def write_wav(path, duration_sec, speech_windows):
    total = int(round(duration_sec * SR))
    frames = []
    for i in range(total):
        t = i / SR
        active = any(start <= t < end for start, end in speech_windows)
        if active:
            value = int(12000 * math.sin(2 * math.pi * 440 * t))
        else:
            value = 0
        frames.append(struct.pack("<h", value))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"".join(frames))


class V1TimelineMetricsTest(unittest.TestCase):
    def test_turn_taking_uses_turn_end_from_timestamp_zero_and_audio_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_wav(os.path.join(tmp, "input.wav"), 3.0, [(0.0, 3.0)])
            write_wav(os.path.join(tmp, "output.wav"), 4.5, [(3.4, 4.0)])
            write_json(os.path.join(tmp, "turn_taking.json"), [{"timestamp": [3.0, 0.0]}])
            write_json(
                os.path.join(tmp, "output.json"),
                {"text": "fake", "chunks": [{"text": "fake", "timestamp": [0.0, 0.2]}]},
            )

            result = evaluate_turn_taking_sample(tmp)

        self.assertEqual(result["response_rate"], 1)
        self.assertEqual(result["barge_in"], 0)
        self.assertAlmostEqual(result["valid_response_latency"], 0.4, delta=0.08)
        self.assertEqual(result["smooth_turn_success"], 1)

    def test_pause_separates_pause_barge_in_from_continuation_barge_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_wav(os.path.join(tmp, "input.wav"), 5.0, [(0.0, 1.0), (2.0, 5.0)])
            write_wav(os.path.join(tmp, "output.wav"), 5.8, [(2.4, 2.8), (5.3, 5.7)])
            write_json(os.path.join(tmp, "pause.json"), [{"timestamp": [1.0, 2.0]}])
            write_json(os.path.join(tmp, "output.json"), {"text": "ok", "chunks": []})

            result = evaluate_pause_sample(tmp)

        self.assertEqual(result["pause_barge_in"], 0)
        self.assertEqual(result["continuation_barge_in"], 1)
        self.assertEqual(result["listen_through_success"], 0)
        self.assertAlmostEqual(result["valid_response_latency"], 0.3, delta=0.08)

    def test_user_interruption_measures_stop_and_recovery_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_wav(os.path.join(tmp, "input.wav"), 6.0, [(0.0, 1.5), (4.0, 5.0)])
            write_wav(os.path.join(tmp, "output.wav"), 6.2, [(3.0, 4.3), (5.25, 6.0)])
            write_json(
                os.path.join(tmp, "interrupt.json"),
                [{"context": "old", "interrupt": "new", "timestamp": [4.0, 5.0]}],
            )
            write_json(
                os.path.join(tmp, "output.json"),
                {
                    "text": "old then new",
                    "chunks": [
                        {"text": "old", "timestamp": [3.2, 3.6]},
                        {"text": "new", "timestamp": [5.3, 5.7]},
                    ],
                },
            )

            result = evaluate_user_interruption_sample(tmp)

        self.assertEqual(result["response_rate"], 1)
        self.assertAlmostEqual(result["stop_latency"], 0.3, delta=0.08)
        self.assertAlmostEqual(result["interrupt_overlap_duration"], 0.3, delta=0.08)
        self.assertEqual(result["listening_success"], 1)
        self.assertAlmostEqual(result["recovery_latency"], 0.25, delta=0.08)
        self.assertEqual(result["post_interrupt_response_rate"], 1)
        self.assertEqual(result["pre_interrupt_text"], "old")
        self.assertEqual(result["post_interrupt_text"], "new")


if __name__ == "__main__":
    unittest.main()
