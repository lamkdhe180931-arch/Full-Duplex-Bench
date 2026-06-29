import os
import json
import argparse
from glob import glob
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

MODEL_NAME = ""
ASR_MODEL_ID = "vinai/PhoWhisper-medium"
_ASR_PIPELINE = None


def disable_safetensors_auto_conversion():
    """
    Transformers can start a background safetensors conversion check that calls
    the Hugging Face Discussions API. PhoWhisper has Discussions disabled, so
    that thread raises 403 even though the PyTorch weights are usable.
    """
    try:
        import transformers.safetensors_conversion as safetensors_conversion
    except Exception:
        return

    def _skip_auto_conversion(*args, **kwargs):
        return None

    safetensors_conversion.auto_conversion = _skip_auto_conversion


def get_asr_pipeline():
    global _ASR_PIPELINE
    if _ASR_PIPELINE is not None:
        return _ASR_PIPELINE

    device = "cpu"
    if torch.cuda.is_available():
        # Dùng GPU số 2 (cuda:1) nếu có 2 GPU để tránh tranh chấp bộ nhớ với GPU số 1 (cuda:0)
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

    dtype = torch.float16 if "cuda" in device else torch.float32
    hf_token = os.getenv("HF_TOKEN") or None
    disable_safetensors_auto_conversion()

    # Load explicitly from PyTorch weights. Letting pipeline resolve the model id
    # can trigger Transformers' background safetensors auto-conversion check,
    # which calls the Hugging Face Discussions API. PhoWhisper has Discussions
    # disabled, so that background thread raises a noisy 403 and can stall
    # notebook runs.
    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID, token=hf_token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_ID,
        dtype=dtype,
        use_safetensors=False,
        token=hf_token,
    )

    _ASR_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        dtype=dtype,
        device=device,
    )
    return _ASR_PIPELINE


def get_time_aligned_transcription(data_path, task, audio_name="output.wav"):
    # Collect all matching audio files under the root directory
    audio_paths = sorted(glob(f"{data_path}/*/{MODEL_NAME}{audio_name}"))

    # JSON output filename mirrors the audio filename (e.g. clean_input.wav -> clean_input.json)
    json_name = audio_name.rsplit(".", 1)[0] + ".json"
    pipe = get_asr_pipeline()

    for audio_path in tqdm(audio_paths):
        print(audio_path)
        # Read the audio file (waveform and sample rate)
        waveform, sr = sf.read(audio_path)
        # If multichannel audio, convert to mono by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Default offset maps output.wav timestamps back to the original input timeline.
        offset = 0.0
        timing_path = os.path.join(os.path.dirname(audio_path), "inference_timing.json")
        if os.path.exists(timing_path):
            with open(timing_path, "r", encoding="utf-8") as f:
                offset = json.load(f).get("response_start_sec") or 0.0

        interrupt_end_time = None
        if task == "user_interruption":
            meta_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", "interrupt.json")
            if not os.path.exists(meta_path):
                meta_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", "metadata.json")

            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Neither interrupt.json nor metadata.json found in {os.path.dirname(audio_path)}")

            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

            if isinstance(meta_data, list):
                _, interrupt_end_time = meta_data[0]["timestamp"]
            else:
                _, interrupt_end_time = meta_data["timestamps"]

        # Legacy synchronized output.wav included the full input timeline, so ASR
        # cropped before the interrupt end. Dynamic output.wav contains only the
        # Gemini response, so inference_timing.json supplies the timeline offset.
        if task == "user_interruption" and not os.path.exists(timing_path):
            offset = interrupt_end_time

            # Compute the sample index to start from, and crop the waveform
            start_idx = int(interrupt_end_time * sr)
            waveform = waveform[start_idx:]

        # Crop trailing silence (padded as absolute zeros) to prevent Whisper hallucinating loops/unk/a.
        non_zero_indices = np.where(np.abs(waveform) > 1e-4)[0]
        if len(non_zero_indices) > 0:
            last_active_idx = non_zero_indices[-1]
            # Add 0.5 seconds of safety padding to ensure the final word is fully captured
            padding_samples = int(0.5 * sr)
            end_idx = min(len(waveform), last_active_idx + padding_samples)
            waveform = waveform[:end_idx]
        else:
            result_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", json_name)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f:
                json.dump({"text": "", "chunks": []}, f, indent=4)
            continue

        import tempfile

        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
                sf.write(tmp.name, waveform, sr)
                prediction = pipe(
                    tmp.name,
                    return_timestamps="word",
                    generate_kwargs={"language": "vietnamese"}
                )
        except ValueError as exc:
            print(f"[WARN] ASR skipped malformed or empty audio: {audio_path}: {exc}")
            result_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", json_name)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f:
                json.dump({"text": "", "chunks": []}, f, indent=4)
            continue
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)

        # Build the output dict, adjusting each timestamp by the offset
        chunks = []
        text = ""
        for chunk in prediction.get("chunks", []):
            word = chunk["text"]
            word_clean = word.strip()
            if not word_clean:
                continue
            if chunk["timestamp"] is None:
                continue
            
            start_time = chunk["timestamp"][0] + offset
            end_time = chunk["timestamp"][1] + offset

            if interrupt_end_time is not None and start_time < interrupt_end_time:
                continue

            text += word_clean + " "
            chunks.append(
                {
                    "text": word_clean,
                    "timestamp": [start_time, end_time],
                }
            )

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        # Write the JSON result next to the WAV file
        result_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", json_name)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(output_dict, f, indent=4)

        # Free GPU memory cache to prevent Out Of Memory
        import gc
        del prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def transcribe_v1_benchmark(root_dir, audio_name="output.wav"):
    tasks = [
        ("synthetic_pause_handling", "default"),
        ("candor_turn_taking", "default"),
        ("synthetic_user_interruption", "user_interruption"),
    ]
    for task_name, task_mode in tasks:
        task_dir = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_dir):
            continue
        json_name = audio_name.rsplit(".", 1)[0] + ".json"
        print(f"ASR {audio_name} -> {json_name}: {task_name}")
        get_time_aligned_transcription(task_dir, task_mode, audio_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe full audio or only after a user interruption"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root folder containing subfolders with output.wav (and interrupt.json)",
    )
    parser.add_argument(
        "--v1_benchmark_root",
        type=str,
        default=None,
        help="Root folder containing v1 task folders. Loads PhoWhisper once and transcribes all v1 tasks.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="default",
        choices=["default", "user_interruption"],
        help="Choose 'default' for entire transcript or 'user_interruption' to crop before ASR.",
    )
    parser.add_argument(
        "--audio_name",
        type=str,
        default="output.wav",
        help="Filename of the audio to transcribe within each sample folder. "
             "Use 'clean_input.wav', 'clean_output.wav', or 'input.wav' for v1.5. "
             "JSON output filename mirrors this (e.g. 'clean_output.wav' -> 'clean_output.json').",
    )
    args = parser.parse_args()

    if args.v1_benchmark_root:
        transcribe_v1_benchmark(args.v1_benchmark_root, args.audio_name)
    else:
        if not args.root_dir:
            raise ValueError("--root_dir is required unless --v1_benchmark_root is set.")
        get_time_aligned_transcription(args.root_dir, args.task, args.audio_name)
