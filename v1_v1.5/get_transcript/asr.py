import os
import json
import argparse
from glob import glob
import torch
import soundfile as sf
from transformers import pipeline
from tqdm import tqdm

MODEL_NAME = ""


def get_time_aligned_transcription(data_path, task, audio_name="output.wav"):
    # Collect all matching audio files under the root directory
    audio_paths = sorted(glob(f"{data_path}/*/{MODEL_NAME}{audio_name}"))

    # JSON output filename mirrors the audio filename (e.g. clean_input.wav -> clean_input.json)
    json_name = audio_name.rsplit(".", 1)[0] + ".json"

    # Load the pretrained PhoWhisper model and move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model="vinai/PhoWhisper-medium",
        chunk_length_s=30,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
    )

    for audio_path in tqdm(audio_paths):
        print(audio_path)
        # Read the audio file (waveform and sample rate)
        waveform, sr = sf.read(audio_path)
        # If multichannel audio, convert to mono by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Default offset is zero (no cropping)
        offset = 0.0

        if task == "user_interruption":
            # Load the interrupt metadata to get [start, end] timestamps
            meta_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", "interrupt.json")
            with open(meta_path, "r") as f:
                interrupt_meta = json.load(f)

            # We only care about the end of the interruption
            _, end_interrupt = interrupt_meta[0]["timestamp"]
            offset = end_interrupt

            # Compute the sample index to start from, and crop the waveform
            start_idx = int(end_interrupt * sr)
            waveform = waveform[start_idx:]

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            prediction = pipe(tmp.name, return_timestamps="word", generate_kwargs={"language": "vietnamese"})
        # remove the temp file so you don't leak disk
        os.unlink(tmp.name)

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
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe full audio or only after a user interruption"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root folder containing subfolders with output.wav (and interrupt.json)",
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

    get_time_aligned_transcription(args.root_dir, args.task, args.audio_name)
