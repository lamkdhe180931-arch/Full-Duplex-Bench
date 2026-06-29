import os
import json
import argparse
from glob import glob
import torch
import soundfile as sf
import numpy as np
import tempfile
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

MODEL_NAME = ""

def disable_safetensors_auto_conversion():
    try:
        import transformers.safetensors_conversion as safetensors_conversion
        safetensors_conversion.auto_conversion = lambda *args, **kwargs: None
    except Exception:
        pass

def init_gemini():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("[WARN] GEMINI_API_KEY not found. Fallback to model 1 (PhoWhisper).")
        return None
    return genai.Client(api_key=gemini_api_key)

def merge_with_gemini(client, t1, t2, t3):
    if not client or (not t1 and not t2 and not t3):
        return 1
        
    system_msg = """Dưới đây là 3 kết quả nhận diện giọng nói (ASR) của cùng 1 đoạn audio tiếng Việt. 
Hãy chọn ra 1 kết quả chính xác nhất, tự nhiên nhất và KHÔNG bị lặp từ (hallucination). 
CHỈ trả về duy nhất số thứ tự của kết quả tốt nhất: 1, 2, hoặc 3. Không giải thích gì thêm."""
    user_msg = f"Kết quả 1: {t1}\nKết quả 2: {t2}\nKết quả 3: {t3}"
    
    try:
        time.sleep(2) # Rate limit protection
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{system_msg}\n\n{user_msg}"
        )
        text = response.text.strip()
        if "1" in text: return 1
        if "2" in text: return 2
        if "3" in text: return 3
        return 1
    except Exception as e:
        print(f"[WARN] Gemini selection failed: {e}. Defaulting to 1.")
        return 1

def format_whisper_chunks(prediction, offset):
    chunks = []
    text = ""
    for chunk in prediction.get("chunks", []):
        word = chunk["text"].strip()
        if not word or chunk["timestamp"] is None:
            continue
        start_time = chunk["timestamp"][0] + offset
        end_time = chunk["timestamp"][1] + offset
        text += word + " "
        chunks.append({"text": word, "timestamp": [start_time, end_time]})
    return {"text": text.strip(), "chunks": chunks}

def format_chunkformer_chunks(prediction, offset):
    chunks = []
    text = ""
    raw_chunks = prediction if isinstance(prediction, list) else prediction.get("chunks", [])
    for chunk in raw_chunks:
        word = chunk.get("text", chunk.get("word", "")).strip()
        if not word: continue
        ts = chunk.get("timestamp", [chunk.get("start"), chunk.get("end")])
        if ts[0] is None or ts[1] is None:
            continue
        start_time = ts[0] + offset
        end_time = ts[1] + offset
        text += word + " "
        chunks.append({"text": word, "timestamp": [start_time, end_time]})
    return {"text": text.strip(), "chunks": chunks}

def load_models():
    """Load all 3 ASR models once. Returns a dict with the loaded pipelines/models."""
    print("Loading models into GPUs...")
    disable_safetensors_auto_conversion()

    device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    device1 = "cuda:1" if torch.cuda.device_count() > 1 else device0

    print(f"Loading PhoWhisper-large -> {device0}")
    pho_processor = AutoProcessor.from_pretrained("vinai/PhoWhisper-large")
    pho_model = AutoModelForSpeechSeq2Seq.from_pretrained("vinai/PhoWhisper-large", torch_dtype=torch.float16, use_safetensors=False)
    pipe_pho = pipeline("automatic-speech-recognition", model=pho_model, tokenizer=pho_processor.tokenizer, feature_extractor=pho_processor.feature_extractor, chunk_length_s=30, torch_dtype=torch.float16, device=device0)

    print(f"Loading openai/whisper-large-v3 -> {device1}")
    pipe_whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", chunk_length_s=30, torch_dtype=torch.float16, device=device1)

    print(f"Loading khanhld/chunkformer-rnnt-large-vie -> {device0}")
    model_chunk = None
    try:
        from chunkformer import ChunkFormerModel
        model_chunk = ChunkFormerModel.from_pretrained("khanhld/chunkformer-rnnt-large-vie")
        if torch.cuda.is_available():
            model_chunk = model_chunk.to(device0)
    except ImportError:
        print("[WARN] chunkformer not installed. Skipping model 3.")

    gemini_client = init_gemini()

    print("✅ All models loaded.")
    return {
        "pipe_pho": pipe_pho,
        "pipe_whisper": pipe_whisper,
        "model_chunk": model_chunk,
        "gemini_client": gemini_client,
    }


def get_time_aligned_transcription(data_path, task, audio_name="output.wav", models=None):
    """Transcribe audio files using pre-loaded models. If models is None, loads them (slow)."""
    if models is None:
        models = load_models()

    pipe_pho = models["pipe_pho"]
    pipe_whisper = models["pipe_whisper"]
    model_chunk = models["model_chunk"]
    gemini_client = models["gemini_client"]

    audio_paths = sorted(glob(f"{data_path}/*/{MODEL_NAME}{audio_name}"))
    json_name = audio_name.rsplit(".", 1)[0] + ".json"

    for audio_path in tqdm(audio_paths, desc="Ensemble ASR"):
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Nếu file hoàn toàn im lặng thì bỏ qua
        if len(waveform) == 0 or np.max(np.abs(waveform)) < 1e-6:
            result_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", json_name)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f: json.dump({"text": "", "chunks": []}, f, indent=4)
            continue

        tmp_name = None
        out1, out2, out3 = {"text": "", "chunks": []}, {"text": "", "chunks": []}, {"text": "", "chunks": []}

        def run_phowhisper(wav_path):
            pred = pipe_pho(wav_path, return_timestamps="word", generate_kwargs={"language": "vietnamese", "condition_on_prev_tokens": False})
            return format_whisper_chunks(pred, 0)

        def run_whisper_v3(wav_path):
            pred = pipe_whisper(wav_path, return_timestamps="word", generate_kwargs={"language": "vietnamese"})
            return format_whisper_chunks(pred, 0)

        def run_chunkformer(wav_path):
            if not model_chunk:
                return {"text": "", "chunks": []}
            try:
                pred = model_chunk.endless_decode(audio_path=wav_path, chunk_size=64, left_context_size=128, right_context_size=128, total_batch_duration=14400, return_timestamps=True)
                result = format_chunkformer_chunks(pred, 0)
                if not result["text"] and isinstance(pred, str):
                    result["text"] = pred
                return result
            except Exception as e:
                print(f"[WARN] Chunkformer failed: {e}")
                return {"text": "", "chunks": []}

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
                sf.write(tmp.name, waveform, sr)

                # Chạy song song 3 model bằng ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=3) as executor:
                    fut1 = executor.submit(run_phowhisper, tmp.name)
                    fut2 = executor.submit(run_whisper_v3, tmp.name)
                    fut3 = executor.submit(run_chunkformer, tmp.name)

                    out1 = fut1.result()
                    out2 = fut2.result()
                    out3 = fut3.result()

        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)

        # Merge with Gemini
        best_idx = merge_with_gemini(gemini_client, out1["text"], out2["text"], out3["text"])
        final_out = out1 if best_idx == 1 else (out2 if best_idx == 2 else out3)
        
        # Save output
        result_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", json_name)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_out, f, indent=4, ensure_ascii=False)

def transcribe_v1_benchmark(root_dir, audio_name="output.wav", models=None):
    """Transcribe all v1 tasks. Loads models once if not provided."""
    if models is None:
        models = load_models()

    tasks = [
        ("synthetic_pause_handling", "default"),
        ("candor_turn_taking", "default"),
        ("synthetic_user_interruption", "user_interruption"),
    ]
    for task_name, task_mode in tasks:
        task_dir = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_dir): continue
        print(f"ASR {audio_name} -> {task_name}")
        get_time_aligned_transcription(task_dir, task_mode, audio_name, models=models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--v1_benchmark_root", type=str, default=None)
    parser.add_argument("--task", type=str, default="default")
    parser.add_argument("--audio_name", type=str, default="output.wav")
    args = parser.parse_args()
    loaded_models = load_models()
    if args.v1_benchmark_root: transcribe_v1_benchmark(args.v1_benchmark_root, args.audio_name, models=loaded_models)
    else: get_time_aligned_transcription(args.root_dir, args.task, args.audio_name, models=loaded_models)

