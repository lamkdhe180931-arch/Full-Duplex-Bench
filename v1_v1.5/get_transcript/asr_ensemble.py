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

def format_whisper_chunks(prediction, offset, interrupt_end_time):
    chunks = []
    text = ""
    for chunk in prediction.get("chunks", []):
        word = chunk["text"].strip()
        if not word or chunk["timestamp"] is None:
            continue
        start_time = chunk["timestamp"][0] + offset
        end_time = chunk["timestamp"][1] + offset
        if interrupt_end_time is not None and start_time < interrupt_end_time:
            continue
        text += word + " "
        chunks.append({"text": word, "timestamp": [start_time, end_time]})
    return {"text": text.strip(), "chunks": chunks}

def format_chunkformer_chunks(prediction, offset, interrupt_end_time):
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
        if interrupt_end_time is not None and start_time < interrupt_end_time:
            continue
        text += word + " "
        chunks.append({"text": word, "timestamp": [start_time, end_time]})
    return {"text": text.strip(), "chunks": chunks}

def get_time_aligned_transcription(data_path, task, audio_name="output.wav"):
    audio_paths = sorted(glob(f"{data_path}/*/{MODEL_NAME}{audio_name}"))
    json_name = audio_name.rsplit(".", 1)[0] + ".json"
    
    gemini_client = init_gemini()

    print("Loading models into GPUs...")
    disable_safetensors_auto_conversion()
    
    print("Loading PhoWhisper-large -> cuda:0")
    pho_processor = AutoProcessor.from_pretrained("vinai/PhoWhisper-large")
    pho_model = AutoModelForSpeechSeq2Seq.from_pretrained("vinai/PhoWhisper-large", torch_dtype=torch.float16, use_safetensors=False)
    pipe_pho = pipeline("automatic-speech-recognition", model=pho_model, tokenizer=pho_processor.tokenizer, feature_extractor=pho_processor.feature_extractor, chunk_length_s=30, torch_dtype=torch.float16, device="cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading openai/whisper-large-v3 -> cuda:1")
    pipe_whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", chunk_length_s=30, torch_dtype=torch.float16, device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    print("Loading khanhld/chunkformer-rnnt-large-vie -> cuda:0")
    model_chunk = None
    try:
        from chunkformer import ChunkFormerModel
        model_chunk = ChunkFormerModel.from_pretrained("khanhld/chunkformer-rnnt-large-vie")
        if torch.cuda.is_available():
            model_chunk = model_chunk.to("cuda:0")
    except ImportError:
        print("[WARN] chunkformer not installed. Skipping model 3.")

    for audio_path in tqdm(audio_paths, desc="Ensemble ASR"):
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        offset = 0.0
        timing_path = os.path.join(os.path.dirname(audio_path), "inference_timing.json")
        output_timeline_aligned = False
        if os.path.exists(timing_path):
            with open(timing_path, "r", encoding="utf-8") as f:
                timing = json.load(f)
                output_timeline_aligned = bool(timing.get("output_timeline_aligned"))
                if not output_timeline_aligned:
                    offset = timing.get("response_start_sec") or 0.0

        interrupt_end_time = None
        if task == "user_interruption":
            meta_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", "interrupt.json")
            if not os.path.exists(meta_path):
                meta_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                if isinstance(meta_data, list):
                    _, interrupt_end_time = meta_data[0]["timestamp"]
                else:
                    _, interrupt_end_time = meta_data["timestamps"]

        if task == "user_interruption" and not os.path.exists(timing_path) and interrupt_end_time:
            offset = interrupt_end_time
            start_idx = int(interrupt_end_time * sr)
            waveform = waveform[start_idx:]

        non_zero_indices = np.where(np.abs(waveform) > 1e-4)[0]
        if len(non_zero_indices) > 0:
            last_active_idx = non_zero_indices[-1]
            padding_samples = int(0.5 * sr)
            end_idx = min(len(waveform), last_active_idx + padding_samples)
            waveform = waveform[:end_idx]
        else:
            result_path = audio_path.replace(f"{MODEL_NAME}{audio_name}", json_name)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f: json.dump({"text": "", "chunks": []}, f, indent=4)
            continue

        tmp_name = None
        out1, out2, out3 = {"text": "", "chunks": []}, {"text": "", "chunks": []}, {"text": "", "chunks": []}
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
                sf.write(tmp.name, waveform, sr)
                
                # Model 1: PhoWhisper
                pred1 = pipe_pho(tmp.name, return_timestamps="word", generate_kwargs={"language": "vietnamese", "condition_on_prev_tokens": False})
                out1 = format_whisper_chunks(pred1, offset, interrupt_end_time)
                
                # Model 2: Whisper-v3
                pred2 = pipe_whisper(tmp.name, return_timestamps="word", generate_kwargs={"language": "vietnamese"})
                out2 = format_whisper_chunks(pred2, offset, interrupt_end_time)
                
                # Model 3: Chunkformer
                if model_chunk:
                    try:
                        pred3 = model_chunk.endless_decode(audio_path=tmp.name, chunk_size=64, left_context_size=128, right_context_size=128, total_batch_duration=14400, return_timestamps=True)
                        out3 = format_chunkformer_chunks(pred3, offset, interrupt_end_time)
                        if not out3["text"] and isinstance(pred3, str):
                            out3["text"] = pred3 # fallback if return_timestamps failed
                    except Exception as e:
                        print(f"[WARN] Chunkformer failed for {tmp.name}: {e}")

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

def transcribe_v1_benchmark(root_dir, audio_name="output.wav"):
    tasks = [
        ("synthetic_pause_handling", "default"),
        ("candor_turn_taking", "default"),
        ("synthetic_user_interruption", "user_interruption"),
    ]
    for task_name, task_mode in tasks:
        task_dir = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_dir): continue
        print(f"ASR {audio_name} -> {task_name}")
        get_time_aligned_transcription(task_dir, task_mode, audio_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--v1_benchmark_root", type=str, default=None)
    parser.add_argument("--task", type=str, default="default")
    parser.add_argument("--audio_name", type=str, default="output.wav")
    args = parser.parse_args()
    if args.v1_benchmark_root: transcribe_v1_benchmark(args.v1_benchmark_root, args.audio_name)
    else: get_time_aligned_transcription(args.root_dir, args.task, args.audio_name)
