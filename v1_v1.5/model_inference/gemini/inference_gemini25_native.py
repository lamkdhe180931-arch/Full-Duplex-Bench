#!/usr/bin/env python3
"""
Gemini 2.5 Flash Native Audio Inference Script

Key Features:
1. TIME-SYNCHRONIZED recording - output aligned with input timeline
2. MULTI-SESSION for interruption handling (supports barge-in recovery)

Usage:
    python inference_gemini25_native.py --base-dir /path/to/data --task synthetic_user_interruption --overwrite
"""
import os
import asyncio
import time
import math
import traceback
import argparse
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal as ss
from dotenv import load_dotenv

from google import genai
from google.genai import types
from glob import glob

# Load environment variables
load_dotenv()

# ===== Audio Config =====
SEND_SAMPLE_RATE = 16000   # Input to Gemini (16kHz)
RECEIVE_SAMPLE_RATE = 24000  # Output from Gemini (24kHz)
CHUNK_SIZE = 1024  # Standard chunk size
REC_TICK_MS = 10   # Recorder tick interval

# ===== Model Config =====
# MODEL = "gemini-2.0-flash-live-001"  # DEPRECATED
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment. Check .env file.")

# See: https://ai.google.dev/api/live for all options
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful and friendly AI assistant.",
    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False,
            # LOW sensitivity = less likely to detect speech (reduce false positives)
            "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
            "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
            # Require longer speech before committing start-of-speech (default ~40ms for HIGH)
            "prefix_padding_ms": 40,
            # Require longer silence before ending speech (default ~300ms for HIGH)
            "silence_duration_ms": 300,
        },
        # Optional: Set to "NO_INTERRUPTION" to completely disable user interruptions
        "activity_handling": "START_OF_ACTIVITY_INTERRUPTS",
    },
}


class SynchronizedRecorder:
    """Records only Gemini response audio, then stops dynamically."""

    def __init__(self, out_sr: int, outfile: str, session_start_time: float):
        self.out_sr = out_sr
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.outfile = outfile
        self.running = True
        self.session_start_time = session_start_time
        self.first_audio_wall_sec = None
        self.last_audio_wall_sec = None
        self.samples_written = 0

    async def add(self, pcm: bytes):
        now = time.time() - self.session_start_time
        if self.first_audio_wall_sec is None:
            self.first_audio_wall_sec = now
        self.last_audio_wall_sec = now
        await self.queue.put(pcm)

    def interrupt(self):
        cleared = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        print(f"[DEBUG] Recorder: INTERRUPTED! Cleared {cleared} queued chunks")

    def stop(self):
        self.running = False

    @property
    def response_duration_sec(self) -> float:
        return self.samples_written / self.out_sr

    async def run(self):
        print("[DEBUG] Recorder: Started")
        os.makedirs(os.path.dirname(self.outfile), exist_ok=True)
        wf = wave.open(self.outfile, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.out_sr)

        try:
            while self.running or not self.queue.empty():
                try:
                    pcm = await asyncio.wait_for(self.queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                wf.writeframes(pcm)
                self.samples_written += len(pcm) // 2
        finally:
            wf.close()

        print(f"[DEBUG] Recorder: Done. Response audio: {self.response_duration_sec:.2f}s")

def resample_to_16k(input_path: Path) -> Tuple[Path, float]:
    """Resample to 16kHz mono."""
    data, sr = sf.read(input_path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    duration = len(data) / sr

    if sr != SEND_SAMPLE_RATE:
        g = math.gcd(int(sr), SEND_SAMPLE_RATE)
        data = ss.resample_poly(data, SEND_SAMPLE_RATE // g, int(sr) // g)

    # REMOVED: Max Normalization
    # This was amplifying background noise and triggering VAD!
    # data = (data / max_val * 32767).astype(np.int16)
    
    # Just clip and convert to int16 to preserve original volume
    data = np.clip(data * 32767, -32768, 32767).astype(np.int16)

    out_path = input_path.with_name(f"{input_path.stem}_16k_mono.wav")
    sf.write(out_path, data, SEND_SAMPLE_RATE, subtype="PCM_16")
    return out_path, duration


def load_audio_chunks(wav16k_path: Path) -> List[bytes]:
    """Split into chunks."""
    data, _ = sf.read(wav16k_path, dtype="int16")
    pad = (-len(data)) % CHUNK_SIZE
    if pad:
        data = np.pad(data, (0, pad))

    return [data[i:i+CHUNK_SIZE].tobytes() for i in range(0, len(data), CHUNK_SIZE)]


async def run_session(
    client: genai.Client,
    session_id: int,
    chunks: List[bytes],
    start_idx: int,
    recorder: SynchronizedRecorder,
    session_start_time: float,
    max_response_sec: float,
) -> int:
    """
    Run one session.
    """
    chunk_duration = CHUNK_SIZE / SEND_SAMPLE_RATE
    session_time_offset = start_idx * chunk_duration
    
    print(f"[DEBUG][Session {session_id}] Start from chunk {start_idx} (t={session_time_offset:.2f}s)")
    
    session_done = False
    idx = start_idx
    total = len(chunks)
    audio_received = 0
    was_interrupted = False
    
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as sess:
        print(f"[DEBUG][Session {session_id}] Connected to Live API")
        
        async def sender():
            nonlocal idx, session_done
            while idx < total and not session_done:
                chunk = chunks[idx]
                
                # Update recorder timeline alignment
                recorder.current_sender_time = idx * chunk_duration
                
                # REMOVED: Client-side Noise Gate
                
                await sess.send_realtime_input(
                    audio={"data": chunk, "mime_type": "audio/pcm"}
                )
                idx += 1
                # Real-time pacing
                await asyncio.sleep(chunk_duration)
            
            if not session_done:
                print(f"[DEBUG][Session {session_id}] Sender finished sending all chunks")
                try:
                    await sess.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
        
        async def receiver():
            nonlocal session_done, audio_received, was_interrupted
            print(f"[DEBUG][Session {session_id}] Receiver started")
            async for resp in sess.receive():
                sc = resp.server_content
                if not sc:
                    continue
                
                # INTERRUPT
                if getattr(sc, "interrupted", False):
                    current_time = time.time() - session_start_time
                    print(f"[DEBUG][Session {session_id}] *** INTERRUPTED at t={current_time:.2f}s, chunk {idx} ***")
                    recorder.interrupt()
                    was_interrupted = True
                    session_done = True
                    return
                
                # Model Audio
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            await recorder.add(part.inline_data.data)
                            audio_received += len(part.inline_data.data)
                        
                        if getattr(part, "generation_complete", False):
                            session_done = True
                            return
                
                # Turn Complete
                if getattr(sc, "turn_complete", False) or getattr(sc, "generation_complete", False):
                    audio_sec = audio_received / 2 / RECEIVE_SAMPLE_RATE
                    print(f"[DEBUG][Session {session_id}] Turn complete (audio: {audio_sec:.2f}s)")
                    session_done = True
                    return
            print(f"[DEBUG][Session {session_id}] Receiver loop ended naturally")
        
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        
        await sender_task

        try:
            await asyncio.wait_for(receiver_task, timeout=max_response_sec)
        except asyncio.TimeoutError:
            print(f"[DEBUG][Session {session_id}] Response timeout after {max_response_sec:.1f}s")
            session_done = True
            recorder.stop()
            receiver_task.cancel()
    
    return idx


async def process_single_file(input_wav: str, output_wav: str, overwrite: bool = True, max_response_sec: float = 30.0) -> bool:
    """Process file with time-synchronized multi-session approach."""
    input_path = Path(input_wav)
    if not input_path.exists():
        print(f"[ERROR] {input_wav} not found.")
        return False

    if os.path.exists(output_wav) and not overwrite:
        return True

    # Prepare audio
    wav16k_path, duration = resample_to_16k(input_path)
    chunks = load_audio_chunks(wav16k_path)
    total_chunks = len(chunks)
    print(f"[INFO] Loaded {total_chunks} chunks, duration: {duration:.2f}s")

    # Track overall start time for synchronization
    session_start_time = time.time()

    # Create recorder. output.wav contains only actual Gemini response audio.
    recorder = SynchronizedRecorder(RECEIVE_SAMPLE_RATE, output_wav, session_start_time)
    recorder_task = asyncio.create_task(recorder.run())

    # Create client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Multi-session loop
    chunk_idx = 0
    session_id = 1
    
    while chunk_idx < total_chunks:
        if session_id > 1:
            print(f"[DEBUG] Cooldown sleep 2.0s before Session {session_id} to avoid concurrent connection limit")
            await asyncio.sleep(2.0)
        try:
            new_idx = await run_session(
                client, session_id, chunks, chunk_idx, recorder, session_start_time, max_response_sec
            )
        except Exception as e:
            print(f"[ERROR] Session {session_id}: {e}")
            traceback.print_exc()
            break
        
        if new_idx == chunk_idx:
            chunk_idx += 1
        else:
            chunk_idx = new_idx
        
        session_id += 1
    
    # Finish recording
    recorder.stop()
    await recorder_task

    response_start_sec = recorder.first_audio_wall_sec
    timing = {
        "input_duration_sec": duration,
        "response_start_sec": response_start_sec,
        "response_latency_sec": None if response_start_sec is None else response_start_sec - duration,
        "response_duration_sec": recorder.response_duration_sec,
        "response_end_sec": None if response_start_sec is None else response_start_sec + recorder.response_duration_sec,
        "max_response_sec": max_response_sec,
        "output_sample_rate": RECEIVE_SAMPLE_RATE,
    }
    timing_path = os.path.join(os.path.dirname(output_wav), "inference_timing.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        import json
        json.dump(timing, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {timing_path}")

    # Cleanup
    if os.path.exists(wav16k_path):
        os.remove(wav16k_path)

    print(f"[INFO] Saved {output_wav}")
    return True


async def batch_process(args):
    """Process all files."""
    base_dir = os.path.expanduser(args.base_dir)
    pattern = os.path.join(base_dir, args.task or "*", "*", f"{args.prefix}input.wav")
    files = sorted(glob(pattern))
    print(f"Found {len(files)} files.")

    success = 0
    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {f}")
        out_wav = os.path.join(os.path.dirname(f), f"{args.prefix}output.wav")
        combined_wav = os.path.join(os.path.dirname(f), f"{args.prefix}combined.wav")

        if os.path.exists(out_wav) and not args.overwrite:
            print("Skip")
            if not os.path.exists(combined_wav):
                try:
                    from pydub import AudioSegment
                    # Tách kênh Stereo: User (Tai trái), AI (Tai phải)
                    import json
                    sound_in = AudioSegment.from_wav(f).pan(-1.0)
                    sound_out = AudioSegment.from_wav(out_wav).pan(1.0)
                    timing_path = os.path.join(os.path.dirname(out_wav), "inference_timing.json")
                    position_ms = 0
                    if os.path.exists(timing_path):
                        with open(timing_path, "r", encoding="utf-8") as tf:
                            position_ms = int((json.load(tf).get("response_start_sec") or 0) * 1000)
                    sound_in.overlay(sound_out, position=position_ms).export(combined_wav, format="wav")
                    print(f"[INFO] Saved {combined_wav}")
                except Exception as e:
                    pass
            success += 1
            continue

        try:
            if await process_single_file(f, out_wav, args.overwrite, args.max_response_sec):
                try:
                    from pydub import AudioSegment
                    # Trộn (mix) audio input và output lại với nhau theo cùng một timeline
                    # Tách kênh Stereo: User (Tai trái), AI (Tai phải) để không bị loạn âm thanh
                    import json
                    sound_in = AudioSegment.from_wav(f).pan(-1.0)
                    sound_out = AudioSegment.from_wav(out_wav).pan(1.0)
                    timing_path = os.path.join(os.path.dirname(out_wav), "inference_timing.json")
                    position_ms = 0
                    if os.path.exists(timing_path):
                        with open(timing_path, "r", encoding="utf-8") as tf:
                            position_ms = int((json.load(tf).get("response_start_sec") or 0) * 1000)
                    sound_in.overlay(sound_out, position=position_ms).export(combined_wav, format="wav")
                    print(f"[INFO] Saved {combined_wav}")
                except Exception as e:
                    print(f"[ERROR] Failed to mix combined audio: {e}")
                success += 1
        except Exception as e:
            print(f"Failed: {e}")

    print(f"\nDone. {success}/{len(files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-response-sec", type=float, default=30.0)
    asyncio.run(batch_process(parser.parse_args()))
