import os
import sys
import json
import copy
import hashlib
from pydub import AudioSegment

# Thêm thư mục cha vào sys.path để import được core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tts_generator import VietnameseTTSGenerator
from core.audio_mixer import AudioMixer
from core.off_axis_effects import apply_off_axis_effect
from core.generated_background import GeneratedBackgroundGenerator
from core.snr_mixer import mix_background_at_snr

AGENT_RESPONSE_WINDOW_SEC = 15

TTS_OVERLAP_SCENARIOS = {
    "user_interruption": {
        "strategy": "tts_delayed_same_speaker",
        "same_speaker": True,
        "overlap_gain_db": 0,
        "agent_response_window_sec": AGENT_RESPONSE_WINDOW_SEC,
    },
    "user_backchannel": {
        "strategy": "tts_quiet_same_speaker_backchannel",
        "same_speaker": True,
        "overlap_gain_db": -6,
        "agent_response_window_sec": AGENT_RESPONSE_WINDOW_SEC,
    },
}

ASSET_REQUIRED_SCENARIOS = {}

BACKGROUND_SPEECH_SCENARIO = {
    "strategy": "generated_background_snr_mix",
    "default_provider": "elevenlabs_sfx",
    "supported_providers": ["elevenlabs_sfx", "audiogen_cli", "audioldm2_cli"],
    "addressed_to_agent": False,
    "target_snr_db": 8,
    "background_duration_sec": 6.0,
    "agent_response_window_sec": AGENT_RESPONSE_WINDOW_SEC,
}

TALKING_TO_OTHER_SCENARIO = {
    "strategy": "tts_off_axis_simulation",
    "same_speaker": True,
    "off_axis": True,
    "addressed_to_agent": False,
    "angle_deg": 90,
    "distance_m": 1.5,
    "overlap_gain_db": -3,
    "agent_response_window_sec": AGENT_RESPONSE_WINDOW_SEC,
}

def generate_interruption_and_backchannel(generator, mixer, templates, category, output_base):
    if category not in TTS_OVERLAP_SCENARIOS:
        raise ValueError(f"{category} is not a TTS overlap scenario.")

    print(f"-> Đang sinh dữ liệu cho task v1.5: {category}")
    task_dir = os.path.join(output_base, category)
    scenario = TTS_OVERLAP_SCENARIOS[category]
    
    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. Sinh file audio cho context_text (câu nói trước khi có Agent)
        context_path = os.path.join(sample_dir, "temp_context.wav")
        generator.generate(item["context_text"], context_path, role="primary")
        context_tts_profile = copy.deepcopy(getattr(generator, "last_synthesis", None))
        
        # 2. Sinh file audio cho current_turn_text (câu nói chen/ngắt lời)
        current_path = os.path.join(sample_dir, "temp_current.wav")
        context_voice = None
        if context_tts_profile and context_tts_profile.get("profile"):
            context_voice = context_tts_profile["profile"].get("voice")
        generator.generate(item["current_turn_text"], current_path, role=category, voice=context_voice)
        current_tts_profile = copy.deepcopy(getattr(generator, "last_synthesis", None))
        
        # Load audio dưới dạng AudioSegment
        context_sound = mixer.load_audio(context_path)
        current_sound = mixer.load_audio(current_path)
        overlap_gain_db = scenario["overlap_gain_db"]
        if overlap_gain_db != 0:
            current_sound = current_sound.apply_gain(overlap_gain_db)
        
        len_context_sec = len(context_sound) / 1000.0
        len_current_ms = len(current_sound)
        
        delay_sec = item["delay_sec"]
        delay_ms = int(delay_sec * 1000)
        
        # 3. Tạo input.wav: context + cửa sổ agent đang trả lời, trong đó user chen lời/backchannel.
        silence_window = AudioSegment.silent(
            duration=int(scenario["agent_response_window_sec"] * 1000),
            frame_rate=16000
        )
        silence_with_current = silence_window.overlay(current_sound, position=delay_ms)
        input_sound = context_sound + silence_with_current
        
        input_wav_path = os.path.join(sample_dir, "input.wav")
        mixer.save_audio(input_sound, input_wav_path)
        
        # 4. Tạo clean_input.wav (file sạch đối chứng - không có câu chen)
        # clean_input.wav sẽ gồm context + 15 giây lặng
        clean_input_sound = context_sound + silence_window
        
        clean_input_wav_path = os.path.join(sample_dir, "clean_input.wav")
        mixer.save_audio(clean_input_sound, clean_input_wav_path)
        
        # 5. Ghi file metadata.json
        metadata = {
            "context_text": item["context_text"],
            "current_turn_text": item["current_turn_text"],
            "timestamps": [
                len_context_sec + delay_sec,
                len_context_sec + delay_sec + (len_current_ms / 1000.0)
            ],
            "tts_profiles": {
                "context": context_tts_profile,
                "overlap": current_tts_profile
            },
            "simulation": dict(scenario)
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        # 6. Dọn dẹp file tạm
        if os.path.exists(context_path):
            os.remove(context_path)
        if os.path.exists(current_path):
            os.remove(current_path)

def generate_other_and_background(generator, mixer, templates, category, output_base):
    print(f"-> Đang sinh dữ liệu cho task v1.5: {category}")
    task_dir = os.path.join(output_base, category)
    
    # Sử dụng giọng đọc phụ (ví dụ: giọng nam vi-VN-NamMinhNeural) để sinh tiếng nói chen ở nền
    bg_seed = None if getattr(generator, "seed", None) is None else generator.seed + 1009
    bg_generator = VietnameseTTSGenerator(provider="edge-tts", seed=bg_seed)
    
    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. Sinh clean_input.wav (file sạch của người nói chính)
        clean_input_wav_path = os.path.join(sample_dir, "clean_input.wav")
        generator.generate(item["current_turn_text"], clean_input_wav_path, role="primary")
        clean_tts_profile = copy.deepcopy(getattr(generator, "last_synthesis", None))
        
        # 2. Sinh file audio nhiễu (overlap_text) bằng giọng đọc khác
        overlap_path = os.path.join(sample_dir, "temp_overlap.wav")
        bg_generator.generate(item["overlap_text"], overlap_path, role=category)
        overlap_tts_profile = copy.deepcopy(getattr(bg_generator, "last_synthesis", None))
        
        # Load audio để đo đạc thời lượng
        clean_sound = mixer.load_audio(clean_input_wav_path)
        overlap_sound = mixer.load_audio(overlap_path)
        
        # Đo độ dài câu thoại sạch gốc trước khi thêm khoảng lặng
        len_clean_speech_ms = len(clean_sound)
        
        # Lấy gain nếu có thiết lập cấu hình trong kịch bản (ví dụ -12dB cho background noise)
        gain = item.get("bg_gain", 0)
        if gain != 0:
            overlap_sound = overlap_sound.apply_gain(gain)
            
        delay_sec = item["delay_sec"]
        # Điểm chèn âm thanh bắt đầu từ lúc nói xong câu hỏi chính + khoảng trễ
        delay_ms = len_clean_speech_ms + int(delay_sec * 1000)
        
        # 3. Tính toán phần thừa để tránh bị cắt cụt file và giữ cho clean_input / input bằng độ dài nhau
        # Thêm 15 giây khoảng lặng đệm vào cuối clean_sound cho phản hồi của Agent
        silence_trailing = AudioSegment.silent(duration=15000, frame_rate=16000)
        clean_sound = clean_sound + silence_trailing
        
        overlap_end_ms = delay_ms + len(overlap_sound)
        if overlap_end_ms > len(clean_sound):
            silence_needed_ms = overlap_end_ms - len(clean_sound)
            silence = AudioSegment.silent(duration=silence_needed_ms, frame_rate=16000)
            
            # Cả hai file clean_input và input đều được kéo dài bằng khoảng lặng ở cuối
            clean_sound = clean_sound + silence
            
        # Ghi đè lại clean_input.wav đã được đồng bộ độ dài
        mixer.save_audio(clean_sound, clean_input_wav_path)
        
        # 4. Tạo input.wav bằng cách trộn đè (overlay)
        input_sound = clean_sound.overlay(overlap_sound, position=delay_ms)
        input_wav_path = os.path.join(sample_dir, "input.wav")
        mixer.save_audio(input_sound, input_wav_path)
        
        # 5. Ghi file metadata.json
        start_overlap_sec = (len_clean_speech_ms / 1000.0) + delay_sec
        metadata = {
            "context_text": item["context_text"],
            "current_turn_text": item["current_turn_text"],
            "timestamps": [
                start_overlap_sec,
                start_overlap_sec + (len(overlap_sound) / 1000.0)
            ],
            "tts_profiles": {
                "clean_input": clean_tts_profile,
                "overlap": overlap_tts_profile
            }
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        # 6. Dọn dẹp file tạm
        if os.path.exists(overlap_path):
            os.remove(overlap_path)

def generate_talking_to_other(generator, mixer, templates, output_base):
    print("-> Đang sinh dữ liệu cho task v1.5: talking_to_other")
    task_dir = os.path.join(output_base, "talking_to_other")
    scenario = TALKING_TO_OTHER_SCENARIO

    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # 1. Sinh clean_input.wav: người dùng nói trực tiếp với agent.
        clean_input_wav_path = os.path.join(sample_dir, "clean_input.wav")
        generator.generate(item["current_turn_text"], clean_input_wav_path, role="primary")
        clean_tts_profile = copy.deepcopy(getattr(generator, "last_synthesis", None))

        # 2. Sinh câu nói với người khác bằng cùng speaker, sau đó xử lý lệch hướng mic.
        overlap_path = os.path.join(sample_dir, "temp_overlap.wav")
        clean_voice = None
        if clean_tts_profile and clean_tts_profile.get("profile"):
            clean_voice = clean_tts_profile["profile"].get("voice")
        generator.generate(item["overlap_text"], overlap_path, role="talking_to_other", voice=clean_voice)
        overlap_tts_profile = copy.deepcopy(getattr(generator, "last_synthesis", None))

        clean_sound = mixer.load_audio(clean_input_wav_path)
        overlap_sound = mixer.load_audio(overlap_path)
        len_clean_speech_ms = len(clean_sound)

        angle_deg = item.get("angle_deg", scenario["angle_deg"])
        distance_m = item.get("distance_m", scenario["distance_m"])
        rir_path = item.get("rir_path")
        rt60 = item.get("rt60", 0.35)
        room_dim = item.get("room_dim", None)
        overlap_sound, audio_effect = apply_off_axis_effect(
            overlap_sound,
            rir_path=rir_path,
            angle_deg=angle_deg,
            distance_m=distance_m,
            rt60=rt60,
            room_dim=room_dim,
        )


        overlap_gain_db = item.get("overlap_gain_db", scenario["overlap_gain_db"])
        if overlap_gain_db != 0:
            overlap_sound = overlap_sound.apply_gain(overlap_gain_db)

        delay_sec = item["delay_sec"]
        delay_ms = len_clean_speech_ms + int(delay_sec * 1000)

        silence_trailing = AudioSegment.silent(
            duration=int(scenario["agent_response_window_sec"] * 1000),
            frame_rate=16000,
        )
        clean_sound = clean_sound + silence_trailing

        overlap_end_ms = delay_ms + len(overlap_sound)
        if overlap_end_ms > len(clean_sound):
            silence_needed_ms = overlap_end_ms - len(clean_sound)
            clean_sound = clean_sound + AudioSegment.silent(duration=silence_needed_ms, frame_rate=16000)

        mixer.save_audio(clean_sound, clean_input_wav_path)

        input_sound = clean_sound.overlay(overlap_sound, position=delay_ms)
        input_wav_path = os.path.join(sample_dir, "input.wav")
        mixer.save_audio(input_sound, input_wav_path)

        start_overlap_sec = (len_clean_speech_ms / 1000.0) + delay_sec
        simulation = dict(scenario)
        simulation["angle_deg"] = angle_deg
        simulation["distance_m"] = distance_m
        simulation["overlap_gain_db"] = overlap_gain_db

        metadata = {
            "context_text": item["context_text"],
            "current_turn_text": item["current_turn_text"],
            "timestamps": [
                start_overlap_sec,
                start_overlap_sec + (len(overlap_sound) / 1000.0)
            ],
            "tts_profiles": {
                "clean_input": clean_tts_profile,
                "overlap": overlap_tts_profile
            },
            "simulation": simulation,
            "audio_effect": audio_effect,
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        if os.path.exists(overlap_path):
            os.remove(overlap_path)

def _background_prompt_from_item(item):
    if item.get("background_prompt"):
        return item["background_prompt"]

    description = item.get("description", "").strip()
    overlap_text = item.get("overlap_text", "").strip()
    parts = [
        description or "natural Vietnamese background speech/noise",
        "indistinct, not addressed to the microphone, realistic room ambience",
    ]
    if overlap_text:
        parts.append(f"semantic hint: {overlap_text}")
    return "; ".join(parts)


def generate_background_speech(
    generator,
    mixer,
    templates,
    output_base,
    background_generator=None,
):
    print("-> Đang sinh dữ liệu cho task v1.5: background_speech")
    task_dir = os.path.join(output_base, "background_speech")
    scenario = BACKGROUND_SPEECH_SCENARIO
    background_generator = background_generator or GeneratedBackgroundGenerator(
        provider=os.getenv("FDB_BACKGROUND_PROVIDER", scenario["default_provider"])
    )

    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        clean_input_wav_path = os.path.join(sample_dir, "clean_input.wav")
        generator.generate(item["current_turn_text"], clean_input_wav_path, role="primary")
        clean_tts_profile = copy.deepcopy(getattr(generator, "last_synthesis", None))

        clean_speech = mixer.load_audio(clean_input_wav_path)
        len_clean_speech_ms = len(clean_speech)

        delay_sec = item["delay_sec"]
        delay_ms = len_clean_speech_ms + int(delay_sec * 1000)
        duration_sec = float(item.get("background_duration_sec", scenario["background_duration_sec"]))
        duration_ms = int(duration_sec * 1000)
        target_snr_db = float(item.get("target_snr_db", scenario["target_snr_db"]))
        provider = item.get("background_provider") or os.getenv(
            "FDB_BACKGROUND_PROVIDER",
            scenario["default_provider"],
        )
        prompt = _background_prompt_from_item(item)

        background_path = os.path.join(sample_dir, "temp_generated_background.wav")
        seed = item.get("seed")
        if seed is None and getattr(generator, "seed", None) is not None:
            stable_offset = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
            seed = generator.seed + (stable_offset % 100000)
        background_metadata = background_generator.generate(
            prompt,
            background_path,
            duration_sec=duration_sec,
            seed=seed,
            provider=provider,
        )

        background_sound = mixer.load_audio(background_path)
        silence_trailing = AudioSegment.silent(
            duration=int(scenario["agent_response_window_sec"] * 1000),
            frame_rate=16000,
        )
        clean_with_window = clean_speech + silence_trailing

        input_sound, snr_metadata = mix_background_at_snr(
            clean_with_window,
            background_sound,
            position_ms=delay_ms,
            duration_ms=duration_ms,
            target_snr_db=target_snr_db,
            reference_sound=clean_speech,
        )

        if len(input_sound) > len(clean_with_window):
            clean_with_window = clean_with_window + AudioSegment.silent(
                duration=len(input_sound) - len(clean_with_window),
                frame_rate=16000,
            )

        mixer.save_audio(clean_with_window, clean_input_wav_path)
        input_wav_path = os.path.join(sample_dir, "input.wav")
        mixer.save_audio(input_sound, input_wav_path)

        start_sec = (len_clean_speech_ms / 1000.0) + delay_sec
        simulation = dict(scenario)
        simulation["target_snr_db"] = target_snr_db
        simulation["background_duration_sec"] = duration_sec
        simulation["provider"] = provider

        metadata = {
            "context_text": item.get("context_text", ""),
            "current_turn_text": item["current_turn_text"],
            "timestamps": [
                start_sec,
                start_sec + (duration_ms / 1000.0),
            ],
            "tts_profiles": {
                "clean_input": clean_tts_profile,
            },
            "simulation": simulation,
            "background_generation": background_metadata,
            "snr": snr_metadata,
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        if os.path.exists(background_path):
            os.remove(background_path)

def main():
    print("=== Khởi chạy sinh dữ liệu Full-Duplex-Bench v1.5 (Tiếng Việt) ===")
    
    # Khởi tạo các module lõi (Mặc định dùng giọng nữ vi-VN-HoaiMyNeural làm giọng chính)
    seed = int(os.getenv("FDB_TTS_SEED", "20260625"))
    generator = VietnameseTTSGenerator(provider="edge-tts", seed=seed)
    mixer = AudioMixer()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_dir, "templates")
    output_base = os.path.abspath(os.path.join(current_dir, "..", "..", "dataset", "v1_5"))
    
    # 1. Sinh dữ liệu User Interruption
    interrupt_path = os.path.join(templates_dir, "user_interruption.json")
    if os.path.exists(interrupt_path):
        with open(interrupt_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        generate_interruption_and_backchannel(generator, mixer, templates, "user_interruption", output_base)
        
    # 2. Sinh dữ liệu User Backchannel
    backchannel_path = os.path.join(templates_dir, "user_backchannel.json")
    if os.path.exists(backchannel_path):
        with open(backchannel_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        generate_interruption_and_backchannel(generator, mixer, templates, "user_backchannel", output_base)
        
    # 3. Sinh dữ liệu Talking to Other với mô phỏng off-axis.
    talk_other_path = os.path.join(templates_dir, "talking_to_other.json")
    if os.path.exists(talk_other_path):
        with open(talk_other_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        generate_talking_to_other(generator, mixer, templates, output_base)

    # 4. Sinh background_speech bằng text-to-audio noise/SFX rồi mix theo SNR.
    background_path = os.path.join(templates_dir, "background_speech.json")
    if os.path.exists(background_path):
        with open(background_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        generate_background_speech(generator, mixer, templates, output_base)
        
    print("\n=== Hoàn thành sinh tập dữ liệu v1.5 ===")

if __name__ == "__main__":
    main()
