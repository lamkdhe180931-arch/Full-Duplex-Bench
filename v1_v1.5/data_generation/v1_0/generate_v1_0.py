import os
import sys
import json
from pydub import AudioSegment

# Thêm thư mục cha vào sys.path để import được core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tts_generator import VietnameseTTSGenerator
from core.audio_mixer import AudioMixer

def generate_pause_handling(generator, mixer, templates, output_base):
    print("-> Đang sinh dữ liệu cho: synthetic_pause_handling")
    task_dir = os.path.join(output_base, "synthetic_pause_handling")
    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Đọc khoảng lặng từ file JSON (mặc định là 1.5s nếu không có)
        pause_duration = item.get("pause_duration_sec", 1.5)
        
        # 1. Tạo các file audio tạm cho part_1 và part_2
        p1_path = os.path.join(sample_dir, "temp_p1.wav")
        p2_path = os.path.join(sample_dir, "temp_p2.wav")
        
        generator.generate(item["part_1"], p1_path)
        p1_profile = generator.last_synthesis.get("profile") if generator.last_synthesis else None
        generator.generate(item["part_2"], p2_path, profile=p1_profile)
        
        # 2. Đọc độ dài của part_1 để tính mốc thời gian bắt đầu khoảng lặng (pause)
        p1_sound = mixer.load_audio(p1_path)
        len_p1_sec = len(p1_sound) / 1000.0
        
        # 3. Ghép nối 2 part bằng khoảng lặng ở giữa và thêm 15 giây lặng phía sau
        input_wav_path = os.path.join(sample_dir, "input.wav")
        mixer.concat_with_silence([p1_path, p2_path], pause_duration, input_wav_path)
        
        # Thêm 15 giây im lặng ở cuối cho phản hồi của Agent
        combined_sound = mixer.load_audio(input_wav_path)
        trailing_silence = AudioSegment.silent(duration=15000, frame_rate=16000)
        combined_sound_with_silence = combined_sound + trailing_silence
        mixer.save_audio(combined_sound_with_silence, input_wav_path)
        
        # 4. Ghi file chú thích pause.json
        pause_info = [
            {
                "text": "[PAUSE]",
                "timestamp": [
                    len_p1_sec,
                    len_p1_sec + pause_duration
                ]
            }
        ]
        with open(os.path.join(sample_dir, "pause.json"), "w", encoding="utf-8") as f:
            json.dump(pause_info, f, indent=4, ensure_ascii=False)
            
        # 5. Dọn dẹp file tạm
        if os.path.exists(p1_path):
            os.remove(p1_path)
        if os.path.exists(p2_path):
            os.remove(p2_path)

def generate_turn_taking(generator, mixer, templates, output_base):
    print("-> Đang sinh dữ liệu cho: candor_turn_taking")
    task_dir = os.path.join(output_base, "candor_turn_taking")
    
    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. Sinh file audio câu hỏi chính
        input_wav_path = os.path.join(sample_dir, "input.wav")
        generator.generate(item["text"], input_wav_path)
        
        # 2. Đo thời lượng câu nói
        sound = mixer.load_audio(input_wav_path)
        len_sound_sec = len(sound) / 1000.0
        
        # Thêm 15 giây im lặng ở cuối câu thoại đầu vào cho phản hồi của Agent
        trailing_silence = AudioSegment.silent(duration=15000, frame_rate=16000)
        sound_with_silence = sound + trailing_silence
        mixer.save_audio(sound_with_silence, input_wav_path)
        
        # 3. Ghi file chú thích turn_taking.json (mốc kết thúc tại timestamp[0])
        turn_info = [
            {
                "text": "[TURN-TAKING]",
                "timestamp": [
                    len_sound_sec,
                    0.0
                ]
            }
        ]
        with open(os.path.join(sample_dir, "turn_taking.json"), "w", encoding="utf-8") as f:
            json.dump(turn_info, f, indent=4, ensure_ascii=False)

def generate_user_interruption(generator, mixer, templates, output_base):
    print("-> Đang sinh dữ liệu cho: synthetic_user_interruption")
    task_dir = os.path.join(output_base, "synthetic_user_interruption")
    
    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. Sinh file context.wav (câu thoại đầu tiên của người dùng)
        context_wav_path = os.path.join(sample_dir, "context.wav")
        generator.generate(item["context"], context_wav_path)
        context_profile = generator.last_synthesis.get("profile") if generator.last_synthesis else None
        context_voice = context_profile.get("voice") if context_profile else None
        
        # 2. Sinh file interrupt.wav (câu thoại cướp lời của người dùng)
        interrupt_wav_path = os.path.join(sample_dir, "interrupt.wav")
        generator.generate(item["interrupt"], interrupt_wav_path, voice=context_voice)
        
        # 3. Tính toán dòng thời gian để ghép thành input.wav
        context_sound = mixer.load_audio(context_wav_path)
        len_context_sec = len(context_sound) / 1000.0
        
        interrupt_delay = item["interrupt_delay_sec"]
        interrupt_sound = mixer.load_audio(interrupt_wav_path)
        len_interrupt_sec = len(interrupt_sound) / 1000.0
        
        # Thêm 15 giây im lặng đằng sau context. Câu thoại ngắt lời sẽ nằm trong khoảng 15s này
        silence_window = AudioSegment.silent(duration=15000, frame_rate=16000)
        interrupt_pos_ms = int(interrupt_delay * 1000)
        silence_with_interrupt = silence_window.overlay(interrupt_sound, position=interrupt_pos_ms)
        
        # input.wav = context_sound + silence_with_interrupt
        input_wav_path = os.path.join(sample_dir, "input.wav")
        mixer.save_audio(context_sound + silence_with_interrupt, input_wav_path)
        
        # 5. Ghi file chú thích interrupt.json
        interrupt_info = [
            {
                "context": item["context"],
                "interrupt": item["interrupt"],
                "timestamp": [
                    len_context_sec + interrupt_delay,
                    len_context_sec + interrupt_delay + len_interrupt_sec
                ]
            }
        ]
        with open(os.path.join(sample_dir, "interrupt.json"), "w", encoding="utf-8") as f:
            json.dump(interrupt_info, f, indent=4, ensure_ascii=False)

def main():
    print("=== Khởi chạy sinh dữ liệu Full-Duplex-Bench v1.0 (Tiếng Việt) ===")
    
    # Khởi tạo các module lõi
    generator = VietnameseTTSGenerator(provider="edge-tts")
    mixer = AudioMixer()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_dir, "templates")
    output_base = os.path.abspath(os.path.join(current_dir, "..", "..", "dataset", "v1_0"))
    
    # 1. Sinh dữ liệu Pause Handling
    pause_template_path = os.path.join(templates_dir, "synthetic_pause_handling.json")
    if os.path.exists(pause_template_path):
        with open(pause_template_path, "r", encoding="utf-8") as f:
            pause_templates = json.load(f)
        generate_pause_handling(generator, mixer, pause_templates, output_base)
        
    # 2. Sinh dữ liệu Turn Taking
    turn_template_path = os.path.join(templates_dir, "candor_turn_taking.json")
    if os.path.exists(turn_template_path):
        with open(turn_template_path, "r", encoding="utf-8") as f:
            turn_templates = json.load(f)
        generate_turn_taking(generator, mixer, turn_templates, output_base)
        
    # 3. Sinh dữ liệu User Interruption
    interrupt_template_path = os.path.join(templates_dir, "synthetic_user_interruption.json")
    if os.path.exists(interrupt_template_path):
        with open(interrupt_template_path, "r", encoding="utf-8") as f:
            interrupt_templates = json.load(f)
        generate_user_interruption(generator, mixer, interrupt_templates, output_base)
        
    print("\n=== Hoàn thành sinh tập dữ liệu v1.0 ===")

if __name__ == "__main__":
    main()
