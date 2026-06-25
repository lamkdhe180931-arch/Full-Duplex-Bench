import os
import sys
import json
from pydub import AudioSegment

# Thêm thư mục cha vào sys.path để import được core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tts_generator import VietnameseTTSGenerator
from core.audio_mixer import AudioMixer

def generate_interruption_and_backchannel(generator, mixer, templates, category, output_base):
    print(f"-> Đang sinh dữ liệu cho task v1.5: {category}")
    task_dir = os.path.join(output_base, category)
    
    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. Sinh file audio cho context_text (câu nói trước khi có Agent)
        context_path = os.path.join(sample_dir, "temp_context.wav")
        generator.generate(item["context_text"], context_path)
        
        # 2. Sinh file audio cho current_turn_text (câu nói chen/ngắt lời)
        current_path = os.path.join(sample_dir, "temp_current.wav")
        generator.generate(item["current_turn_text"], current_path)
        
        # Load audio dưới dạng AudioSegment
        context_sound = mixer.load_audio(context_path)
        current_sound = mixer.load_audio(current_path)
        
        len_context_sec = len(context_sound) / 1000.0
        len_current_ms = len(current_sound)
        
        delay_sec = item["delay_sec"]
        delay_ms = int(delay_sec * 1000)
        
        # 3. Tạo input.wav (chứa cả câu trước + 15 giây lặng, trong đó câu chen được trộn đè vào)
        silence_window = AudioSegment.silent(duration=15000, frame_rate=16000)
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
            ]
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
    bg_generator = VietnameseTTSGenerator(provider="edge-tts")
    bg_generator.default_voices["edge-tts"] = "vi-VN-NamMinhNeural"
    
    for item in templates:
        sample_id = item["id"]
        sample_dir = os.path.join(task_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. Sinh clean_input.wav (file sạch của người nói chính)
        clean_input_wav_path = os.path.join(sample_dir, "clean_input.wav")
        generator.generate(item["current_turn_text"], clean_input_wav_path)
        
        # 2. Sinh file audio nhiễu (overlap_text) bằng giọng đọc khác
        overlap_path = os.path.join(sample_dir, "temp_overlap.wav")
        bg_generator.generate(item["overlap_text"], overlap_path)
        
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
            ]
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        # 6. Dọn dẹp file tạm
        if os.path.exists(overlap_path):
            os.remove(overlap_path)

def main():
    print("=== Khởi chạy sinh dữ liệu Full-Duplex-Bench v1.5 (Tiếng Việt) ===")
    
    # Khởi tạo các module lõi (Mặc định dùng giọng nữ vi-VN-HoaiMyNeural làm giọng chính)
    generator = VietnameseTTSGenerator(provider="edge-tts")
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
        
    # 3. Sinh dữ liệu Talking to Other
    talk_other_path = os.path.join(templates_dir, "talking_to_other.json")
    if os.path.exists(talk_other_path):
        with open(talk_other_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        generate_other_and_background(generator, mixer, templates, "talking_to_other", output_base)
        
    # 4. Sinh dữ liệu Background Speech
    bg_speech_path = os.path.join(templates_dir, "background_speech.json")
    if os.path.exists(bg_speech_path):
        with open(bg_speech_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        generate_other_and_background(generator, mixer, templates, "background_speech", output_base)
        
    print("\n=== Hoàn thành sinh tập dữ liệu v1.5 ===")

if __name__ == "__main__":
    main()
