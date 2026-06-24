import os
from pydub import AudioSegment

class AudioMixer:
    """
    Module hỗ trợ cắt, ghép, chèn khoảng lặng và trộn đè các file âm thanh.
    Đầu ra luôn được chuẩn hóa về định dạng Benchmark (WAV, 16000Hz, Mono, 16-bit PCM).
    """

    @staticmethod
    def load_audio(path):
        """Đọc file audio và tự động chuyển về định dạng mono, 16kHz"""
        sound = AudioSegment.from_file(path)
        return sound.set_frame_rate(16000).set_channels(1)

    @staticmethod
    def save_audio(sound, output_path):
        """Lưu file audio dưới dạng WAV 16-bit PCM chuẩn của benchmark"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sound.export(output_path, format="wav", codec="pcm_s16le")

    def normalize_loudness(self, sound, target_dbfs=-20.0):
        """
        Chuẩn hóa âm lượng của file audio về mức dBFS chỉ định 
        để đảm bảo âm lượng đồng đều giữa các file.
        """
        change_in_db = target_dbfs - sound.dBFS
        return sound.apply_gain(change_in_db)

    def overlay(self, base_path, overlap_path, start_time_sec, output_path, overlap_gain=0):
        """
        Trộn đè (Mix) âm thanh overlap_path lên trên âm thanh base_path tại mốc thời gian start_time_sec.
        Thường dùng cho v1.5 (ngắt lời, backchannel, tiếng ồn nền).
        
        Args:
            base_path (str): Đường dẫn file âm thanh nền (gốc).
            overlap_path (str): Đường dẫn file âm thanh đè lên.
            start_time_sec (float): Thời điểm bắt đầu trộn đè (tính bằng giây).
            output_path (str): Đường dẫn xuất file đầu ra.
            overlap_gain (int): Điều chỉnh âm lượng (dB) của đoạn đè lên (tăng/giảm).
        """
        base_sound = self.load_audio(base_path)
        overlap_sound = self.load_audio(overlap_path)

        # Điều chỉnh âm lượng cho đoạn đè lên nếu cần
        if overlap_gain != 0:
            overlap_sound = overlap_sound.apply_gain(overlap_gain)

        # Chuyển đổi giây sang mili giây
        start_time_ms = int(start_time_sec * 1000)

        # Kiểm tra xem đoạn đè lên có vượt quá thời lượng của file gốc hay không
        overlap_end_ms = start_time_ms + len(overlap_sound)
        if overlap_end_ms > len(base_sound):
            # Chèn thêm khoảng lặng vào cuối file gốc để không bị mất đoạn cuối của overlap
            silence_needed_ms = overlap_end_ms - len(base_sound)
            silence = AudioSegment.silent(duration=silence_needed_ms, frame_rate=base_sound.frame_rate)
            base_sound = base_sound + silence

        # Thực hiện trộn đè (overlay) không lo bị cắt cụt
        mixed_sound = base_sound.overlay(overlap_sound, position=start_time_ms)

        # Lưu lại
        self.save_audio(mixed_sound, output_path)
        print(f"[Mixer] Đã trộn đè thành công tại {start_time_sec}s: {output_path}")


    def concat_with_silence(self, audio_paths, silence_dur_sec, output_path):
        """
        Ghép nối chuỗi các file audio lại với nhau, phân cách bởi một khoảng lặng.
        Thường dùng cho v1.0 Pause Handling (Người dùng nói vế 1 -> nghỉ một lúc -> nói tiếp vế 2).
        
        Args:
            audio_paths (list): Danh sách đường dẫn các file audio cần ghép nối theo thứ tự.
            silence_dur_sec (float): Độ dài khoảng lặng chèn ở giữa (tính bằng giây).
            output_path (str): Đường dẫn xuất file đầu ra.
        """
        if not audio_paths:
            raise ValueError("Danh sách audio_paths không được rỗng.")

        # Tạo khoảng lặng chuẩn hóa
        silence_segment = AudioSegment.silent(duration=int(silence_dur_sec * 1000), frame_rate=16000)
        
        combined_sound = self.load_audio(audio_paths[0])
        
        for path in audio_paths[1:]:
            next_sound = self.load_audio(path)
            # Nối tiếp: Audio trước + Khoảng lặng + Audio sau
            combined_sound = combined_sound + silence_segment + next_sound

        self.save_audio(combined_sound, output_path)
        print(f"[Mixer] Đã ghép nối {len(audio_paths)} file với {silence_dur_sec}s im lặng: {output_path}")

# Chạy thử nghiệm nhanh nếu thực thi file trực tiếp
if __name__ == "__main__":
    # Test thử nghiệm chèn khoảng lặng
    # Giả lập: test_part1.wav + 1.5 giây im lặng + test_part2.wav
    mixer = AudioMixer()
    
    # Tạo 2 âm thanh im lặng giả lập để test nối tiếp
    s1 = AudioSegment.silent(duration=1000, frame_rate=16000) # 1 giây im lặng
    s2 = AudioSegment.silent(duration=1000, frame_rate=16000) # 1 giây im lặng
    
    os.makedirs("../test_temp", exist_ok=True)
    s1.export("../test_temp/p1.wav", format="wav")
    s2.export("../test_temp/p2.wav", format="wav")
    
    try:
        mixer.concat_with_silence(
            ["../test_temp/p1.wav", "../test_temp/p2.wav"],
            silence_dur_sec=1.5,
            output_path="../test_temp/combined_test.wav"
        )
        print("Test concat_with_silence thành công.")
        
        # Dọn dẹp test
        os.remove("../test_temp/p1.wav")
        os.remove("../test_temp/p2.wav")
        os.remove("../test_temp/combined_test.wav")
        os.rmdir("../test_temp")
    except Exception as e:
        print(f"Lỗi test: {e}")
