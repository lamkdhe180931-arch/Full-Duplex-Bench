import os
import asyncio
import tempfile
from pydub import AudioSegment

class VietnameseTTSGenerator:
    """
    Module hỗ trợ sinh giọng nói tiếng Việt từ văn bản (Text-to-Speech)
    Hỗ trợ nhiều nhà cung cấp (Providers):
    1. 'edge-tts' (Mặc định - Giọng đọc rất tự nhiên, miễn phí, không cần API key)
    2. 'gtts' (Google TTS - Miễn phí, đơn giản, giọng hơi máy)
    3. 'openai' (Yêu cầu OPENAI_API_KEY trong môi trường)
    """
    
    def __init__(self, provider="edge-tts", api_key=None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Cấu hình giọng đọc mặc định cho từng provider
        self.default_voices = {
            "edge-tts": "vi-VN-HoaiMyNeural",  # Giọng nữ Nam Bộ ấm áp, hoặc vi-VN-NamMinhNeural (giọng Nam)
            "gtts": "vi",
            "openai": "alloy"  # alloy, echo, fable, onyx, nova, shimmer
        }

    def generate(self, text, output_path, voice=None):
        """
        Hàm chính để sinh file audio .wav (16kHz, mono, 16-bit PCM)
        """
        # Tạo thư mục chứa nếu chưa có
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Chọn giọng đọc
        voice = voice or self.default_voices.get(self.provider)
        
        # Tạo một file tạm để lưu kết quả thô từ provider trước khi convert sang chuẩn wav 16kHz
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            if self.provider == "edge-tts":
                asyncio.run(self._generate_edge_tts(text, tmp_path, voice))
            elif self.provider == "gtts":
                self._generate_gtts(text, tmp_path, voice)
            elif self.provider == "openai":
                self._generate_openai(text, tmp_path, voice)
            else:
                raise ValueError(f"Không hỗ trợ provider: {self.provider}")
            
            # Chuẩn hóa file audio về định dạng benchmark chuẩn: WAV, 16000Hz, Mono
            self._convert_to_benchmark_wav(tmp_path, output_path)
            
        finally:
            # Dọn dẹp file tạm
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        print(f"[TTS] Đã sinh thành công audio ({self.provider}): {output_path}")
        return output_path

    async def _generate_edge_tts(self, text, output_path, voice):
        """Sinh audio bằng thư viện edge-tts (async)"""
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def _generate_gtts(self, text, output_path, voice):
        """Sinh audio bằng gTTS"""
        from gtts import gTTS
        tts = gTTS(text=text, lang=voice)
        tts.save(output_path)

    def _generate_openai(self, text, output_path, voice):
        """Sinh audio bằng OpenAI Audio API"""
        if not self.api_key:
            raise ValueError("Cần cung cấp OPENAI_API_KEY để sử dụng OpenAI TTS.")
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        response.stream_to_file(output_path)

    def _convert_to_benchmark_wav(self, input_path, output_path):
        """
        Chuẩn hóa file âm thanh về định dạng chuẩn của Full-Duplex-Bench:
        - Format: WAV (16-bit PCM)
        - Sample Rate: 16000 Hz
        - Channels: Mono (1 channel)
        """
        # Load audio bằng pydub
        sound = AudioSegment.from_file(input_path)
        
        # Chuyển đổi định dạng: 16kHz, mono
        sound = sound.set_frame_rate(16000).set_channels(1)
        
        # Export ra file wav PCM 16-bit
        sound.export(output_path, format="wav", codec="pcm_s16le")

# Chạy thử nghiệm nhanh nếu thực thi file trực tiếp
if __name__ == "__main__":
    # Nạp biến môi trường từ file .env nếu có
    from dotenv import load_dotenv
    load_dotenv()
    
    test_text = "Xin chào, tôi là trợ lý ảo hỗ trợ đánh giá hệ thống đàm thoại song song."
    test_output = "../test_tts.wav"
    
    # Thử nghiệm với edge-tts (mặc định)
    generator = VietnameseTTSGenerator(provider="edge-tts")
    try:
        generator.generate(test_text, test_output)
        print("Đã tạo file chạy thử thành công tại: data_generation/test_tts.wav")
    except Exception as e:
        print(f"Lỗi thử nghiệm: {e}")
        print("Lưu ý: Bạn cần cài đặt thư viện 'edge-tts' và 'pydub' trước khi chạy.")
