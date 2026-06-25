import os
import asyncio
import random
import tempfile
import threading
from pydub import AudioSegment


EDGE_TTS_PROFILES = {
    "primary": [
        {"voice": "vi-VN-HoaiMyNeural", "rate": "+0%", "volume": "+0%", "pitch": "+0Hz"},
        {"voice": "vi-VN-HoaiMyNeural", "rate": "-4%", "volume": "-1%", "pitch": "-4Hz"},
        {"voice": "vi-VN-NamMinhNeural", "rate": "+1%", "volume": "-1%", "pitch": "+0Hz"},
    ],
    "user_interruption": [
        {"voice": "vi-VN-HoaiMyNeural", "rate": "+8%", "volume": "+2%", "pitch": "+8Hz"},
        {"voice": "vi-VN-HoaiMyNeural", "rate": "+12%", "volume": "+1%", "pitch": "+12Hz"},
        {"voice": "vi-VN-NamMinhNeural", "rate": "+9%", "volume": "+1%", "pitch": "+6Hz"},
    ],
    "user_backchannel": [
        {"voice": "vi-VN-HoaiMyNeural", "rate": "-8%", "volume": "-5%", "pitch": "-8Hz"},
        {"voice": "vi-VN-HoaiMyNeural", "rate": "-4%", "volume": "-4%", "pitch": "+2Hz"},
        {"voice": "vi-VN-NamMinhNeural", "rate": "-6%", "volume": "-5%", "pitch": "-4Hz"},
    ],
    "talking_to_other": [
        {"voice": "vi-VN-NamMinhNeural", "rate": "+4%", "volume": "-2%", "pitch": "+2Hz"},
        {"voice": "vi-VN-HoaiMyNeural", "rate": "+3%", "volume": "-3%", "pitch": "-2Hz"},
    ],
    "background_speech": [
        {"voice": "vi-VN-NamMinhNeural", "rate": "+0%", "volume": "-6%", "pitch": "-4Hz"},
        {"voice": "vi-VN-HoaiMyNeural", "rate": "-2%", "volume": "-7%", "pitch": "+4Hz"},
    ],
    "secondary": [
        {"voice": "vi-VN-NamMinhNeural", "rate": "+0%", "volume": "-2%", "pitch": "+0Hz"},
        {"voice": "vi-VN-HoaiMyNeural", "rate": "+2%", "volume": "-2%", "pitch": "-2Hz"},
    ],
}

EDGE_TTS_DEFAULT_PROFILE = {
    "voice": "vi-VN-HoaiMyNeural",
    "rate": "+0%",
    "volume": "+0%",
    "pitch": "+0Hz",
}

class VietnameseTTSGenerator:
    """
    Module hỗ trợ sinh giọng nói tiếng Việt từ văn bản (Text-to-Speech)
    Hỗ trợ nhiều nhà cung cấp (Providers):
    1. 'edge-tts' (Mặc định - Giọng đọc rất tự nhiên, miễn phí, không cần API key)
    2. 'gtts' (Google TTS - Miễn phí, đơn giản, giọng hơi máy)
    3. 'openai' (Yêu cầu OPENAI_API_KEY trong môi trường)
    """
    
    def __init__(self, provider="edge-tts", api_key=None, seed=None, profiles=None, max_retries=2):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.seed = seed
        self._rng = random.Random(seed)
        self.profiles = profiles or EDGE_TTS_PROFILES
        self.max_retries = max_retries
        self.last_synthesis = None
        
        # Cấu hình giọng đọc mặc định cho từng provider
        self.default_voices = {
            "edge-tts": "vi-VN-HoaiMyNeural",  # Giọng nữ Nam Bộ ấm áp, hoặc vi-VN-NamMinhNeural (giọng Nam)
            "gtts": "vi",
            "openai": "alloy"  # alloy, echo, fable, onyx, nova, shimmer
        }

    def select_profile(self, role="primary"):
        """Chọn profile TTS theo vai trò hội thoại, có thể tái lập bằng seed."""
        choices = (
            self.profiles.get(role)
            or self.profiles.get("secondary")
            or [EDGE_TTS_DEFAULT_PROFILE]
        )
        return dict(self._rng.choice(choices))

    def _resolve_profile(self, voice=None, profile=None, role="primary"):
        if self.provider != "edge-tts":
            selected = {"voice": voice or self.default_voices.get(self.provider)}
            return selected

        selected = dict(profile) if profile is not None else self.select_profile(role)
        if voice:
            selected["voice"] = voice

        resolved = dict(EDGE_TTS_DEFAULT_PROFILE)
        resolved.update(selected)
        return resolved

    @staticmethod
    def _is_retryable_tts_error(exc):
        retryable_names = {
            "NoAudioReceived",
            "ClientConnectorError",
            "ClientConnectorDNSError",
            "TimeoutError",
        }
        if exc.__class__.__name__ in retryable_names:
            return True

        message = str(exc).lower()
        return (
            "no audio was received" in message
            or "cannot connect" in message
            or "timed out" in message
        )

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def generate(self, text, output_path, voice=None, profile=None, role="primary"):
        """
        Hàm chính để sinh file audio .wav (16kHz, mono, 16-bit PCM)
        """
        # Tạo thư mục chứa nếu chưa có
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        synthesis_profile = self._resolve_profile(voice=voice, profile=profile, role=role)
        voice = synthesis_profile.get("voice")
        self.last_synthesis = {
            "provider": self.provider,
            "role": role,
            "profile": dict(synthesis_profile),
        }
        
        # Tạo một file tạm để lưu kết quả thô từ provider trước khi convert sang chuẩn wav 16kHz
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            attempt = 0
            while True:
                try:
                    if self.provider == "edge-tts":
                        self._run_async(
                            self._generate_edge_tts(
                                text,
                                tmp_path,
                                voice,
                                rate=synthesis_profile["rate"],
                                volume=synthesis_profile["volume"],
                                pitch=synthesis_profile["pitch"],
                            )
                        )
                    elif self.provider == "gtts":
                        self._generate_gtts(text, tmp_path, voice)
                    elif self.provider == "openai":
                        self._generate_openai(text, tmp_path, voice)
                    else:
                        raise ValueError(f"Không hỗ trợ provider: {self.provider}")
                    break
                except Exception as exc:
                    if (
                        self.provider != "edge-tts"
                        or attempt >= self.max_retries
                        or not self._is_retryable_tts_error(exc)
                    ):
                        raise

                    attempt += 1
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    print(
                        f"[TTS] Thử lại Edge TTS lần {attempt}/{self.max_retries} "
                        f"do lỗi tạm thời: {exc}"
                    )
            
            # Chuẩn hóa file audio về định dạng benchmark chuẩn: WAV, 16000Hz, Mono
            self._convert_to_benchmark_wav(tmp_path, output_path)
            
        finally:
            # Dọn dẹp file tạm
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        print(f"[TTS] Đã sinh thành công audio ({self.provider}): {output_path}")
        return output_path

    async def _generate_edge_tts(self, text, output_path, voice, rate="+0%", volume="+0%", pitch="+0Hz"):
        """Sinh audio bằng thư viện edge-tts (async)"""
        import edge_tts
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )
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
