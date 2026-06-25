import json
import os
import shlex
import subprocess
import tempfile
import urllib.request

from pydub import AudioSegment


BACKGROUND_AUDIO_PROVIDERS = {
    "elevenlabs_sfx",
    "audiogen_cli",
    "audiogen_inline",
    "audioldm2_cli",
}


# Lazy-loaded AudioGen model singleton to avoid reloading on every sample.
_AUDIOGEN_MODEL = None
_AUDIOGEN_DEVICE = None


def _get_audiogen_model(device=None):
    """Load AudioGen model once and cache it for reuse."""
    global _AUDIOGEN_MODEL, _AUDIOGEN_DEVICE

    if device is None:
        import torch
        device = "cuda:1" if torch.cuda.is_available() else "cpu"

    if _AUDIOGEN_MODEL is not None and _AUDIOGEN_DEVICE == device:
        return _AUDIOGEN_MODEL

    try:
        from audiocraft.models import AudioGen
    except ImportError:
        raise ImportError(
            "audiocraft is required for audiogen_inline provider. Install with:\n"
            "  pip install -U git+https://github.com/facebookresearch/audiocraft\n"
            "Requires Python 3.9+ and PyTorch 2.1.0+"
        )

    print(f"[AudioGen] Loading facebook/audiogen-medium onto {device}...")
    _AUDIOGEN_MODEL = AudioGen.get_pretrained("facebook/audiogen-medium", device=device)
    _AUDIOGEN_DEVICE = device
    print("[AudioGen] Model loaded.")
    return _AUDIOGEN_MODEL


class GeneratedBackgroundGenerator:
    """
    Text-to-audio wrapper for generating background speech/noise.

    Providers:
    - elevenlabs_sfx: calls ElevenLabs Sound Effects API.
    - audiogen_inline: uses audiocraft's AudioGen directly in-process (recommended for Kaggle).
    - audiogen_cli: runs a configured local AudioGen command via subprocess.
    - audioldm2_cli: runs a configured local AudioLDM 2 command via subprocess.
    """

    def __init__(self, provider=None, prompt_influence=None, timeout_sec=180, device=None):
        self.provider = (provider or os.getenv("FDB_BACKGROUND_PROVIDER") or "elevenlabs_sfx").lower()
        self.prompt_influence = float(
            prompt_influence
            if prompt_influence is not None
            else os.getenv("ELEVENLABS_SFX_PROMPT_INFLUENCE", "0.35")
        )
        self.timeout_sec = timeout_sec
        self.device = device or os.getenv("FDB_AUDIOGEN_DEVICE")

    def generate(self, prompt, output_path, duration_sec, seed=None, provider=None):
        provider = (provider or self.provider).lower()
        if provider not in BACKGROUND_AUDIO_PROVIDERS:
            raise ValueError(
                f"Unsupported background provider: {provider}. "
                f"Use one of: {', '.join(sorted(BACKGROUND_AUDIO_PROVIDERS))}."
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if provider == "audiogen_inline":
            provider_metadata = self._generate_audiogen_inline(
                prompt, output_path, duration_sec, seed
            )
            provider_metadata.update(
                {
                    "provider": provider,
                    "prompt": prompt,
                    "duration_sec": duration_sec,
                    "seed": seed,
                    "output_path": os.path.abspath(output_path),
                }
            )
            return provider_metadata

        suffix = ".mp3" if provider == "elevenlabs_sfx" else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            raw_path = tmp_file.name

        try:
            if provider == "elevenlabs_sfx":
                provider_metadata = self._generate_elevenlabs_sfx(prompt, raw_path, duration_sec)
            elif provider == "audiogen_cli":
                provider_metadata = self._generate_cli_provider(
                    "AUDIOGEN_CMD",
                    prompt,
                    raw_path,
                    duration_sec,
                    seed,
                )
            else:
                provider_metadata = self._generate_cli_provider(
                    "AUDIOLDM2_CMD",
                    prompt,
                    raw_path,
                    duration_sec,
                    seed,
                )

            self._convert_to_benchmark_wav(raw_path, output_path)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)

        provider_metadata.update(
            {
                "provider": provider,
                "prompt": prompt,
                "duration_sec": duration_sec,
                "seed": seed,
                "output_path": os.path.abspath(output_path),
            }
        )
        return provider_metadata

    def _generate_audiogen_inline(self, prompt, output_path, duration_sec, seed):
        """Generate background audio using AudioGen directly in-process (no subprocess)."""
        import torch
        import numpy as np

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model = _get_audiogen_model(device=self.device)
        model.set_generation_params(duration=duration_sec)

        print(f"[AudioGen] Generating: '{prompt[:80]}...' ({duration_sec}s)")
        with torch.no_grad():
            wav = model.generate([prompt])  # (1, 1, num_samples)

        sampling_rate = model.sample_rate  # 16000
        audio_data = wav[0].cpu().numpy()  # (1, num_samples)
        if audio_data.ndim > 1:
            audio_data = audio_data[0]

        # Normalize to int16 for WAV
        peak = float(np.max(np.abs(audio_data)))
        if peak > 0:
            audio_data = audio_data / peak * 0.95
        audio_int16 = (audio_data * 32767).astype(np.int16)

        import scipy.io.wavfile
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_int16)
        print(f"[AudioGen] Saved: {output_path}")

        return {
            "model_id": "facebook/audiogen-medium",
            "device": str(self.device),
            "sample_rate": sampling_rate,
        }

    def _generate_elevenlabs_sfx(self, prompt, output_path, duration_sec):
        api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("XI_API_KEY")
        if not api_key:
            raise ValueError("Set ELEVENLABS_API_KEY or XI_API_KEY to use elevenlabs_sfx.")

        endpoint = os.getenv("ELEVENLABS_SFX_ENDPOINT", "https://api.elevenlabs.io/v1/sound-generation")
        payload = {
            "text": prompt,
            "duration_seconds": float(duration_sec),
            "prompt_influence": self.prompt_influence,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key,
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            audio_bytes = response.read()

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return {
            "endpoint": endpoint,
            "prompt_influence": self.prompt_influence,
        }

    def _generate_cli_provider(self, env_var, prompt, output_path, duration_sec, seed):
        command_template = os.getenv(env_var)
        if not command_template:
            raise ValueError(
                f"Set {env_var} to run this provider. Example: "
                f"python scripts/generate_audio.py --prompt {{prompt}} "
                f"--duration {{duration}} --output {{output}} --seed {{seed}}"
            )

        mapping = {
            "prompt": prompt,
            "duration": str(duration_sec),
            "output": output_path,
            "seed": "0" if seed is None else str(seed),
        }
        args = [part.format(**mapping) for part in shlex.split(command_template)]
        subprocess.run(args, check=True, timeout=self.timeout_sec)

        return {
            "command_env": env_var,
            "command": command_template,
        }

    @staticmethod
    def _convert_to_benchmark_wav(input_path, output_path):
        sound = AudioSegment.from_file(input_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(output_path, format="wav", codec="pcm_s16le")
