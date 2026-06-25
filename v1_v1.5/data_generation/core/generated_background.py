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
    "audioldm2_cli",
}


class GeneratedBackgroundGenerator:
    """
    Text-to-audio wrapper for generating background speech/noise.

    Providers:
    - elevenlabs_sfx: calls ElevenLabs Sound Effects API.
    - audiogen_cli: runs a configured local AudioGen command.
    - audioldm2_cli: runs a configured local AudioLDM 2 command.
    """

    def __init__(self, provider=None, prompt_influence=None, timeout_sec=180):
        self.provider = (provider or os.getenv("FDB_BACKGROUND_PROVIDER") or "elevenlabs_sfx").lower()
        self.prompt_influence = float(
            prompt_influence
            if prompt_influence is not None
            else os.getenv("ELEVENLABS_SFX_PROMPT_INFLUENCE", "0.35")
        )
        self.timeout_sec = timeout_sec

    def generate(self, prompt, output_path, duration_sec, seed=None, provider=None):
        provider = (provider or self.provider).lower()
        if provider not in BACKGROUND_AUDIO_PROVIDERS:
            raise ValueError(
                f"Unsupported background provider: {provider}. "
                f"Use one of: {', '.join(sorted(BACKGROUND_AUDIO_PROVIDERS))}."
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
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
