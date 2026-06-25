#!/usr/bin/env python3
"""Generate background audio using Meta's AudioGen model via the audiocraft library.

Usage (called as subprocess by GeneratedBackgroundGenerator):
    python generate_audiogen.py \
        --prompt "muffled speech in a cafe" \
        --duration 5.0 \
        --output /tmp/background.wav \
        --seed 42 \
        --device cuda:1
"""
import argparse
import os
import torch
import scipy.io.wavfile
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate background audio using AudioGen on a specific GPU"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--output", type=str, required=True, help="Output WAV path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda:1",
        help="Device to run inference on (default: cuda:1)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # ── Load AudioGen via audiocraft (Meta's official library) ──
    try:
        from audiocraft.models import AudioGen
    except ImportError:
        raise ImportError(
            "audiocraft is required for AudioGen. Install it with:\n"
            "  pip install audiocraft\n"
            "Or on Kaggle/Colab:\n"
            "  !pip install audiocraft"
        )

    print(f"Loading facebook/audiogen-medium model onto {args.device}...")
    model = AudioGen.get_pretrained("facebook/audiogen-medium", device=args.device)
    model.set_generation_params(duration=args.duration)

    print(f"Generating audio for prompt: '{args.prompt}' (Duration: {args.duration}s)...")
    with torch.no_grad():
        wav = model.generate([args.prompt])  # shape: (1, 1, num_samples)

    # AudioGen outputs at 16000 Hz
    sampling_rate = model.sample_rate  # 16000
    audio_data = wav[0].cpu().numpy()  # shape: (1, num_samples)

    # Flatten to 1D if needed
    if audio_data.ndim > 1:
        audio_data = audio_data[0]

    # Normalize to int16 range for scipy wav write
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak * 0.95
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Write to WAV
    scipy.io.wavfile.write(args.output, rate=sampling_rate, data=audio_int16)
    print(f"Audio generated successfully and saved to: {args.output}")


if __name__ == "__main__":
    main()
