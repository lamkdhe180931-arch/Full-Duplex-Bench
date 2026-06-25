#!/usr/bin/env python3
"""Generate background audio using Meta's AudioGen model via the audiocraft library.

Official API reference: https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md

Requirements:
    pip install -U audiocraft
    (audiocraft requires Python 3.9+ and PyTorch 2.1.0+)

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
    # Ref: https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md
    try:
        from audiocraft.models import AudioGen
        from audiocraft.data.audio import audio_write
    except ImportError:
        raise ImportError(
            "audiocraft is required for AudioGen. Install it with:\n"
            "  pip install -U audiocraft\n"
            "Note: audiocraft requires Python 3.9+ and PyTorch 2.1.0+\n"
            "On Kaggle/Colab:\n"
            "  !pip install -U audiocraft"
        )

    print(f"Loading facebook/audiogen-medium model onto {args.device}...")
    model = AudioGen.get_pretrained("facebook/audiogen-medium", device=args.device)
    model.set_generation_params(duration=args.duration)

    print(f"Generating audio for prompt: '{args.prompt}' (Duration: {args.duration}s)...")
    with torch.no_grad():
        wav = model.generate([args.prompt])  # shape: (1, 1, num_samples)

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Use audiocraft's official audio_write with loudness normalization
    # audio_write expects path without extension, adds .wav automatically
    output_stem = args.output
    if output_stem.endswith(".wav"):
        output_stem = output_stem[:-4]

    audio_write(
        output_stem,
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )
    print(f"Audio generated successfully and saved to: {args.output}")


if __name__ == "__main__":
    main()
