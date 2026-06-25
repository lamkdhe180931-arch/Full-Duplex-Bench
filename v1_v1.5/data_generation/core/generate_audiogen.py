#!/usr/bin/env python3
import argparse
import os
import torch
import scipy.io.wavfile
from transformers import AudiogenForConditionalGeneration, AutoProcessor

def main():
    parser = argparse.ArgumentParser(description="Generate background audio using AudioGen on a specific GPU")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of the generated audio in seconds")
    parser.add_argument("--output", type=str, required=True, help="Output path for the generated WAV file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run inference on (default: cuda:1)")
    
    args = parser.parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f"Loading facebook/audiogen-medium model onto {args.device}...")
    
    # Load model and processor
    # We load in float16 if cuda is used to save VRAM and run faster
    dtype = torch.float16 if "cuda" in args.device else torch.float32
    
    processor = AutoProcessor.from_pretrained("facebook/audiogen-medium")
    model = AudiogenForConditionalGeneration.from_pretrained(
        "facebook/audiogen-medium", 
        torch_dtype=dtype
    ).to(args.device)

    print(f"Generating audio for prompt: '{args.prompt}' (Duration: {args.duration}s)...")
    
    # Preprocess prompt
    inputs = processor(text=[args.prompt], return_tensors="pt").to(args.device)
    
    # Calculate target length in tokens. Standard AudioGen frame rate is 50 Hz.
    # So 1 second of audio corresponds to 50 tokens.
    max_new_tokens = int(args.duration * 50)
    
    # Generate
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Save to file (sampling rate of AudioGen is 16000 Hz)
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].cpu().numpy()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Write to WAV
    scipy.io.wavfile.write(args.output, rate=sampling_rate, data=audio_data)
    print(f"Audio generated successfully and saved to: {args.output}")

if __name__ == "__main__":
    main()
