#!/usr/bin/env python3
"""
flux1_cpu.py
------------
Generates an image using the quantized FLUX.1-schnell GGUF model on CPU.
This script uses the Hugging Face diffusers library to load the GGUF transformer
and other components, runs the inference on CPU, and displays the image.
"""

import os
import sys
import torch
from PIL import Image

# Ensure required libraries can be imported
try:
    from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please ensure diffusers, transformers, accelerate, and gguf are installed in your environment.")
    print("You can run: pip install diffusers transformers accelerate gguf")
    sys.exit(1)

# Paths
MODEL_PATH = "models/flux1-schnell-Q4_1.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    sys.exit(1)

def main():
    # 1. Ask user for prompt
    print("\n================ FLUX.1 Schnell CPU Generator ================")
    prompt = input("Enter prompt: ").strip()
    if not prompt:
        prompt = "A majestic dragon sitting on top of a mountain peak, photorealistic"
        print(f"No prompt entered. Using default: '{prompt}'")

    print("\n1. Loading quantized transformer from GGUF...")
    print(f"   Source: {MODEL_PATH}")
    
    # Load transformer with float32 computation on CPU
    transformer = FluxTransformer2DModel.from_single_file(
        MODEL_PATH,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32),
        torch_dtype=torch.float32,
    )
    print("   [✓] Transformer loaded successfully.")

    print("\n2. Initializing Flux pipeline (downloading components if needed)...")
    # Load the full pipeline using the quantized transformer
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=torch.float32,
    )
    
    # Ensure it's on CPU
    pipe.to("cpu")
    print("   [✓] Pipeline initialized on CPU.")

    # 3. Generate image
    print("\n3. Generating image (this will take several minutes on CPU)...")
    print("   Running 4 inference steps (FLUX Schnell default)...")
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            guidance_scale=0.0,           # Required for Schnell
            num_inference_steps=4,         # Required for Schnell (usually 4 steps)
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(42),
        )
        
    image = output.images[0]
    print("   [✓] Image generation complete.")

    # 4. Display the image (no saving, as requested)
    print("\n4. Displaying the image...")
    image.show()
    print("   Displaying image complete. (Not saved to disk).")
    print("==============================================================\n")

if __name__ == "__main__":
    main()
