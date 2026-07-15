#!/usr/bin/env python3
"""
flux1_gpu.py
------------
Generates an image using the quantized FLUX.1-schnell GGUF model on GPU (CUDA).
This script uses the Hugging Face diffusers library to load the GGUF transformer
and other components, runs the inference on GPU, and displays the image.
"""

import os
import sys
import time
import torch
from PIL import Image
from rich.console import Console

console = Console()

# Ensure required libraries can be imported
try:
    from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
except ImportError as e:
    console.print(f"[bold red]Error importing required packages: {e}[/bold red]")
    console.print("Please ensure diffusers, transformers, accelerate, and gguf are installed in your environment.")
    console.print("You can run: pip install diffusers transformers accelerate gguf")
    sys.exit(1)

# Paths
MODEL_PATH = "models/flux1-schnell-Q4_1.gguf"

if not os.path.exists(MODEL_PATH):
    console.print(f"[bold red]Error: Model file not found at '{MODEL_PATH}'.[/bold red]")
    sys.exit(1)

def main():
    # 1. Ask user for prompt
    console.print("\n[bold cyan]================ FLUX.1 Schnell GPU Generator ================[/bold cyan]")
    prompt = input("Enter prompt: ").strip()
    if not prompt:
        prompt = "A majestic dragon sitting on top of a mountain peak, photorealistic"
        console.print(f"No prompt entered. Using default: '[italic]{prompt}[/italic]'")

    console.print("\n[bold]1. Loading quantized transformer from GGUF...[/bold]")
    console.print(f"   Source: {MODEL_PATH}")
    
    # Load transformer with bfloat16 computation on GPU
    transformer = FluxTransformer2DModel.from_single_file(
        MODEL_PATH,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    console.print("   [green][✓] Transformer loaded successfully.[/green]")

    console.print("\n[bold]2. Initializing Flux pipeline (downloading components if needed)...[/bold]")
    # Load the full pipeline using the quantized transformer
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    
    # Ensure it's on GPU (CUDA)
    pipe.to("cuda")
    console.print("   [green][✓] Pipeline initialized on GPU (CUDA).[/green]")

    # 3. Generate image
    console.print("\n[bold]3. Generating image on GPU...[/bold]")
    console.print("   Running 4 inference steps (FLUX Schnell default)...")
    
    start_time = time.perf_counter()
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            guidance_scale=0.0,           # Required for Schnell
            num_inference_steps=4,         # Required for Schnell (usually 4 steps)
            max_sequence_length=256,
            generator=torch.Generator("cuda").manual_seed(42),
        )
    elapsed_time = time.perf_counter() - start_time
        
    image = output.images[0]
    console.print(f"   [green][✓] Image generation complete.[/green]")
    console.print(f"   [bold yellow]Response time (generation): {elapsed_time:.2f} seconds[/bold yellow]")

    # 4. Display the image (no saving, as requested)
    console.print("\n[bold]4. Displaying the image...[/bold]")
    image.show()
    console.print("   Displaying image complete. (Not saved to disk).")
    console.print("[bold cyan]==============================================================[/bold cyan]\n")

if __name__ == "__main__":
    main()
