import torch
from diffusers import FluxPipeline

MODEL = "black-forest-labs/FLUX.1-dev"

print("Loading model...")

pipe = FluxPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
)

# Save VRAM
pipe.enable_model_cpu_offload()

prompt = """
A futuristic cyberpunk city at night,
heavy rain,
neon reflections,
cinematic lighting,
ultra detailed,
8k
"""

print("Generating...")

image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=30,
    max_sequence_length=512,
).images[0]

image.save("output.png")

print("Saved as output.png")