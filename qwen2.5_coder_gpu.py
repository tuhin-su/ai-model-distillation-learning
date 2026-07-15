import os
import sys
import time
from rich.console import Console

console = Console()

# Self-exec with LD_PRELOAD to prevent C++ ABI segmentation fault in llama-cpp-python
if "LD_PRELOAD" not in os.environ or "libstdc++.so.6" not in os.environ["LD_PRELOAD"]:
    os.environ["LD_PRELOAD"] = "/usr/lib/libstdc++.so.6"
    try:
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
    except Exception as e:
        console.print(f"[bold yellow]Warning: Failed to self-exec with LD_PRELOAD: {e}[/bold yellow]", style="red")

from llama_cpp import Llama

MODEL_PATH = "models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

console.print("[bold blue]Loading model on GPU...[/bold blue]")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,      # Change according to your CPU/GPU setup
    n_batch=512,
    n_gpu_layers=-1,  # Offload all layers to GPU
    verbose=False,
)

console.print("[bold green]Model loaded.[/bold green]")
console.print("Type 'exit' to quit.\n")

SYSTEM_PROMPT = (
    "You are Qwen2.5-Coder, a helpful AI programming assistant."
)

while True:
    try:
        user = input("You: ")
    except (KeyboardInterrupt, EOFError):
        break

    if user.lower() in ("exit", "quit"):
        break

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    start_time = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        max_tokens=1024,
    )
    elapsed_time = time.perf_counter() - start_time

    # Get token stats
    usage = response.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tokens_per_sec = comp_tokens / elapsed_time if elapsed_time > 0 else 0

    console.print("\n[bold cyan]Assistant:[/bold cyan]")
    console.print(response["choices"][0]["message"]["content"])
    console.print(
        f"\n[dim yellow]Response time: {elapsed_time:.2f}s | "
        f"Tokens generated: {comp_tokens} | "
        f"Speed: {tokens_per_sec:.2f} tok/s[/dim yellow]\n"
    )
