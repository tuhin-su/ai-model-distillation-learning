import time
import threading
import queue
import sys
import os
import matplotlib.pyplot as plt
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

# Choose the model you are using
MODEL_PATH = "models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

def load_model():
    console.print("[bold blue]Loading model for realtime graph (on GPU)...[/bold blue]")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,      # Change according to your CPU/GPU setup
        n_batch=512,
        n_gpu_layers=-1,  # Offload all layers to GPU
        verbose=False,
    )
    console.print("[bold green]Model loaded.[/bold green]")
    return llm

def inference_thread(llm, update_queue):
    SYSTEM_PROMPT = "You are Qwen2.5-Coder, a helpful AI programming assistant."
    console.print("Type 'exit' to quit.\n")
    
    while True:
        try:
            # We take input in the background thread. 
            user = input("You: ")
            if user.lower() in ("exit", "quit"):
                update_queue.put("QUIT")
                break

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]

            console.print("\n[bold cyan]Assistant: [/bold cyan]", end="", flush=True)
            
            start_time = time.time()
            token_count = 0
            
            # Signal the GUI thread to clear the graph for a new generation
            update_queue.put(("RESET", None, None))
            
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=1024,
                stream=True
            )

            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    content = delta["content"]
                    print(content, end="", flush=True)
                    
                    token_count += 1
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        speed = token_count / elapsed
                        # Send the new data point to the GUI thread
                        update_queue.put(("UPDATE", elapsed, speed))

            elapsed = time.time() - start_time
            speed = token_count / elapsed if elapsed > 0 else 0
            console.print(
                f"\n[dim yellow]Response time: {elapsed:.2f}s | "
                f"Tokens generated: {token_count} | "
                f"Speed: {speed:.2f} tok/s[/dim yellow]\n"
            )
        except Exception as e:
            console.print(f"\n[bold red]Error in inference thread: {e}[/bold red]")
            break

if __name__ == "__main__":
    try:
        llm = load_model()
    except Exception as e:
        console.print(f"[bold red]Failed to load model from {MODEL_PATH}. Error: {e}[/bold red]")
        sys.exit(1)
        
    update_queue = queue.Queue()
    
    # Run the model inference in a separate thread so it doesn't block the GUI
    t = threading.Thread(target=inference_thread, args=(llm, update_queue), daemon=True)
    t.start()
    
    # Set up matplotlib for interactive drawing
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Model Metrics")
    ax.set_title("Real-time Token Generation Speed")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (tokens/sec)")
    line, = ax.plot([], [], 'b-', linewidth=2)
    
    times = []
    speeds = []
    
    plt.show(block=False)
    
    # GUI loop runs in the main thread
    while t.is_alive():
        try:
            # Process any queued updates from the inference thread
            updated = False
            while not update_queue.empty():
                msg = update_queue.get_nowait()
                if msg == "QUIT":
                    plt.close('all')
                    sys.exit(0)
                
                action, x, y = msg
                if action == "RESET":
                    times.clear()
                    speeds.clear()
                    line.set_data([], [])
                    updated = True
                elif action == "UPDATE":
                    times.append(x)
                    speeds.append(y)
                    line.set_data(times, speeds)
                    updated = True
            
            if updated and len(times) > 0:
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
            
            # Keep the GUI responsive
            fig.canvas.flush_events()
            time.sleep(0.05)
            
            # If the window is closed manually by the user, we exit
            if not plt.fignum_exists(fig.number):
                print("\nGraph window closed. Exiting...")
                sys.exit(0)
                
        except KeyboardInterrupt:
            break
        except Exception:
            pass
