import time
import threading
import queue
import sys
import matplotlib.pyplot as plt
from llama_cpp import Llama

# Choose the model you are using
MODEL_PATH = "models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

def load_model():
    print("Loading model for realtime graph (this might take a moment)...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,      # Change according to your CPU
        n_batch=512,
        n_gpu_layers=0,   # CPU only
        verbose=False,
    )
    print("Model loaded.")
    return llm

def inference_thread(llm, update_queue):
    SYSTEM_PROMPT = "You are Qwen2.5-Coder, a helpful AI programming assistant."
    print("Type 'exit' to quit.\n")
    
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

            print("\nAssistant: ", end="", flush=True)
            
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

            print("\n")
        except Exception as e:
            print(f"\nError in inference thread: {e}")
            break

if __name__ == "__main__":
    try:
        llm = load_model()
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}. Error: {e}")
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
