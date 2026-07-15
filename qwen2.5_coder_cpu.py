from llama_cpp import Llama

MODEL_PATH = "models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

print("Loading model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,      # Change according to your CPU
    n_batch=512,
    n_gpu_layers=0,   # CPU only
    verbose=False,
)

print("Model loaded.")
print("Type 'exit' to quit.\n")

SYSTEM_PROMPT = (
    "You are Qwen2.5-Coder, a helpful AI programming assistant."
)

while True:
    user = input("You: ")

    if user.lower() in ("exit", "quit"):
        break

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        max_tokens=1024,
    )

    print("\nAssistant:")
    print(response["choices"][0]["message"]["content"])
    print()
