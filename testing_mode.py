import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "black-forest-labs/FLUX.1-dev"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print("Ready!")

# Single prompt
user = "Fuck you!"

messages = [
    {"role": "user", "content": user}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

reply = tokenizer.decode(
    output[0][inputs.input_ids.shape[-1]:],
    skip_special_tokens=True,
)

print("\nAI:", reply)