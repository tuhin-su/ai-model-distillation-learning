from datasets import load_dataset
from rich import print
from rich.progress import track
import json
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL       = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_FILE = "output_data.jsonl"
MAX_SAMPLES = 50          # set to None to run the full dataset
MAX_NEW_TOKENS = 200
# ─────────────────────────────────────────────────────────────────────────────

print("[bold cyan]Loading tokenizer & model…[/bold cyan]")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
)
print(f"[green]Model ready:[/green] {MODEL}\n")

# Load dataset
ds = load_dataset("microsoft/orca-agentinstruct-1M-v1")


def generate_reply(conversation: list[dict]) -> str:
    """Run the model on a conversation history and return the assistant reply."""
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        output[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    return reply.strip()


def process_sample(sample: dict, sample_idx: int) -> list[dict]:
    """
    Walk through all messages in a sample.
    For every user/system turn, get the model's reply and record the pair.
    Returns a list of {prompt, ai_response} records.
    """
    raw_messages = json.loads(sample["messages"])

    # Filter out messages with empty content
    messages = [m for m in raw_messages if m.get("content", "").strip()]

    records = []
    conversation_so_far = []

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]

        if role in ("system", "user"):
            conversation_so_far.append(msg)

            # Ask the model after every user/system turn
            reply = generate_reply(conversation_so_far)

            record = {
                "sample_idx":  sample_idx,
                "timestamp":   datetime.datetime.utcnow().isoformat(),
                "role":        role,
                "prompt":      content,
                "ai_response": reply,
            }
            records.append(record)

            # Append model reply to keep conversation coherent
            conversation_so_far.append({"role": "assistant", "content": reply})

        elif role == "assistant":
            # Use ground-truth assistant turn to stay faithful to the dataset
            conversation_so_far.append(msg)

    return records


# ── Main loop ─────────────────────────────────────────────────────────────────
# Open file in append mode — safe to re-run without losing previous results
with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for split_name, dataset in ds.items():
        samples = (
            dataset
            if MAX_SAMPLES is None
            else dataset.select(range(min(MAX_SAMPLES, len(dataset))))
        )

        print(f"\n[bold yellow]Split:[/bold yellow] {split_name}  |  Samples: [cyan]{len(samples):,}[/cyan]")

        total_records = 0
        for idx, sample in enumerate(track(samples, description=f"Processing {split_name}…")):
            try:
                records = process_sample(sample, idx)
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                total_records += len(records)

                # Live preview of last record
                if records:
                    last = records[-1]
                    print(f"\n[dim]Sample {idx}[/dim]")
                    print(f"  [bold]Prompt:[/bold]   {last['prompt'][:120]}…")
                    print(f"  [bold]AI reply:[/bold] {last['ai_response'][:120]}…")

            except Exception as e:
                print(f"[red]Error on sample {idx}:[/red] {e}")
                continue

        print(f"\n[green]✓ Done.[/green] Wrote [cyan]{total_records}[/cyan] records → [bold]{OUTPUT_FILE}[/bold]")
