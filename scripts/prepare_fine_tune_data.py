"""Prepare fine-tuning dataset for QWEN 0.5B.
Combines TinyStories + folk tales into instruction-response format."""
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset

OUTPUT = os.path.join(os.path.dirname(__file__), "..",
                      "data", "train_dataset.jsonl")

# ── 1. Folk tales from contract ──────────────────────────────────────────
print("Loading folk tales...")
from melm.contracts.validation import load_folk_tales
folk = load_folk_tales()
folk_stories = folk.get("stories", [])
print(f"  {len(folk_stories)} folk tales loaded")

# ── 2. TinyStories (streaming, diverse subset) ────────────────────────────
print("Loading TinyStories (streaming, 20K sample)...")
tinyds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
tiny_stories = []
for i, row in enumerate(tinyds):
    if i >= 20000:
        break
    tiny_stories.append(row["text"])
print(f"  {len(tiny_stories)} TinyStories loaded")

# ── 3. Format for fine-tuning ──────────────────────────────────────────────
def format_story(text: str, title: str = "") -> str:
    parts = []
    if title:
        parts.append(f"Tell me a story about {title}.")
    else:
        parts.append("Tell me a story.")
    parts.append("")
    parts.append(text.strip())
    return "\n\n".join(parts)

examples = []

# TinyStories
for text in tiny_stories:
    examples.append(format_story(text))

# Folk tales  
for s in folk_stories:
    examples.append(format_story(s["text"], s["title"]))

print(f"Total training examples: {len(examples)}")

# Stats
total_chars = sum(len(ex) for ex in examples)
print(f"Total chars: {total_chars:,}")
print(f"Avg chars: {total_chars // len(examples):,}")
print(f"Max chars: {max(len(ex) for ex in examples):,}")
print(f"Min chars: {min(len(ex) for ex in examples):,}")

# ── 4. Save ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps({"text": ex}, ensure_ascii=False) + "\n")

file_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Saved to {OUTPUT} ({file_mb:.1f} MB)")
