"""Evaluate fine-tuned model vs base model. Compare sample story outputs."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qwen-lora-adapter")

prompts = [
    "Tell me a story about a brave child who discovers a magical forest.",
    "Tell me a story about a lost kitten finding its way home.",
    "In a small village in Ghana, there lived a clever girl named Ama.",
]

print("Loading base model (4-bit for CPU-like eval)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt
    story = text[len(prompt):].strip()
    print(f"\n{'='*60}")
    print(f"PROMPT {i+1}: {prompt[:60]}...")
    print(f"{'='*60}")
    print(story[:500])
    print()

print("Done. LoRA adapter at:", ADAPTER_DIR)
print("Adapter files:")
for f in os.listdir(ADAPTER_DIR):
    size = os.path.getsize(os.path.join(ADAPTER_DIR, f))
    print(f"  {f}: {size:,} bytes")
