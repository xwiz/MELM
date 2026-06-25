"""
LoRA fine-tuning script for Qwen2.5-0.5B on story generation.
Trains on TinyStories + folk tales, saves LoRA adapters.

Usage:
    python scripts/fine_tune_qwen.py                    # full run
    python scripts/fine_tune_qwen.py --resume adapter   # resume from adapter
    python scripts/fine_tune_qwen.py --dry-run          # 10 steps only
"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "train_dataset.jsonl")
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "..", "data", "qwen-lora-adapter")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None, help="Path to adapter to resume from")
parser.add_argument("--dry-run", action="store_true", help="Train 10 steps only")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
args = parser.parse_args()


def load_and_tokenize():
    """Load dataset and tokenize."""
    print(f"Loading dataset from {DATA_PATH}")
    texts = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    print(f"Loaded {len(texts)} examples")
    # Truncate long texts early to save memory
    max_chars = 1500
    short_texts = [t[:max_chars] for t in texts]
    dataset = Dataset.from_dict({"text": short_texts})

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"],
                            desc="Tokenizing", num_proc=0)
    print(f"Tokenized: {len(tokenized)} examples")
    return tokenized, tokenizer


def create_model():
    """Create model with LoRA."""
    print(f"Loading model: {MODEL_NAME}")

    # Use 4-bit if available, else FP16
    use_4bit = False
    try:
        import bitsandbytes
        use_4bit = True
    except ImportError:
        pass

    if use_4bit and args.resume is None:
        print("Using 4-bit QLoRA quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("Using FP16 (no quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    if args.resume:
        print(f"Loading existing adapter from {args.resume}")
        model = PeftModel.from_pretrained(model, args.resume)
        return model, model

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, lora_config


def train():
    """Main training loop."""
    tokenized, tokenizer = load_and_tokenize()
    model, lora_config = create_model()

    # Limit dataset for dry run
    if args.dry_run:
        tokenized = tokenized.select(range(min(10, len(tokenized))))
        args.epochs = 1

    steps = max(10, len(tokenized) // args.batch_size // 2)

    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs if not args.dry_run else 0.1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        warmup_steps=min(100, steps // 10),
        logging_steps=10,
        save_strategy="epoch" if not args.dry_run else "steps",
        save_steps=10 if args.dry_run else None,
        save_total_limit=1,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
    )

    torch.cuda.empty_cache()
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # Save
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"LoRA adapters saved to {ADAPTER_DIR}")

    # Print loss
    log = trainer.state.log_history
    losses = [x.get("loss", None) for x in log if "loss" in x]
    if losses:
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Min loss: {min(losses):.4f}")


if __name__ == "__main__":
    train()
