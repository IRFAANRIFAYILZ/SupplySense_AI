"""
lora_training.py
----------------
Fine-tunes the base instruction model on the domain-specific instruction
dataset (llm/data/instruction_dataset.jsonl) using QLoRA:
  - Base model loaded in 4-bit precision (bitsandbytes, NF4 quantization)
  - LoRA adapters injected into the attention projection layers (PEFT)
  - Only the small LoRA adapter weights are trained; the 4-bit base model
    stays frozen. This is what makes QLoRA feasible on a single consumer
    GPU (or a free-tier Colab T4) instead of requiring full fine-tuning.

Why this matters for the project:
  RAG (rag/) supplies fresh, factual, company-specific knowledge at query
  time by retrieving from real documents. LoRA/QLoRA (this file) instead
  changes the model's *behavior*: it teaches the base model to respond in
  SupplySense AI's expected structured business format (e.g. "Supplier
  Risk: HIGH / Possible actions: 1. ... 2. ...") consistently, which a
  generic base model does not reliably do out of the box. The two
  techniques are complementary, not redundant -- see llm/evaluation.py for
  the side-by-side comparison this is meant to demonstrate.

Base model: Qwen/Qwen2.5-0.5B-Instruct (~0.5B params -- realistically
fine-tunable on a free Colab T4 GPU in well under an hour for this dataset
size; the 0.5B/1.1B class is the right scope for a college project with no
dedicated compute budget. Swap BASE_MODEL for a larger model if you have
access to better hardware).

IMPORTANT — environment note:
  This script requires `torch`, `transformers`, `peft`, `bitsandbytes`, and
  `datasets`, plus a CUDA GPU. It is NOT executed inside this project's
  authoring sandbox (no GPU, no internet access to download model weights
  or packages). Run it on your own machine or in Google Colab -- see the
  README's "How to Run: LoRA/QLoRA" section for exact steps and expected
  runtime.

Run (on a GPU machine, after `pip install -r requirements.txt`):
    python llm/build_instruction_dataset.py     # if not already built
    python llm/lora_training.py
"""

from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "artifacts" / "lora_adapter"

LORA_CONFIG = dict(
    r=16,                 # LoRA rank -- small, appropriate for a ~0.5B model / small dataset
    lora_alpha=32,        # scaling factor, conventionally 2x r
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention projections
    task_type="CAUSAL_LM",
)

TRAINING_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",   # QLoRA-standard memory-efficient optimizer
)


def format_example(example):
    """Turns an {instruction, input, output} record into a single chat-style
    training string using the base model's own chat template conventions."""
    system = ("You are SupplySense AI, a supply-chain analyst assistant. Give concise, "
              "structured, data-grounded responses about supplier risk, delivery "
              "performance, and operational recommendations.")
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\n{example['input']}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def load_datasets():
    from datasets import load_dataset
    train_ds = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"), split="train")
    train_ds = train_ds.map(format_example)
    val_ds = val_ds.map(format_example)
    return train_ds, val_ds


def main():
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    print(f"Loading base model in 4-bit (QLoRA): {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading instruction dataset...")
    train_ds, val_ds = load_datasets()
    print(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    OUTPUT_DIR.parent.mkdir(exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        **TRAINING_CONFIG,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Done. Load with: PeftModel.from_pretrained(base_model, "
          f"'{OUTPUT_DIR}') -- see llm/inference.py")


if __name__ == "__main__":
    main()
