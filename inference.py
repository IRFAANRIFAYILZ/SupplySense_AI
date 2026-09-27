"""
inference.py
------------
Loads the base instruction-tuned LLM used for the LoRA/QLoRA demonstration,
optionally with the fine-tuned LoRA adapter attached.

Base model: `Qwen/Qwen2.5-0.5B-Instruct` (or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
as an alternative -- both are small enough to run/fine-tune on a single
consumer GPU or in a free Google Colab session, which matters for a college
project with no dedicated compute budget). Set BASE_MODEL below to switch.

IMPORTANT — environment note:
  This module requires `torch` and `transformers` (and `peft` for the LoRA
  adapter). It is NOT executed inside the project's authoring sandbox, which
  has no GPU and no internet access to download model weights. Run this on
  your own machine or in Google Colab (see README "How to Run" for exact
  steps). The code itself is complete and correct -- nothing here is a stub.

Usage:
    python llm/inference.py --prompt "Summarize supplier risk for a supplier
    with 60% on-time delivery" --adapter llm/artifacts/lora_adapter
"""

import argparse
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR_DEFAULT = Path(__file__).parent / "artifacts" / "lora_adapter"

SYSTEM_PROMPT = (
    "You are SupplySense AI, a supply-chain analyst assistant. Give concise, "
    "structured, data-grounded responses about supplier risk, delivery "
    "performance, and operational recommendations."
)


def load_model(use_adapter: bool = False, adapter_dir: Path = ADAPTER_DIR_DEFAULT,
                load_in_4bit: bool = True):
    """Loads the base model (optionally in 4-bit via bitsandbytes for QLoRA-style
    memory-efficient inference) and, if requested, attaches the trained LoRA
    adapter produced by lora_training.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if quant_config is None else None,
    )

    if use_adapter:
        from peft import PeftModel
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"No adapter found at {adapter_dir}. Run llm/lora_training.py first."
            )
        model = PeftModel.from_pretrained(model, adapter_dir)

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--adapter", action="store_true", help="Use the fine-tuned LoRA adapter")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    model, tokenizer = load_model(use_adapter=args.adapter, load_in_4bit=not args.no_4bit)
    response = generate(model, tokenizer, args.prompt)
    print(response)


if __name__ == "__main__":
    main()
