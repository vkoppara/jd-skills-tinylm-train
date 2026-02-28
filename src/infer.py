import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def build_prompt(text):
    return (
        "instruction: Extract a concise list of skills from the job description.\n"
        f"input: {text}\n"
        "output:"
    )


def resolve_base_model(model_path, base_model_override=None):
    if base_model_override:
        return base_model_override
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        return config.get("base_model_name_or_path", model_path)
    return model_path


def load_model(model_path, device, base_model_override=None, local_only=False):
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        base_name = resolve_base_model(model_path, base_model_override)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_name, local_files_only=local_only
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, local_files_only=local_only
        )
    model.to(device)
    model.eval()
    return model


def parse_output(text):
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    cleaned = text.replace("\n", ",")
    items = [item.strip(" -\t") for item in cleaned.split(",") if item.strip()]
    seen = set()
    deduped = []
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Run skill extraction")
    parser.add_argument("--model", required=True, help="Model or adapter path")
    parser.add_argument("--text", required=True, help="Job description text")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Path to local base model (e.g., base_model/flan-t5-small)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Disable Hugging Face Hub downloads",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    base_name = resolve_base_model(args.model, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_name, use_fast=True, local_files_only=args.local_only
    )
    model = load_model(
        args.model,
        device,
        base_model_override=args.base_model,
        local_only=args.local_only,
    )

    prompt = build_prompt(args.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    skills = parse_output(decoded)

    print(json.dumps({"skills": skills, "raw": decoded}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
