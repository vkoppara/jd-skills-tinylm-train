import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def build_prompt(instruction, job_text):
    return f"instruction: {instruction}\ninput: {job_text}\noutput:"


def tokenize_batch(batch, tokenizer, max_source_length, max_target_length):
    inputs = [build_prompt(inst, text) for inst, text in zip(batch["instruction"], batch["input"])]
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=batch["output"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Train a tiny skill extractor")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset = load_dataset("json", data_files=config["train_file"], split="train")
    if config.get("val_split_ratio", 0) > 0:
        split = dataset.train_test_split(test_size=config["val_split_ratio"], seed=config["seed"])
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            config["max_source_length"],
            config["max_target_length"],
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            lambda batch: tokenize_batch(
                batch,
                tokenizer,
                config["max_source_length"],
                config["max_target_length"],
            ),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=config.get("eval_steps", config["save_steps"]),
        save_total_limit=2,
        predict_with_generate=True,
        report_to=[],
        seed=config["seed"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()
