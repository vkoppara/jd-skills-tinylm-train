import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Download base model artifacts")
    parser.add_argument("--model", default="google/flan-t5-small")
    args = parser.parse_args()

    AutoTokenizer.from_pretrained(args.model, use_fast=True)
    AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print(f"Downloaded base model: {args.model}")


if __name__ == "__main__":
    main()
