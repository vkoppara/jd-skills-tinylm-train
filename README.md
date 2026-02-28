# MicroLLM Skills Extractor

Goal: build a tiny instruction-tuned model that takes a job description and returns a clean list of skill sets. The pipeline below uses a larger LLM as a teacher to generate training data, then distills into a small model that can run on low-capacity machines.

## Quick plan

1. Data: collect job descriptions and generate skill lists with a teacher LLM.
2. Train: fine-tune a small seq2seq model (default: `google/flan-t5-small`) with LoRA.
3. Evaluate: measure JSON validity and skill extraction quality.
4. Deploy: run CPU inference and optionally export to ONNX.

## Why this approach

- Task is narrow (skills extraction), so a small instruction model works well.
- Distillation keeps inference cheap and fast.
- LoRA keeps training cost low and allows CPU-only training on small datasets.

## Directory layout

- `data/sample.jsonl` - tiny demo dataset
- `configs/train.yaml` - training config
- `src/prepare_data.py` - convert raw data to training JSONL
- `src/train.py` - LoRA fine-tuning
- `src/infer.py` - inference demo

## Prompt format

```
instruction: Extract a concise list of skills from the job description.
input: <job description text>
output: JSON array of skill strings
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download the base model (Git LFS)

This repo stores the base model (`flan-t5-small`) via Git LFS under `base_model/flan-t5-small`.

```
git lfs install
git lfs pull
```

You can also run:

```
python scripts/download_base_model.py
```

## Prepare data

Start from your own CSV with columns `job_description` and `skills` (comma-separated or JSON list).

```
python src/prepare_data.py --input data/raw_jobs.csv --output data/train.jsonl
```

If you need teacher-generated skills, use the `--emit-prompts` option to produce prompts you can feed to a larger LLM, then merge the results.

## Train

```
python src/train.py --config configs/train.yaml
```

The fine-tuned model is saved to `outputs/skills-lora` by default.

## Inference

```
python src/infer.py --model outputs/skills-lora --text "<job description>"
```

## Deployment notes

- CPU-only inference works fine for `flan-t5-small` (80M params).
- If you want faster latency, export to ONNX and use `optimum`.
- If you later switch to a small Llama model, you can quantize to GGUF and run with `llama.cpp`.

## Next steps

- Collect 1k-10k labeled examples via teacher LLM.
- Add evaluation with JSON validity and skill overlap metrics.
- Add post-processing to normalize skills (lowercase, canonical mapping).
