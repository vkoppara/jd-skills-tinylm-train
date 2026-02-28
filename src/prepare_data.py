import argparse
import csv
import json
from pathlib import Path

INSTRUCTION = "Extract a concise list of skills from the job description."


def normalize_skills(raw_skills):
    if raw_skills is None:
        return []
    if isinstance(raw_skills, list):
        skills = raw_skills
    else:
        text = str(raw_skills).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                skills = json.loads(text)
            except json.JSONDecodeError:
                skills = [s.strip() for s in text.strip("[]").split(",")]
        else:
            skills = [s.strip() for s in text.split(",")]
    cleaned = []
    seen = set()
    for skill in skills:
        if not skill:
            continue
        normalized = " ".join(skill.split())
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        cleaned.append(normalized)
    return cleaned


def emit_prompts(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            prompt = (
                f"instruction: {INSTRUCTION}\n"
                f"input: {row['job_description']}\n"
                "output: JSON array of skill strings"
            )
            record = {"id": idx, "prompt": prompt}
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL training data")
    parser.add_argument("--input", required=True, help="Path to CSV or JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--emit-prompts", action="store_true", help="Write teacher prompts JSONL")
    parser.add_argument("--prompts-output", default="data/prompts.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = []

    if input_path.suffix.lower() == ".csv":
        with input_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
    else:
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))

    prepared = []
    for row in rows:
        job_description = row.get("job_description") or row.get("input") or ""
        raw_skills = row.get("skills") or row.get("output") or []
        skills = normalize_skills(raw_skills)
        prepared.append(
            {
                "instruction": INSTRUCTION,
                "input": job_description.strip(),
                "output": json.dumps(skills, ensure_ascii=True),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in prepared:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    if args.emit_prompts:
        emit_prompts(prepared, Path(args.prompts_output))


if __name__ == "__main__":
    main()
