import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

INSTRUCTION = "Extract a concise list of skills from the job description."


ROLE_SKILLS = {
    "Data Analyst": [
        "SQL",
        "Tableau",
        "Python",
        "Statistics",
        "Data visualization",
        "Reporting",
    ],
    "Backend Engineer": [
        "Java",
        "Spring Boot",
        "REST APIs",
        "PostgreSQL",
        "Docker",
        "AWS",
    ],
    "Frontend Engineer": [
        "JavaScript",
        "TypeScript",
        "React",
        "HTML",
        "CSS",
        "Webpack",
    ],
    "ML Engineer": [
        "Python",
        "PyTorch",
        "Model deployment",
        "MLOps",
        "Feature engineering",
        "AWS",
    ],
    "DevOps Engineer": [
        "Linux",
        "Terraform",
        "Kubernetes",
        "CI/CD",
        "Docker",
        "Monitoring",
    ],
    "Product Manager": [
        "Product strategy",
        "Roadmapping",
        "User research",
        "Analytics",
        "Stakeholder management",
        "Agile",
    ],
    "UX Designer": [
        "Figma",
        "Wireframing",
        "Prototyping",
        "User testing",
        "Design systems",
        "Information architecture",
    ],
    "QA Engineer": [
        "Test automation",
        "Selenium",
        "API testing",
        "Python",
        "Regression testing",
        "Jira",
    ],
    "Data Engineer": [
        "Python",
        "SQL",
        "ETL",
        "Airflow",
        "Spark",
        "Data warehousing",
    ],
    "Mobile Engineer": [
        "Kotlin",
        "Android",
        "Jetpack",
        "REST APIs",
        "CI/CD",
        "UI testing",
    ],
    "Sales Engineer": [
        "Technical demos",
        "CRM",
        "Solution design",
        "Negotiation",
        "Customer onboarding",
        "Presentation",
    ],
    "Security Analyst": [
        "SIEM",
        "Incident response",
        "Threat modeling",
        "Vulnerability management",
        "Network security",
        "Python",
    ],
}


TEMPLATES = [
    (
        "We are hiring a {role} to join our team. You will {responsibility} and {responsibility2}. "
        "Required skills include {skills}. Experience with {tools} is important.",
        "Responsibilities",
    ),
    (
        "As a {role}, you will lead initiatives across the organization. Key duties include {responsibility} "
        "and {responsibility2}. The ideal candidate has hands-on experience with {skills}.",
        "Duties",
    ),
    (
        "Seeking a {role} who can {responsibility}. You will collaborate on {responsibility2}. "
        "Must be proficient in {skills} and comfortable with {tools}.",
        "Summary",
    ),
]


RESPONSIBILITIES = [
    "designing scalable workflows",
    "building dashboards for stakeholders",
    "maintaining production systems",
    "optimizing performance bottlenecks",
    "conducting user research sessions",
    "owning feature delivery from discovery to launch",
    "automating manual reporting",
    "creating reusable components",
    "deploying services to the cloud",
    "writing test plans and executing QA cycles",
]


TOOLS = [
    "Jira",
    "GitHub",
    "Notion",
    "Slack",
    "Confluence",
    "Datadog",
    "Looker",
    "Grafana",
]


def build_job_description(role, skills):
    template, _ = random.choice(TEMPLATES)
    responsibility = random.choice(RESPONSIBILITIES)
    responsibility2 = random.choice([r for r in RESPONSIBILITIES if r != responsibility])
    tools = ", ".join(random.sample(TOOLS, k=2))
    skill_text = ", ".join(skills)
    return template.format(
        role=role,
        responsibility=responsibility,
        responsibility2=responsibility2,
        skills=skill_text,
        tools=tools,
    )


def build_prompt(job_text):
    return f"instruction: {INSTRUCTION}\ninput: {job_text}\noutput:"


def parse_skills(text):
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    cleaned = text.replace("\n", ",")
    items = [item.strip(" -\t") for item in cleaned.split(",") if item.strip()]
    deduped = []
    seen = set()
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Generate teacher-labeled dataset")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--teacher-model", default="google/flan-t5-base")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model)
    model.to("cpu")
    model.eval()

    roles = list(ROLE_SKILLS.items())
    samples = []
    for _ in range(args.size):
        role, skills = random.choice(roles)
        job_text = build_job_description(role, skills)
        samples.append(job_text)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            prompts = [build_prompt(text) for text in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for job_text, raw_output in zip(batch, decoded):
                skills = parse_skills(raw_output)
                record = {
                    "instruction": INSTRUCTION,
                    "input": job_text,
                    "output": json.dumps(skills, ensure_ascii=True),
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
