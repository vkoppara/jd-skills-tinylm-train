import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fetch base model via Git LFS")
    parser.add_argument(
        "--model-dir",
        default="base_model/flan-t5-small",
        help="Path where the base model should exist",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    expected_file = model_dir / "config.json"
    if expected_file.exists():
        print(f"Base model already present at: {model_dir}")
        return

    result = subprocess.run(["git", "lfs", "pull"], check=False)
    if result.returncode != 0:
        raise SystemExit("git lfs pull failed. Install git-lfs and try again.")

    if not expected_file.exists():
        raise SystemExit(
            f"Base model not found at {model_dir}. Ensure LFS files were pulled."
        )

    print(f"Base model ready at: {model_dir}")


if __name__ == "__main__":
    main()
