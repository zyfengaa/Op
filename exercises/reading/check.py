"""Deterministic quiz grading; no third-party dependencies."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=Path(__file__).with_name("student.json"))
    args = parser.parse_args()
    expected = json.loads(Path(__file__).with_name("solutions.json").read_text(encoding="utf-8"))
    try:
        submitted = json.loads(args.answers.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    score = 0
    for task, answers in expected.items():
        values = submitted.get(task, []) if isinstance(submitted, dict) else []
        if not isinstance(values, list):
            values = []
        for index, correct in enumerate(answers):
            actual = str(values[index]).strip().upper() if index < len(values) else ""
            passed = actual == correct
            print(f"{task}.{index+1}: {'PASS' if passed else 'TRY AGAIN'}")
            score += passed
    print(f"score: {score}/15")
    return 0 if score == 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
