import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads(
        (ROOT / "configs/labels.json").read_text(encoding="utf-8-sig")
    )
    allowed_labels = set(config["labels"])
    counts = Counter()
    seen_ids = set()
    seen_texts = set()

    path = ROOT / "data/raw/seed.jsonl"
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue

        example = json.loads(line)
        example_id = example["id"]
        text = example["text"]
        label = example["label"]

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Line {line_number}: empty or invalid text")
        if label not in allowed_labels:
            raise ValueError(f"Line {line_number}: unknown label {label}")

        normalized_text = " ".join(text.casefold().split())
        if example_id in seen_ids:
            raise ValueError(f"Line {line_number}: duplicate ID")
        if normalized_text in seen_texts:
            raise ValueError(f"Line {line_number}: duplicate text")

        seen_ids.add(example_id)
        seen_texts.add(normalized_text)
        counts[label] += 1

    missing = allowed_labels - counts.keys()
    if missing:
        raise ValueError(f"Missing categories: {', '.join(sorted(missing))}")

    print(f"Validated {sum(counts.values())} examples.")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()