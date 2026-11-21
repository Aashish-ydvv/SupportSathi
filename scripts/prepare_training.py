import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def normalize(text):
    return " ".join(text.casefold().split())


def main():
    config = json.loads(
        (ROOT / "configs/labels.json").read_text(encoding="utf-8-sig")
    )
    allowed_labels = set(config["labels"])

    examples = (
        read_jsonl(ROOT / "data/raw/seed.jsonl")
        + read_jsonl(ROOT / "data/raw/additional.jsonl")
    )
    validation = read_jsonl(ROOT / "data/processed/validation.jsonl")
    validation_texts = {
        normalize(example["text"])
        for example in validation
    }

    seen_ids = set()
    seen_texts = set()
    counts = Counter()

    for example in examples:
        example_id = example["id"]
        text = example["text"]
        label = example["label"]

        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("Every example needs a non-empty string ID.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid text: {example_id}")
        if label not in allowed_labels:
            raise ValueError(f"Unknown label: {example_id}: {label}")

        normalized_text = normalize(text)

        if example_id in seen_ids:
            raise ValueError(f"Duplicate ID: {example_id}")
        if normalized_text in seen_texts:
            raise ValueError(f"Duplicate training text: {example_id}")
        if normalized_text in validation_texts:
            raise ValueError(f"Training/validation text overlap: {example_id}")

        seen_ids.add(example_id)
        seen_texts.add(normalized_text)
        counts[label] += 1

    missing_labels = allowed_labels - counts.keys()
    if missing_labels:
        raise ValueError(f"Missing categories: {sorted(missing_labels)}")

    output_path = ROOT / "data/processed/train_v2.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(example, ensure_ascii=False) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )

    print(f"Prepared {len(examples)} training examples.")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    print("No exact text overlap with validation data.")
    print("Paraphrase overlap still requires manual review.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()