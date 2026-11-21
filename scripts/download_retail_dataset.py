import json
from collections import Counter
from pathlib import Path

from datasets import ClassLabel, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "Hari5115/hinglish-retail-intent-dataset"
DATASET_REVISION = "3b4ab333bfeab463c9407911e958a5e6af6199ac"


def main():
    revision = DATASET_REVISION
    dataset = load_dataset(DATASET_ID, revision=revision)

    summary = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "splits": {},
    }
    all_labels = set()
    text_splits = {}
    text_labels = {}

    for split_name, split in dataset.items():
        required_columns = {"text", "label"}
        if not required_columns.issubset(split.column_names):
            raise ValueError(
                f"{split_name}: missing text or label column."
            )

        label_feature = split.features["label"]
        counts = Counter()
        seen_texts = set()
        duplicate_rows = 0

        for row in split:
            text = row["text"]
            label = row["label"]

            if isinstance(label_feature, ClassLabel):
                label = label_feature.int2str(label)

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split_name}: empty or invalid text.")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{split_name}: invalid label.")

            normalized = " ".join(text.casefold().split())

            if normalized in seen_texts:
                duplicate_rows += 1

            seen_texts.add(normalized)
            text_splits.setdefault(normalized, set()).add(split_name)
            text_labels.setdefault(normalized, set()).add(label)
            counts[label] += 1
            all_labels.add(label)

        summary["splits"][split_name] = {
            "rows": len(split),
            "label_counts": dict(sorted(counts.items())),
            "duplicate_text_rows": duplicate_rows,
        }

        print(f"\n{split_name}: {len(split)} rows")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
        print(f"  Duplicate text rows: {duplicate_rows}")

    summary["labels"] = sorted(all_labels)
    summary["cross_split_duplicate_texts"] = sum(
        len(splits) > 1 for splits in text_splits.values()
    )
    summary["conflicting_label_texts"] = sum(
        len(labels) > 1 for labels in text_labels.values()
    )

    output_dir = ROOT / "data/raw/public_retail"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / revision

    if not snapshot_dir.exists():
        dataset.save_to_disk(str(snapshot_dir))

    report_path = ROOT / "reports/public_dataset_inspection.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal labels: {len(all_labels)}")
    print(
        "Cross-split duplicate texts:",
        summary["cross_split_duplicate_texts"],
    )
    print(
        "Texts with conflicting labels:",
        summary["conflicting_label_texts"],
    )
    print(f"Dataset saved: {snapshot_dir}")
    print(f"Inspection report: {report_path}")
    print("Exact-text checks only; paraphrase leakage is not checked.")


if __name__ == "__main__":
    main()