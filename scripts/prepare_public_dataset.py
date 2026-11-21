import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import ClassLabel, load_from_disk

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.casefold().split())


def main():
    inspection_path = ROOT / "reports/public_dataset_inspection.json"
    inspection = json.loads(
        inspection_path.read_text(encoding="utf-8")
    )

    revision = inspection["revision"]
    source_dir = ROOT / "data/raw/public_retail" / revision
    dataset = load_from_disk(str(source_dir))

    expected_splits = {"train", "validation", "test"}
    if set(dataset.keys()) != expected_splits:
        raise ValueError("Expected train, validation, and test splits.")

    expected_labels = set(inspection["labels"])
    if len(expected_labels) != 13:
        raise ValueError("Expected exactly 13 labels.")

    cleaned = {}
    summary = {}
    seen = {}

    # Keep evaluation examples; remove their matches from earlier stages.
    for split_name in ["test", "validation", "train"]:
        split = dataset[split_name]
        label_feature = split.features["label"]
        records = []
        within_split_duplicates = 0
        cross_split_duplicates = 0

        for index, row in enumerate(split):
            text = row["text"]
            label = row["label"]

            if isinstance(label_feature, ClassLabel):
                label = label_feature.int2str(label)

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text: {split_name}, row {index}")
            if label not in expected_labels:
                raise ValueError(f"Unknown label: {label}")

            normalized = normalize(text)

            if normalized in seen:
                previous_label, previous_split = seen[normalized]

                if previous_label != label:
                    raise ValueError(
                        f"Conflicting labels: {split_name}, row {index}"
                    )

                if previous_split == split_name:
                    within_split_duplicates += 1
                else:
                    cross_split_duplicates += 1
                continue

            seen[normalized] = (label, split_name)
            group_id = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            records.append({
                "id": f"retail_{split_name}_{index:05d}",
                "text": text.strip(),
                "label": label,
                "group_id": group_id,
                "source": inspection["dataset_id"],
                "source_revision": revision,
                "source_split": split_name,
                "source_row": index,
            })

        counts = Counter(record["label"] for record in records)
        missing = expected_labels - counts.keys()

        if missing:
            raise ValueError(
                f"{split_name} lost all examples for: {sorted(missing)}"
            )

        cleaned[split_name] = records
        summary[split_name] = {
            "original_rows": len(split),
            "cleaned_rows": len(records),
            "within_split_duplicates_removed": within_split_duplicates,
            "cross_split_duplicate_rows_removed": cross_split_duplicates,
            "label_counts": dict(sorted(counts.items())),
        }

    output_dir = ROOT / "data/processed/retail_public_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        records = cleaned[split_name]
        output_path = output_dir / f"{split_name}.jsonl"
        output_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

        summary[split_name]["sha256"] = hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest()

        print(f"\n{split_name}")
        print(f"  Original rows: {summary[split_name]['original_rows']}")
        print(f"  Cleaned rows: {len(records)}")
        print(
            "  Within-split duplicates removed:",
            summary[split_name]["within_split_duplicates_removed"],
        )
        print(
            "  Cross-split duplicate rows removed:",
            summary[split_name]["cross_split_duplicate_rows_removed"],
        )

    manifest = {
        "dataset_id": inspection["dataset_id"],
        "source_url": inspection["source_url"],
        "revision": revision,
        "declared_license": "MIT",
        "data_type": "synthetic",
        "labels": sorted(expected_labels),
        "normalization": "Unicode NFC, casefold, collapsed whitespace",
        "duplicate_priority": ["test", "validation", "train"],
        "paraphrase_leakage_checked": False,
        "splits": summary,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved cleaned dataset: {output_dir}")
    print("Exact normalized texts are disjoint across splits.")
    print("Paraphrase leakage still requires review.")


if __name__ == "__main__":
    main()