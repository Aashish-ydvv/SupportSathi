import hashlib
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/retail_public_v1"
OUTPUT_DIR = ROOT / "models/retail_public_v1"


def main():
    config = json.loads(
        (ROOT / "configs/labels_public_v1.json").read_text(
            encoding="utf-8-sig"
        )
    )
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )

    labels = list(config["labels"])
    if set(labels) != set(manifest["labels"]):
        raise ValueError("Configuration labels do not match the dataset.")

    training_path = DATA_DIR / "train.jsonl"
    training_hash = hashlib.sha256(
        training_path.read_bytes()
    ).hexdigest()

    if training_hash != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Training file differs from the cleaned manifest.")

    examples = [
        json.loads(line)
        for line in training_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not examples:
        raise ValueError("Training data is empty.")

    for example in examples:
        if example["label"] not in labels:
            raise ValueError(f"Unknown label: {example['label']}")

    model = Pipeline([
        ("features", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(
        [example["text"] for example in examples],
        [example["label"] for example in examples],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "baseline.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_version": "retail-tfidf-public-v1",
        "method": "character TF-IDF and logistic regression",
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["revision"],
        "training_examples": len(examples),
        "training_sha256": training_hash,
        "labels": labels,
        "seed": 42,
        "evaluation": "not_run",
    }

    (OUTPUT_DIR / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Trained on {len(examples)} examples.")
    print(f"Categories: {len(model.classes_)}")
    print(f"Saved model: {model_path}")
    print("Validation and test data were not used for fitting.")


if __name__ == "__main__":
    main()