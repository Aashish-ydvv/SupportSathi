import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from validate_data import main as validate_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    validate_data()

    dataset_path = ROOT / "data/raw/seed.jsonl"
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    model = Pipeline([
        ("features", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    model.fit(
        [example["text"] for example in examples],
        [example["label"] for example in examples],
    )

    output_path = ROOT / "models/baseline.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print(f"\nSaved model: {output_path}")
    print("Starter baseline only; held-out evaluation has not been run.")


if __name__ == "__main__":
    main()