import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "models/lora_v2"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def normalize(text):
    return " ".join(text.casefold().split())


def summarize(name, examples, predictions, labels):
    expected = [example["label"] for example in examples]

    result = {
        "accuracy": accuracy_score(expected, predictions),
        "macro_f1": f1_score(
            expected,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "per_class": classification_report(
            expected,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    print(f"\n{name}")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print("Mistakes:")

    mistakes = []

    for example, prediction in zip(examples, predictions):
        if prediction != example["label"]:
            mistakes.append({
                "id": example["id"],
                "expected": example["label"],
                "predicted": prediction,
            })
            print(
                f"  {example['id']}: expected={example['label']}, "
                f"predicted={prediction}"
            )

    if not mistakes:
        print("  None")

    result["mistakes"] = mistakes
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    metadata_path = ADAPTER_DIR / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = metadata["labels"]

    training_path = ROOT / "data/processed/train_v2.jsonl"
    validation_path = ROOT / "data/processed/validation.jsonl"

    training_hash = hashlib.sha256(
        training_path.read_bytes()
    ).hexdigest()

    saved_training_hash = metadata.get("dataset_sha256")
    if saved_training_hash is not None:
        if training_hash != saved_training_hash:
            raise ValueError(
                "Training data changed after this model was trained."
            )
    else:
        print(
            "No training-data hash was saved during training. "
            "The current file hash will be recorded in this report."
        )

    examples = read_jsonl(validation_path)
    training = read_jsonl(training_path)

    if not examples:
        raise ValueError("Validation data is empty.")

    training_texts = {
        normalize(example["text"])
        for example in training
    }
    seen_ids = set()
    seen_texts = set()

    for example in examples:
        example_id = example["id"]
        text = example["text"]
        label = example["label"]

        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("Every validation example needs a string ID.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text: {example_id}")

        if label not in labels:
            raise ValueError(f"Unknown label: {label}")

        normalized_text = normalize(text)

        if example_id in seen_ids:
            raise ValueError(f"Duplicate validation ID: {example_id}")

        if normalized_text in seen_texts:
            raise ValueError(f"Duplicate validation text: {example_id}")

        if normalized_text in training_texts:
            raise ValueError(
                f"Training/validation text overlap: {example_id}"
            )

        seen_ids.add(example_id)
        seen_texts.add(normalized_text)

    counts = Counter(example["label"] for example in examples)
    missing = set(labels) - counts.keys()

    if missing:
        raise ValueError(
            f"Missing validation categories: {sorted(missing)}"
        )

    texts = [example["text"] for example in examples]

    baseline_path = ROOT / "models/baseline_v2.joblib"
    baseline = joblib.load(baseline_path)
    baseline_predictions = baseline.predict(texts).tolist()

    tokenizer = XLMRobertaTokenizer.from_pretrained(ADAPTER_DIR)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        metadata["base_model"],
        revision=metadata.get("base_revision"),
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={
            label: index
            for index, label in enumerate(labels)
        },
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.to("cuda")
    model.eval()

    lora_predictions = []

    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            inputs = {
                key: value.to("cuda")
                for key, value in inputs.items()
            }

            logits = model(**inputs).logits
            predicted_id = logits.argmax(dim=-1).item()
            lora_predictions.append(labels[predicted_id])

    baseline_results = summarize(
        "tfidf_baseline_v2",
        examples,
        baseline_predictions,
        labels,
    )
    lora_results = summarize(
        "lora_v2",
        examples,
        lora_predictions,
        labels,
    )

    results = {
        "split": "development_validation",
        "validation_examples": len(examples),
        "training_examples": len(training),
        "training_sha256": training_hash,
        "training_hash_verified": saved_training_hash is not None,
        "validation_sha256": hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest(),
        "tfidf_baseline_v2": baseline_results,
        "lora_v2": lora_results,
        "predictions": [
            {
                "id": example["id"],
                "expected": example["label"],
                "baseline": baseline_prediction,
                "lora": lora_prediction,
            }
            for example, baseline_prediction, lora_prediction in zip(
                examples,
                baseline_predictions,
                lora_predictions,
            )
        ],
    }

    output_path = ROOT / "reports/validation_v2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved results: {output_path}")
    print(
        "Development validation only; "
        "not final-test or real-world accuracy."
    )


if __name__ == "__main__":
    main()