import hashlib
import json
from pathlib import Path

import joblib
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/retail_public_v1"
MODEL_DIR = ROOT / "models/retail_public_v1"
ADAPTER_DIR = MODEL_DIR / "lora"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(name, examples, predictions, labels):
    expected = [example["label"] for example in examples]

    metrics = {
        "accuracy": float(accuracy_score(expected, predictions)),
        "macro_f1": float(f1_score(
            expected,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )),
        "per_class": classification_report(
            expected,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            expected, predictions, labels=labels
        ).tolist(),
        "confusion_matrix_labels": labels,
    }

    mistakes = [
        {
            "id": example["id"],
            "expected": example["label"],
            "predicted": prediction,
        }
        for example, prediction in zip(examples, predictions)
        if prediction != example["label"]
    ]

    print(f"\n{name}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Mistakes: {len(mistakes)}/{len(examples)}")

    for mistake in mistakes:
        print(
            f"  {mistake['id']}: expected={mistake['expected']}, "
            f"predicted={mistake['predicted']}"
        )

    metrics["mistakes"] = mistakes
    return metrics


def main():
    manifest = read_json(DATA_DIR / "manifest.json")
    baseline_metadata = read_json(MODEL_DIR / "baseline_metadata.json")
    lora_metadata = read_json(ADAPTER_DIR / "training_metadata.json")
    labels = lora_metadata["labels"]

    if set(labels) != set(manifest["labels"]):
        raise ValueError("LoRA labels do not match the dataset.")
    if set(labels) != set(baseline_metadata["labels"]):
        raise ValueError("Baseline and LoRA labels differ.")

    training_hash = file_hash(DATA_DIR / "train.jsonl")
    validation_path = DATA_DIR / "validation.jsonl"
    validation_hash = file_hash(validation_path)

    for metadata in [baseline_metadata, lora_metadata]:
        if metadata["training_sha256"] != training_hash:
            raise ValueError("A model used a different training file.")

    if training_hash != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Training data differs from the manifest.")
    if validation_hash != manifest["splits"]["validation"]["sha256"]:
        raise ValueError("Validation data differs from the manifest.")
    if validation_hash != lora_metadata["validation_sha256"]:
        raise ValueError("Validation data changed since LoRA training.")

    examples = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError("Validation data is empty.")

    texts = [example["text"] for example in examples]

    baseline = joblib.load(MODEL_DIR / "baseline.joblib")
    baseline_predictions = baseline.predict(texts).tolist()

    tokenizer = XLMRobertaTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        lora_metadata["base_model"],
        revision=lora_metadata.get("base_revision"),
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={
            label: index for index, label in enumerate(labels)
        },
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    lora_predictions = []
    batch_size = 4

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=lora_metadata["max_length"],
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }
            predicted_ids = (
                model(**inputs).logits.argmax(dim=-1).cpu().tolist()
            )
            lora_predictions.extend(
                labels[index] for index in predicted_ids
            )

    results = {
        "split": "validation",
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["revision"],
        "training_examples": baseline_metadata["training_examples"],
        "evaluation_examples": len(examples),
        "training_sha256": training_hash,
        "evaluation_sha256": validation_hash,
        "lora_selected_epoch": lora_metadata["best_epoch"],
        "models": {
            "baseline": summarize(
                baseline_metadata["model_version"],
                examples,
                baseline_predictions,
                labels,
            ),
            "lora": summarize(
                lora_metadata["model_version"],
                examples,
                lora_predictions,
                labels,
            ),
        },
        "predictions": [
            {
                "id": example["id"],
                "expected": example["label"],
                "baseline": baseline_prediction,
                "lora": lora_prediction,
            }
            for example, baseline_prediction, lora_prediction in zip(
                examples, baseline_predictions, lora_predictions
            )
        ],
    }

    output_path = ROOT / "reports/public_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved report: {output_path}")
    print("Validation results only. Final test set remains unevaluated.")


if __name__ == "__main__":
    main()