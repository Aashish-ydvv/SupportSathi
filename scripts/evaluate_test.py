import json
from pathlib import Path

import joblib
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
)

from evaluate_validation import file_hash, read_json, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/retail_public_v1"
MODEL_DIR = ROOT / "models/retail_public_v1"
ADAPTER_DIR = MODEL_DIR / "lora"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    manifest = read_json(DATA_DIR / "manifest.json")
    baseline_metadata = read_json(MODEL_DIR / "baseline_metadata.json")
    lora_metadata = read_json(ADAPTER_DIR / "training_metadata.json")
    labels = lora_metadata["labels"]

    if set(labels) != set(manifest["labels"]):
        raise ValueError("LoRA labels differ from the dataset.")
    if set(labels) != set(baseline_metadata["labels"]):
        raise ValueError("Model labels differ.")

    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        if file_hash(path) != manifest["splits"][split_name]["sha256"]:
            raise ValueError(f"{split_name} data differs from its manifest.")

    training_hash = file_hash(DATA_DIR / "train.jsonl")
    for metadata in [baseline_metadata, lora_metadata]:
        if metadata["training_sha256"] != training_hash:
            raise ValueError("A model used a different training file.")

    validation_hash = file_hash(DATA_DIR / "validation.jsonl")
    if validation_hash != lora_metadata["validation_sha256"]:
        raise ValueError("Checkpoint selection used different validation data.")

    examples = read_jsonl(DATA_DIR / "test.jsonl")
    if not examples:
        raise ValueError("Test data is empty.")

    texts = [example["text"] for example in examples]

    baseline_path = MODEL_DIR / "baseline.joblib"
    baseline = joblib.load(baseline_path)
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

    with torch.inference_mode():
        for start in range(0, len(texts), 4):
            inputs = tokenizer(
                texts[start:start + 4],
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
        "split": "final_test",
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["revision"],
        "training_examples": baseline_metadata["training_examples"],
        "evaluation_examples": len(examples),
        "training_sha256": training_hash,
        "evaluation_sha256": file_hash(DATA_DIR / "test.jsonl"),
        "lora_selected_epoch": lora_metadata["best_epoch"],
        "artifact_hashes": {
            "baseline.joblib": file_hash(baseline_path),
            "adapter_config.json": file_hash(
                ADAPTER_DIR / "adapter_config.json"
            ),
            "adapter_model.safetensors": file_hash(
                ADAPTER_DIR / "adapter_model.safetensors"
            ),
            "training_metadata.json": file_hash(
                ADAPTER_DIR / "training_metadata.json"
            ),
        },
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

    output_path = ROOT / "reports/public_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved final-test report: {output_path}")
    print("Report this result without tuning against the test examples.")


if __name__ == "__main__":
    main()