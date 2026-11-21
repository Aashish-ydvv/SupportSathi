import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizer

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "models/lora_seed_v1"


def main():
    metadata = json.loads(
        (ADAPTER_DIR / "training_metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata["labels"]

    tokenizer = XLMRobertaTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        metadata["base_model"],
        revision=metadata["base_revision"],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.to("cuda")
    model.eval()

    examples = [
        json.loads(line)
        for line in (ROOT / "data/raw/seed.jsonl")
        .read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    predictions = []
    with torch.inference_mode():
        for example in examples:
            inputs = tokenizer(
                example["text"],
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            predicted_id = model(**inputs).logits.argmax(dim=-1).item()
            predictions.append(labels[predicted_id])

    correct = sum(
        prediction == example["label"]
        for prediction, example in zip(predictions, examples)
    )

    print(f"\nTraining examples matched: {correct}/{len(examples)}")
    print("Predicted category counts:", dict(Counter(predictions)))
    print("These are training-set results, not held-out accuracy.")


if __name__ == "__main__":
    main()