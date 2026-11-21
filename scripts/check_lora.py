import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "models/lora_seed_debug"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    metadata = json.loads(
        (ADAPTER_DIR / "training_metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata["labels"]

    tokenizer = XLMRobertaTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        metadata["base_model"],
        revision=metadata.get("base_revision"),
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
    matched = 0

    with torch.inference_mode():
        for example in examples:
            inputs = tokenizer(
                example["text"],
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
            prediction = labels[predicted_id]

            predictions.append(prediction)
            matched += prediction == example["label"]

            if prediction != example["label"]:
                print(
                    f"Mismatch: {example['id']} | "
                    f"expected={example['label']} | predicted={prediction}"
                )

    print(f"Training examples matched: {matched}/{len(examples)}")
    print("Predicted category counts:", dict(Counter(predictions)))
    print("Training-set check only; held-out evaluation has not been run.")


if __name__ == "__main__":
    main()