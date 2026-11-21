import hashlib
import json
from pathlib import Path

import joblib
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    XLMRobertaTokenizer,
    set_seed,
)

from prepare_training import main as prepare_training

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "microsoft/Multilingual-MiniLM-L12-H384"

EPOCHS = 20
BATCH_SIZE = 2
LEARNING_RATE = 5e-4
MAX_LENGTH = 128
SEED = 42


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    prepare_training()
    set_seed(SEED)

    dataset_path = ROOT / "data/processed/train_v2.jsonl"
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    config = json.loads(
        (ROOT / "configs/labels.json").read_text(encoding="utf-8-sig")
    )
    labels = list(config["labels"])
    label_to_id = {label: index for index, label in enumerate(labels)}

    texts = [example["text"] for example in examples]
    targets = [example["label"] for example in examples]

    baseline = Pipeline([
        ("features", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=SEED,
        )),
    ])
    baseline.fit(texts, targets)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, models_dir / "baseline_v2.joblib")
    print("\nSaved TF-IDF baseline v2.")

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
        attn_implementation="eager",
    )
    base_revision = getattr(base_model.config, "_commit_hash", None)

    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            modules_to_save=["classifier", "pooler"],
        ),
    ).to("cuda")

    model.print_trainable_parameters()

    records = []
    for example in examples:
        record = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        record["labels"] = label_to_id[example["label"]]
        records.append(record)

    loader = DataLoader(
        records,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        num_workers=0,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    print(f"\nTraining LoRA on {torch.cuda.get_device_name(0)}")
    history = []
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        examples_seen = 0

        for batch in loader:
            batch = {
                key: value.to("cuda")
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)

            loss = model(**batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss during epoch {epoch + 1}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )
            optimizer.step()

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            examples_seen += batch_size

        average_loss = total_loss / examples_seen
        history.append({
            "epoch": epoch + 1,
            "training_loss": average_loss,
        })
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | loss: {average_loss:.4f}",
            flush=True,
        )

    output_dir = models_dir / "lora_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": MODEL_ID,
        "base_revision": base_revision,
        "training_file": "data/processed/train_v2.jsonl",
        "training_sha256": hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest(),
        "training_examples": len(examples),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["query", "value"],
        "modules_to_save": ["classifier", "pooler"],
        "labels": labels,
        "data_source": "AI-generated development examples",
        "evaluation": "not_run",
        "history": history,
    }

    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved LoRA v2: {output_dir}")
    print("Validation and final-test performance are not yet measured.")


if __name__ == "__main__":
    main()