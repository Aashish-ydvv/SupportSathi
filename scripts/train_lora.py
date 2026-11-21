import hashlib
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    XLMRobertaTokenizer,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/retail_public_v1"
OUTPUT_DIR = ROOT / "models/retail_public_v1/lora"
MODEL_ID = "microsoft/Multilingual-MiniLM-L12-H384"

MAX_EPOCHS = 8
PATIENCE = 2
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
MAX_LENGTH = 128
SEED = 42


def read_split(name, manifest, allowed_labels):
    path = DATA_DIR / f"{name}.jsonl"
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    if file_hash != manifest["splits"][name]["sha256"]:
        raise ValueError(f"{name} data differs from its manifest.")

    examples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not examples:
        raise ValueError(f"{name} data is empty.")

    for example in examples:
        if example["label"] not in allowed_labels:
            raise ValueError(f"Unknown label in {name}: {example['label']}")
        if not example["text"].strip():
            raise ValueError(f"Empty message in {name}.")

    return examples, file_hash


def make_loader(examples, tokenizer, label_to_id, shuffle):
    records = []

    for example in examples:
        record = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        record["labels"] = label_to_id[example["label"]]
        records.append(record)

    return DataLoader(
        records,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),
        num_workers=0,
    )


def evaluate(model, loader, label_count):
    model.eval()
    expected = []
    predictions = []

    with torch.inference_mode():
        for batch in loader:
            targets = batch.pop("labels")
            inputs = {
                key: value.to("cuda")
                for key, value in batch.items()
            }

            logits = model(**inputs).logits
            predictions.extend(
                logits.argmax(dim=-1).cpu().tolist()
            )
            expected.extend(targets.tolist())

    return {
        "accuracy": float(accuracy_score(expected, predictions)),
        "macro_f1": float(f1_score(
            expected,
            predictions,
            labels=list(range(label_count)),
            average="macro",
            zero_division=0,
        )),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    set_seed(SEED)

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

    label_to_id = {
        label: index for index, label in enumerate(labels)
    }

    training, training_hash = read_split("train", manifest, labels)
    validation, validation_hash = read_split(
        "validation", manifest, labels
    )

    training_texts = {
        " ".join(example["text"].casefold().split())
        for example in training
    }
    validation_texts = {
        " ".join(example["text"].casefold().split())
        for example in validation
    }
    if training_texts & validation_texts:
        raise ValueError("Training and validation contain matching texts.")

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

    train_loader = make_loader(
        training, tokenizer, label_to_id, shuffle=True
    )
    validation_loader = make_loader(
        validation, tokenizer, label_to_id, shuffle=False
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = None
    best_metrics = None
    stale_epochs = 0
    history = []

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Training examples: {len(training)}")
    print(f"Validation examples: {len(validation)}")
    print("Final test set is not used.\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        examples_seen = 0

        for batch in train_loader:
            batch = {
                key: value.to("cuda")
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)

            loss = model(**batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss in epoch {epoch}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=1.0
            )
            optimizer.step()

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            examples_seen += batch_size

        metrics = evaluate(
            model, validation_loader, len(labels)
        )
        average_loss = total_loss / examples_seen

        history.append({
            "epoch": epoch,
            "training_loss": average_loss,
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
        })

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"loss: {average_loss:.4f} | "
            f"val accuracy: {metrics['accuracy']:.2%} | "
            f"val macro-F1: {metrics['macro_f1']:.4f}",
            flush=True,
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = metrics
            stale_epochs = 0

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print("  Saved new best checkpoint.", flush=True)
        else:
            stale_epochs += 1

        metadata = {
            "model_version": "retail-minilm-lora-public-v1",
            "base_model": MODEL_ID,
            "base_revision": base_revision,
            "dataset_id": manifest["dataset_id"],
            "dataset_revision": manifest["revision"],
            "training_examples": len(training),
            "validation_examples": len(validation),
            "training_sha256": training_hash,
            "validation_sha256": validation_hash,
            "labels": labels,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "target_modules": ["query", "value"],
            "modules_to_save": ["classifier", "pooler"],
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_validation": best_metrics,
            "selection_metric": "validation_macro_f1",
            "final_test_evaluated": False,
            "history": history,
        }

        (OUTPUT_DIR / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        if stale_epochs >= PATIENCE:
            print("Early stopping: validation macro-F1 stopped improving.")
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Saved checkpoint: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()