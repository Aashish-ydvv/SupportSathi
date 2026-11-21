import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    XLMRobertaTokenizer,
    set_seed,
)

from validate_data import main as validate_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "microsoft/Multilingual-MiniLM-L12-H384"

EPOCHS = 40
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
MAX_LENGTH = 128
SEED = 42


def main():
    validate_data()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the project's virtual environment "
            "and check your PyTorch installation."
        )

    config = json.loads(
        (ROOT / "configs/labels.json").read_text(encoding="utf-8-sig")
    )
    labels = list(config["labels"])
    label_to_id = {label: index for index, label in enumerate(labels)}

    dataset_path = ROOT / "data/raw/seed.jsonl"
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_ID)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
        attn_implementation="eager",
    )
    base_revision = getattr(base_model.config, "_commit_hash", None)

    adapter_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier", "pooler"],
    )

    model = get_peft_model(base_model, adapter_config).to("cuda")
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

    print(f"\nTraining on: {torch.cuda.get_device_name(0)}")
    print(f"Examples: {len(examples)}")
    print("Diagnostic run: training-set learning only.\n")

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
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss: {average_loss:.4f}"
        )

    output_dir = ROOT / "models/lora_seed_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": MODEL_ID,
        "base_revision": base_revision,
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
        "status": "training_fit_diagnostic_only",
        "evaluation": "not_run",
        "labels": labels,
        "history": history,
    }

    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved adapter and classifier: {output_dir}")
    print("Held-out evaluation has not been run.")


if __name__ == "__main__":
    main()