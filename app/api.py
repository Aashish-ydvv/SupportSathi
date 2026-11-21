import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

import joblib
import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field, field_validator
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/retail_public_v1"
ADAPTER_DIR = MODEL_DIR / "lora"
inference_lock = Lock()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    baseline_metadata = read_json(MODEL_DIR / "baseline_metadata.json")
    lora_metadata = read_json(ADAPTER_DIR / "training_metadata.json")
    config = read_json(ROOT / "configs/labels_public_v1.json")

    labels = lora_metadata["labels"]

    if set(labels) != set(config["labels"]):
        raise RuntimeError("Model labels differ from the configuration.")

    if set(labels) != set(baseline_metadata["labels"]):
        raise RuntimeError("Baseline and LoRA categories differ.")

    if (
        baseline_metadata["training_sha256"]
        != lora_metadata["training_sha256"]
    ):
        raise RuntimeError("Models were trained on different data.")

    baseline = joblib.load(MODEL_DIR / "baseline.joblib")
    if set(baseline.classes_) != set(labels):
        raise RuntimeError("Baseline artifact has unexpected categories.")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    lora.to(device)
    lora.eval()

    app.state.baseline = baseline
    app.state.tokenizer = tokenizer
    app.state.lora = lora
    app.state.device = device
    app.state.labels = labels
    app.state.baseline_metadata = baseline_metadata
    app.state.lora_metadata = lora_metadata
    app.state.max_tokens = lora_metadata["max_length"]

    yield


app = FastAPI(
    title="SupportSathi API",
    version="0.3.0",
    description=(
        "Thirteen-category retail support classifier for "
        "Roman-script Hinglish. Trained on public synthetic data."
    ),
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    model: Literal["baseline", "lora"] = "baseline"

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a non-empty support message.")
        return value


class PredictionResponse(BaseModel):
    intent: str
    model_version: str
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": ["baseline", "lora"],
        "label_count": len(app.state.labels),
        "lora_device": app.state.device,
    }


@app.get("/v1/model-info")
def model_info():
    return {
        "default_model": "baseline",
        "available_models": {
            "baseline": app.state.baseline_metadata["model_version"],
            "lora": app.state.lora_metadata["model_version"],
        },
        "labels": app.state.labels,
        "training_examples": app.state.lora_metadata["training_examples"],
        "dataset_id": app.state.lora_metadata["dataset_id"],
        "dataset_revision": app.state.lora_metadata["dataset_revision"],
        "data_source": "public synthetic retail support dataset",
        "max_characters": 1000,
        "lora_max_tokens": app.state.max_tokens,
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with inference_lock:
        started = perf_counter()

        if request.model == "baseline":
            intent = str(
                app.state.baseline.predict([request.text])[0]
            )
            version = app.state.baseline_metadata["model_version"]
        else:
            inputs = app.state.tokenizer(
                request.text,
                truncation=False,
                return_tensors="pt",
            )

            if inputs["input_ids"].shape[1] > app.state.max_tokens:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Message exceeds the LoRA model's "
                        f"{app.state.max_tokens}-token limit. "
                        "Please shorten it."
                    ),
                )

            inputs = {
                key: value.to(app.state.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                logits = app.state.lora(**inputs).logits
                predicted_id = logits.argmax(dim=-1).item()

            intent = app.state.labels[predicted_id]
            version = app.state.lora_metadata["model_version"]

        if intent not in app.state.labels:
            raise HTTPException(
                status_code=500,
                detail="Model returned an unsupported category.",
            )

        return PredictionResponse(
            intent=intent,
            model_version=version,
            latency_ms=round(
                (perf_counter() - started) * 1000,
                2,
            ),
        )