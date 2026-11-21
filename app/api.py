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
ADAPTER_DIR = ROOT / "models/lora_v2"
MAX_TOKENS = 128
inference_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    metadata = json.loads(
        (ADAPTER_DIR / "training_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    labels = metadata["labels"]

    app.state.baseline = joblib.load(
        ROOT / "models/baseline_v2.joblib"
    )
    app.state.tokenizer = XLMRobertaTokenizer.from_pretrained(
        ADAPTER_DIR
    )

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.to(device)
    model.eval()

    app.state.lora = model
    app.state.device = device
    app.state.labels = labels

    yield

    del app.state.lora
    if device == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(
    title="SupportSathi API",
    version="0.2.0",
    description=(
        "Support intent classification with TF-IDF and LoRA models. "
        "Development prototype trained on synthetic examples."
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
        "lora_device": app.state.device,
    }


@app.get("/v1/model-info")
def model_info():
    return {
        "default_model": "baseline",
        "available_models": {
            "baseline": "tfidf-logistic-v2",
            "lora": "minilm-lora-v2",
        },
        "labels": app.state.labels,
        "training_examples": 100,
        "data_source": "AI-generated development examples",
        "evaluation_status": "development_validation_only",
        "max_characters": 1000,
        "lora_max_tokens": MAX_TOKENS,
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with inference_lock:
        started = perf_counter()

        if request.model == "baseline":
            intent = str(
                app.state.baseline.predict([request.text])[0]
            )
            version = "tfidf-logistic-v2"
        else:
            inputs = app.state.tokenizer(
                request.text,
                truncation=False,
                return_tensors="pt",
            )

            if inputs["input_ids"].shape[1] > MAX_TOKENS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Message exceeds the LoRA model's "
                        "128-token limit. Please shorten it."
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
            version = "minilm-lora-v2"

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