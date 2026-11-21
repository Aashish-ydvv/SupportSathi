from pathlib import Path
from time import perf_counter

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/baseline.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError("Model missing. Run scripts/train_baseline.py first.")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SupportSathi API",
    version="0.1.0",
    description="Starter support intent classifier. Not yet evaluated.",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)

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
    return {"status": "ok", "model_loaded": True}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    started = perf_counter()
    intent = str(model.predict([request.text])[0])
    latency_ms = (perf_counter() - started) * 1000

    return PredictionResponse(
        intent=intent,
        model_version="tfidf-logistic-seed-v1",
        latency_ms=round(latency_ms, 2),
    )