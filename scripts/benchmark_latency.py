import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000"
SAMPLE_SIZE = 100
WARMUP_REQUESTS = 5
SEED = 42


def summarize(values):
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "mean_ms": round(float(np.mean(values)), 2),
    }


def gpu_details():
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "Unavailable"


def predict(session, text, model, expected_version, allowed_labels):
    started = perf_counter()
    response = session.post(
        f"{API_URL}/v1/predict",
        json={"text": text, "model": model},
        timeout=60,
    )
    client_ms = (perf_counter() - started) * 1000

    if response.status_code != 200:
        raise RuntimeError(
            f"{model}: HTTP {response.status_code}: {response.text[:500]}"
        )

    result = response.json()
    if result["model_version"] != expected_version:
        raise RuntimeError("Unexpected model version.")
    if result["intent"] not in allowed_labels:
        raise RuntimeError("Unexpected intent.")

    return client_ms, float(result["latency_ms"])


def main():
    validation_path = (
        ROOT / "data/processed/retail_public_v1/validation.jsonl"
    )
    examples = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not examples:
        raise ValueError("Validation data is empty.")

    sample = random.Random(SEED).sample(
        examples, min(SAMPLE_SIZE, len(examples))
    )

    with requests.Session() as session:
        health_response = session.get(
            f"{API_URL}/health", timeout=30
        )
        health_response.raise_for_status()
        health = health_response.json()

        info_response = session.get(
            f"{API_URL}/v1/model-info", timeout=30
        )
        info_response.raise_for_status()
        info = info_response.json()

        if len(info["labels"]) != 13:
            raise RuntimeError("Start the updated thirteen-label API.")

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement": "local HTTP requests, sequential, batch size one",
            "sample_split": "validation",
            "sample_ids": [example["id"] for example in sample],
            "requests_per_model": len(sample),
            "warmup_requests_per_model": WARMUP_REQUESTS,
            "sample_seed": SEED,
            "hardware": {
                "cpu": platform.processor(),
                "os": platform.platform(),
                "gpu": gpu_details(),
                "lora_device": health["lora_device"],
            },
            "models": {},
        }

        for model in ["baseline", "lora"]:
            version = info["available_models"][model]
            allowed_labels = set(info["labels"])

            print(f"\nWarming up {model}...", flush=True)
            for index in range(WARMUP_REQUESTS):
                predict(
                    session,
                    sample[index % len(sample)]["text"],
                    model,
                    version,
                    allowed_labels,
                )

            client_times = []
            processing_times = []

            for index, example in enumerate(sample, start=1):
                client_ms, processing_ms = predict(
                    session,
                    example["text"],
                    model,
                    version,
                    allowed_labels,
                )
                client_times.append(client_ms)
                processing_times.append(processing_ms)

                if index % 20 == 0:
                    print(
                        f"{model}: {index}/{len(sample)} requests",
                        flush=True,
                    )

            client_summary = summarize(client_times)
            processing_summary = summarize(processing_times)

            report["models"][model] = {
                "model_version": version,
                "client_round_trip": client_summary,
                "api_processing": processing_summary,
                "client_times_ms": client_times,
                "api_processing_times_ms": processing_times,
            }

            print(
                f"{model} HTTP round trip: "
                f"p50={client_summary['p50_ms']} ms, "
                f"p95={client_summary['p95_ms']} ms"
            )
            print(
                f"{model} API processing: "
                f"p50={processing_summary['p50_ms']} ms, "
                f"p95={processing_summary['p95_ms']} ms"
            )

    output_path = ROOT / "reports/latency_public.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved benchmark: {output_path}")


if __name__ == "__main__":
    main()