# SupportSathi

SupportSathi classifies Roman-script Hinglish retail-support messages into
13 intent categories. It compares a character TF-IDF classifier with a
multilingual MiniLM model adapted using LoRA.

The project includes local training, evaluation, a FastAPI prediction service,
and a Streamlit demo.

## What the project demonstrates

- Preparing a public intent-classification dataset.
- Detecting and removing exact duplicate text across dataset splits.
- Training a traditional text-classification baseline.
- Applying LoRA while also training the classifier and pooling layer.
- Selecting a checkpoint using validation macro-F1.
- Evaluating frozen models on a separate test split.
- Serving both models through a validated API.
- Measuring warm, sequential local response times.

This version compares TF-IDF with MiniLM + LoRA. A larger prompted-model
comparison and operating-cost study are outside the completed scope.

## Dataset

Source: [Hinglish Retail Intent Dataset](https://huggingface.co/datasets/Hari5115/hinglish-retail-intent-dataset)

Dataset revision used:

`3b4ab333bfeab463c9407911e958a5e6af6199ac`

The publisher lists an MIT license. The dataset is synthetic and primarily
contains Hindi-English code-switching written in Roman script.

| Split | Original rows | Cleaned rows |
|---|---:|---:|
| Training | 2,292 | 2,036 |
| Validation | 286 | 278 |
| Test | 287 | 286 |

Cleaning normalized Unicode to NFC, folded case, and collapsed whitespace
for duplicate comparisons. Matching texts were retained in priority order:
test, validation, then training.

The source snapshot is preserved locally. Cleaning outputs include row
provenance, dataset revision, split hashes, and label counts.

These are custom cleaned splits. Their results should not be presented as
scores on the publisher's unchanged benchmark.

## Intent categories

- cancel_order
- damaged_item
- delivery_complaint
- discount_query
- exchange_product
- other
- payment_issue
- product_availability
- product_information
- refund_status
- return_request
- track_order
- wrong_item

Definitions are stored in `configs/labels_public_v1.json`.

## Models

### TF-IDF baseline

Character n-grams of length 2–5, sublinear TF-IDF, and logistic regression.

### MiniLM + LoRA

Base model: `microsoft/Multilingual-MiniLM-L12-H384`

- LoRA rank: 4
- LoRA alpha: 8
- LoRA dropout: 0.1
- Target modules: query and value
- Additional trained modules: classifier and pooler
- Batch size: 4
- Learning rate: 0.0005
- Maximum sequence length: 128 tokens
- Maximum epochs: 8
- Early-stopping patience: 2 epochs
- Random seed: 42
- Selected checkpoint: epoch 8
- Selection criterion: validation macro-F1

Base-model revision and training-file hashes are recorded in model metadata.

## Results

### Validation

| Model | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|
| TF-IDF | 95.68% | 0.9586 | 12/278 |
| MiniLM + LoRA | 93.53% | 0.9360 | 18/278 |

Validation was used for LoRA checkpoint selection.

### Final test

| Model | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|
| TF-IDF | 96.85% | 0.9685 | 9/286 |
| MiniLM + LoRA | 96.15% | 0.9607 | 11/286 |

TF-IDF made two fewer errors on this test set. These results do not demonstrate
a performance advantage for LoRA.

Final-test data was excluded from model fitting and checkpoint selection.
No further training changes were made in response to these test results.

Detailed reports contain per-class metrics, confusion matrices, predictions,
and model-artifact hashes.

## Latency

Measured using the same 100 sampled validation messages for each model,
five warm-up requests per model, and sequential HTTP requests.

| Model | HTTP p50 | HTTP p95 | Processing p50 | Processing p95 |
|---|---:|---:|---:|---:|
| TF-IDF | 9.16 ms | 16.09 ms | 2.77 ms | 7.15 ms |
| MiniLM + LoRA | 42.58 ms | 76.81 ms | 38.48 ms | 72.05 ms |

Hardware: Intel Core i5-10300H, 8 GB RAM, NVIDIA GTX 1650 Ti with 4 GB VRAM.
TF-IDF runs on CPU; LoRA inference used the GPU.

HTTP timings include the local request round trip. API processing timings
exclude request queueing and network transfer.

These measurements exclude model startup and do not represent concurrent-load
performance. Raw timings and hardware details are saved in
`reports/latency_public.json`.

## Local setup

Development used Windows and Python 3.12.6.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the recorded dependencies. The additional index supplies the
CUDA-enabled PyTorch build used by this project:

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

GPU training requires a compatible NVIDIA driver and working CUDA-enabled
PyTorch installation. Check it with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Prepare data and train

```powershell
python scripts/download_retail_dataset.py
python scripts/prepare_public_dataset.py
python scripts/train_baseline.py
python scripts/train_lora.py
```

The downloader records the source revision. For reproduction of the reported
results, use the dataset revision documented above.

Evaluate:

```powershell
python scripts/evaluate_validation.py
python scripts/evaluate_test.py
```

Use validation for development. Do not repeatedly tune models against the
final test set.

Large model files and downloaded source data are excluded from Git.
They must be generated locally or obtained as a separately documented release.

## Run the app

Start the API in one activated terminal:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`.

Start the interface in a second activated terminal:

```powershell
python -m streamlit run app/ui.py
```

- Demo: http://localhost:8501
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## API example

Send a POST request to `/v1/predict`:

```json
{
  "text": "Mera refund kab aayega?",
  "model": "lora"
}
```

The response contains:

- `intent`: one of the thirteen supported labels.
- `model_version`: the model used.
- `latency_ms`: measured API processing time.

Use `"baseline"` to select TF-IDF. Baseline is the default.

Other endpoints:

- `GET /health`
- `GET /v1/model-info`

Input must contain non-whitespace text and at most 1,000 characters.
LoRA requests exceeding 128 tokens are rejected instead of silently truncated.

## API checks and benchmarking

With the API running:

```powershell
python scripts/check_api.py
python scripts/benchmark_latency.py
```

API checks cover readiness, model versions, label configuration, invalid inputs,
token limits, and response structure. They are not a complete security audit.

## Limitations

- Data is synthetic and does not establish real-customer accuracy.
- Exact duplicates were removed, but paraphrase leakage remains unaudited.
- Some examples may have ambiguous or inconsistent labels.
- Devanagari Hindi performance has not been separately evaluated.
- Predictions have no calibrated confidence or reliable abstention mechanism.
- The classifier can assign an inappropriate label to an unsupported message.
- The service is intended for local demonstration.
- Public-deployment authentication, rate limits, and load testing are pending.
- Electricity, hardware depreciation, hosting, and maintenance costs were not measured.
- No claim of a 50% operating-cost reduction is made.

## Failure analysis

Both models confuse some order-tracking requests with delivery complaints.
Other mistakes involve exchanges versus wrong-item reports, and damaged-item
reports versus other product or delivery issues.

These errors reflect overlapping intent boundaries as well as model limitations.
They should be reviewed before using the classifier for actual support routing.

## Next improvements

- Review label consistency and paraphrase overlap.
- Evaluate independently collected, consented customer-style messages.
- Compare LoRA with a frozen-encoder classifier.
- Add a larger prompted-model baseline if the scope is extended.
- Evaluate confidence calibration and a human-review policy.
- Package and test deployment separately from local development.