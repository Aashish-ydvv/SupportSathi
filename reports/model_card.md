# SupportSathi Model Card

## Purpose

Classify Roman-script Hinglish retail-support messages into thirteen intents.
Intended for demonstration and support-routing experiments.

This model does not generate customer replies.

## Model

- Version: retail-minilm-lora-public-v1
- Base: microsoft/Multilingual-MiniLM-L12-H384
- Base license listed by publisher: MIT
- Method: LoRA on query and value projections
- Additional trained layers: classifier and pooler
- LoRA rank: 4
- LoRA alpha: 8
- LoRA dropout: 0.1
- Maximum sequence length: 128 tokens
- Batch size: 4
- Learning rate: 0.0005
- Random seed: 42
- Selected checkpoint: epoch 8
- Selection metric: validation macro-F1

The exact base-model revision is recorded in the saved training metadata.

## Data

Source: Hari5115/hinglish-retail-intent-dataset

Cleaned split sizes:

- Training: 2,036
- Validation: 278
- Final test: 286

The data is synthetic. Exact normalized text duplicates were removed.
Paraphrase leakage has not been audited.

## Evaluation

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 93.53% | 0.9360 |
| Final test | 96.15% | 0.9607 |

Validation was used for checkpoint selection. The final test was evaluated
after training and checkpoint selection.

The TF-IDF baseline scored 96.85% accuracy and 0.9685 macro-F1 on the same
final test set. The experiment does not demonstrate superiority over TF-IDF.

## Latency

Warm sequential local HTTP requests:

- Median: 42.58 ms
- p95: 76.81 ms
- Sample: 100 validation messages
- Warm-up: five requests
- GPU: NVIDIA GTX 1650 Ti, 4 GB
- System: Intel Core i5-10300H, 8 GB RAM

These measurements exclude model startup and do not describe concurrent load.

## Limitations

- Synthetic-data scores do not establish real-customer accuracy.
- Similar templates or paraphrases may occur across splits.
- Some intent boundaries and source annotations are ambiguous.
- Devanagari Hindi has not been separately evaluated.
- No calibrated confidence or validated abstention mechanism is provided.
- Unsupported messages can receive incorrect supported labels.
- Public deployment requires additional operational safeguards.

## Intended use

Local demonstrations, reproducible experiments, and supervised routing research.

Do not use these results to justify unattended production routing or claims of
cost savings. Human review and independent evaluation are needed first.

## Evidence

- public_validation.json
- public_test.json
- latency_public.json
- api_checks_public.json