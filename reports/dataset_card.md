# SupportSathi Dataset Card

## Source

Dataset: Hari5115/hinglish-retail-intent-dataset

URL: https://huggingface.co/datasets/Hari5115/hinglish-retail-intent-dataset

Revision: 3b4ab333bfeab463c9407911e958a5e6af6199ac

License listed by the publisher: MIT.

The publisher describes synthetic Hinglish retail-support messages generated
from English seed examples, with a ten-percent manual spot-check.

## Task and coverage

Thirteen retail-support intent labels.

Messages primarily use Roman-script Hinglish. This is not a representative
sample of real customer traffic or all Hindi-speaking populations.

## Preparation

| Split | Source rows | Retained rows |
|---|---:|---:|
| Training | 2,292 | 2,036 |
| Validation | 286 | 278 |
| Test | 287 | 286 |

Duplicate comparison used Unicode NFC normalization, case folding, and
collapsed whitespace.

Matching texts were retained in test before validation, and validation before
training. Duplicate copies in lower-priority splits were removed.

The downloaded source snapshot was preserved.

## Provenance and integrity

Prepared records contain source revision, source split, source row, and an
exact-text group identifier.

The manifest records split sizes, label counts, removal counts, and file hashes.

The group identifier represents normalized exact text. It does not identify
paraphrase families.

## Evaluation rules

Training data was used for fitting.
Validation was used for checkpoint selection.
Test data was excluded from fitting and checkpoint selection.

The earlier manually generated ten-label dataset is not part of this
thirteen-label experiment.

## Limitations

- Data is synthetic and only partially reviewed by its publisher.
- Paraphrase and template leakage remain unaudited.
- Annotation ambiguities may affect measured performance.
- Cleaned splits differ from the publisher's original benchmark.
- Evaluation does not establish production or real-customer accuracy.

## Files

Prepared data: data/processed/retail_public_v1/

Source snapshot: data/raw/public_retail/

Manifest: data/processed/retail_public_v1/manifest.json

Inspection report: reports/public_dataset_inspection.json