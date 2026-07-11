# Model Selection Record

## Context

Employees upload receipt images for reimbursement or expense control. The business needs structured fields for monthly reporting and consistency checks.

## Task Type

- Structured extraction: read receipt fields from an image.
- Verification: compare extracted totals and field formats.
- Evaluation: compare model output against `data/labels/expected_receipts.csv`.

## Candidate Models

| Model | Backend | Why Evaluate It | Constraints |
| --- | --- | --- | --- |
| Gemini 2.0 Flash | API | Fast multimodal extraction | Sends data to cloud API |
| Gemini 2.5 Pro | API | Higher quality ceiling | Higher cost and latency |
| LM Studio local model | Local | Keeps data on machine | Requires local setup and hardware |

## Hard Filters

| Filter | Requirement | Result |
| --- | --- | --- |
| Multimodal support | Must process receipt images directly or through a local OCR path | To be completed in class |
| Privacy | Real fiscal data must be handled intentionally | To be completed in class |
| Context window | Must fit prompt plus image/text representation | To be completed in class |
| Output contract | Must produce parseable JSON | To be completed in class |

## Weighted Criteria

| Criterion | Weight | Measurement |
| --- | ---: | --- |
| Extraction accuracy | 50% | Match against CSV expected values |
| Cost per receipt | 20% | Estimated from input/output tokens |
| Latency | 15% | Average seconds per receipt |
| Operational fit | 15% | Setup complexity, privacy, maintainability |

## Decision

Complete after running the 10-ticket evaluation.

Recommended decision format:

> We choose `<model>` for `<use case>` because it achieved `<accuracy>` average accuracy, cost `<cost>` per receipt, and met the privacy/latency constraints.

## Review Date

Re-evaluate when:

- Ticket formats change materially.
- Provider pricing changes.
- A new model family becomes available.
- The dataset grows beyond the initial 10-ticket evaluation.
