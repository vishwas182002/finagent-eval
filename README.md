# FinAgent-Eval

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)](#launch-dashboard)
[![Colab](https://img.shields.io/badge/reproduce-Colab%20T4-orange)](https://colab.research.google.com/github/vishwas182002/finagent-eval/blob/main/notebooks/finagent_eval_qwen_vl_inference.ipynb)
[![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen)](#run-tests)

**FinAgent-Eval** is a reproducible evaluation harness for visual question answering over financial document screenshots from SEC/company filings. It evaluates document-AI baselines and modern vision-language models on **397 hand-curated questions** across **79 document images** and **10 US public companies**, with grading, failure analysis, reliability probing, domain-adaptation analysis, and an interactive Streamlit dashboard.

The project asks a practical question:

> How much do modern VLMs actually improve financial document understanding, and where do they still fail?

## Highlights

- Benchmarks **5 models** on the same 397-question Track A dataset.
- Adds a modern open-source VLM baseline: **Qwen2.5-VL-7B-Instruct**.
- Implements financial-aware grading: ANLS, exact match, numeric tolerance, and financial number normalization.
- Adds failure labels for incorrect answers, including numeric mismatch, wrong year, empty output, and manual diagnosis.
- Includes a **25-question paraphrase reliability probe** to test answer stability.
- Includes a separate **Pix2Struct LoRA domain-adaptation analysis** on 90 held-out questions.
- Ships a polished Streamlit dashboard for leaderboard, failures, reliability, and domain-adaptation views.

## Key Results

| Model | ANLS | Exact Match | Notes |
|---|---:|---:|---|
| **Qwen2.5-VL-7B-Instruct** | **0.6368** | **0.5945** | Modern open-source VLM, 4-bit Colab T4 run |
| LayoutLMv3 | 0.1472 | 0.1335 | Legacy document baseline |
| Pix2Struct | 0.1284 | 0.1134 | Legacy document baseline |
| OCR+RoBERTa | 0.1108 | 0.0856 | OCR + text QA baseline |
| Donut | 0.1021 | 0.0756 | OCR-free document baseline |

**Headline finding:** Qwen2.5-VL-7B achieved a **4.3x relative ANLS gain** over the best legacy document baseline, improving ANLS from **0.1472** to **0.6368**. It also improved exact match from **13.35%** to **59.45%**.

The improvement is large, but not complete: Qwen still failed **161/397** questions, with numerical reasoning remaining the clearest weakness.

## Why This Matters

Financial documents are not ordinary OCR tasks. They combine dense tables, visual layout, accounting formats, temporal references, charts, and numeric reasoning. A model can read a number correctly and still answer the wrong year, wrong row, or wrong table section.

FinAgent-Eval is designed to make those failure modes visible rather than hiding them behind one aggregate score.

## Dashboard

Run the dashboard locally:

```bash
streamlit run dashboard/app.py
```

Dashboard tabs:

- **Overview** - leaderboard, ANLS vs exact match, confidence intervals
- **Category Breakdown** - heatmap and per-category metrics
- **Failure Explorer** - inspect model failures with source document images
- **Reliability** - paraphrase consistency and correctness stability
- **Domain Adaptation** - Pix2Struct zero-shot vs LoRA fine-tuned analysis
- **Dataset Card** - dataset composition, companies, limitations
- **Run Your Own Model** - expected JSON schema and integration workflow

## Reproduce Qwen2.5-VL Baseline

The modern VLM baseline was run in Google Colab with a free T4 GPU.

[Open the Qwen inference notebook in Colab](https://colab.research.google.com/github/vishwas182002/finagent-eval/blob/main/notebooks/finagent_eval_qwen_vl_inference.ipynb)

Notebook path:

```text
notebooks/finagent_eval_qwen_vl_inference.ipynb
```

The notebook includes:

- GPU and dependency setup
- Dataset download and task construction
- Qwen2.5-VL-7B-Instruct loading with 4-bit quantization
- Single-question smoke test
- Full 397-question inference with checkpointing
- Download step for `qwen25vl_7b_results.json`
- Optional 25-question paraphrase reliability probe

After downloading the Qwen predictions into `results/track_a/qwen25vl_7b_results.json`, regenerate local artifacts:

```bash
python scripts/enrich_results.py
python scripts/generate_reports.py
```

## Reliability Probe

A 25-question stratified mini-probe tested Qwen2.5-VL with 4 phrasings per task: the original question plus 3 paraphrases.

| Metric | Result |
|---|---:|
| Answer consistency | 11/25 = 44.0% |
| Correctness stability | 20/25 = 80.0% |
| Original-question accuracy | 13/25 = 52.0% |
| All variants correct | 9/25 = 36.0% |
| Any variant correct | 14/25 = 56.0% |

Per-category finding:

| Category | Consistent | All Correct |
|---|---:|---:|
| Extractive | 4/8 | 3/8 |
| Layout understanding | 5/6 | 5/6 |
| Numerical reasoning | 0/6 | 0/6 |
| Chart interpretation | 2/5 | 1/5 |

This suggests Qwen2.5-VL is strong overall but brittle under paraphrasing, especially for numerical questions.

## Domain Adaptation

Pix2Struct LoRA was trained on **307 QA pairs** from 7 companies and tested on **90 held-out questions** from JPM, MSFT, and WMT.

| Model | Held-Out ANLS | Held-Out Exact Match |
|---|---:|---:|
| Pix2Struct zero-shot | 0.0973 | 0.0667 |
| Pix2Struct LoRA | 0.1676 | 0.0556 |

LoRA fine-tuning improved held-out ANLS by **+72.3% relative**, but exact match slightly decreased. This analysis is intentionally separate from the main 397-question leaderboard because it uses only the 90-question held-out subset.

## Project Structure

```text
finagent-eval/
├── finagent_eval/
│   ├── graders/
│   ├── failure_modes.py
│   └── report.py
├── scripts/
│   ├── build_tasks.py
│   ├── normalize_results.py
│   ├── validate_data.py
│   ├── enrich_results.py
│   ├── generate_reports.py
│   └── analyze_domain_adaptation.py
├── dashboard/
│   └── app.py
├── notebooks/
│   └── finagent_eval_qwen_vl_inference.ipynb
├── templates/
├── tests/
├── data/track_a/
├── results/
│   ├── track_a/
│   ├── reliability/
│   └── domain_adaptation/
└── reports/
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Pipeline

```bash
python scripts/build_tasks.py
python scripts/normalize_results.py
python scripts/validate_data.py
python scripts/enrich_results.py
python scripts/generate_reports.py
python scripts/analyze_domain_adaptation.py
```

## Run Tests

```bash
python -m pytest tests -v
```

Expected:

```text
11 passed
```

## Evaluation Metrics

- **ANLS:** Average Normalized Levenshtein Similarity, the primary metric.
- **Exact Match:** Binary match after text and financial normalization.
- **Numeric Match:** 2% tolerance for numeric answers, with percent-status agreement.
- **Bootstrap CIs:** 95% confidence intervals for aggregate scorecards.

Financial normalization handles:

- Currency symbols: `$394,328` to `394328`
- Accounting negatives: `($500)` to `-500`
- Suffixes: `1.5M` to `1500000`
- Percentages: `15.7%` preserved as a percentage
- Whitespace and case normalization

## Failure Taxonomy

Incorrect answers are assigned one of the following labels:

| Label | Meaning |
|---|---|
| `empty_output` | Model returned no answer |
| `abstention` | Model explicitly refused or said unavailable |
| `format_mismatch` | Numerically equivalent but formatted differently |
| `numeric_mismatch` | Number differs beyond tolerance |
| `wrong_year` | Answer references a different year than the question |
| `hallucinated_number` | Predicted number is not present in supplied source text |
| `manual_diagnosis_required` | Needs human inspection |

## Dataset

- **397 questions**
- **79 document images**
- **10 companies:** AAPL, AMZN, BAC, GS, JNJ, JPM, MSFT, TSLA, WMT, XOM
- **4 categories:**
  - extractive: 243
  - layout understanding: 72
  - numerical reasoning: 65
  - chart interpretation: 17

## Add Your Own Model

Save one prediction row per task in `results/track_a/your_model.json`:

```json
{
  "task_id": "AAPL_001",
  "model": "YourModel",
  "prediction": "143,756",
  "gold_answer": "143,756",
  "category": "extractive",
  "latency_ms": 1250,
  "evidence_available": false,
  "evidence": null,
  "raw_output": "143,756"
}
```

Then run:

```bash
python scripts/enrich_results.py
python scripts/generate_reports.py
streamlit run dashboard/app.py
```

## Limitations

- Single annotator; no inter-annotator agreement study.
- Chart interpretation has only 17 examples, so chart-specific comparisons are directional.
- Reliability paraphrases were model-generated and may introduce noise.
- Source images are screenshots from public filings, so visual quality varies.
- Pix2Struct LoRA is reported only as a held-out domain-adaptation analysis, not as a full leaderboard baseline.

## License

For non-commercial educational and portfolio evaluation purposes.
