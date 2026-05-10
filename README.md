# FinAgent-Eval

A reproducible evaluation harness for visual question answering over financial document screenshots from SEC/company filings. The project evaluates document-AI baselines and modern vision-language models on 397 hand-curated questions across 10 US public companies, with grading, failure analysis, reliability probing, domain-adaptation analysis, and an interactive Streamlit dashboard.

## Key Results

| Model | ANLS | Exact Match | Notes |
|---|---:|---:|---|
| **Qwen2.5-VL-7B-Instruct** | **0.6368** | **0.5945** | Modern open-source VLM, 4-bit Colab T4 run |
| LayoutLMv3 | 0.1472 | 0.1335 | Legacy document baseline |
| Pix2Struct | 0.1284 | 0.1134 | Legacy document baseline |
| OCR+RoBERTa | 0.1108 | 0.0856 | OCR + text QA baseline |
| Donut | 0.1021 | 0.0756 | OCR-free document baseline |

Qwen2.5-VL-7B achieved a **4.3x relative ANLS gain** over the best legacy document baseline, improving ANLS from 0.1472 to 0.6368. However, it still failed 161/397 questions, especially on numerical reasoning.

## Reliability Probe

A 25-question stratified mini-probe tested Qwen2.5-VL with 4 phrasings per task.

- **Answer consistency:** 11/25 = 44.0%
- **Correctness stability:** 20/25 = 80.0%
- **All variants correct:** 9/25 = 36.0%
- **Numerical reasoning:** 0/6 consistent, 0/6 all-correct
- **Layout understanding:** 5/6 consistent, 5/6 all-correct

This suggests Qwen2.5-VL is strong overall but remains brittle under paraphrasing, especially for numerical questions.

## Domain Adaptation

Pix2Struct LoRA was trained on 307 QA pairs from 7 companies and tested on 90 held-out questions from JPM, MSFT, and WMT.

| Model | Held-Out ANLS | Held-Out Exact Match |
|---|---:|---:|
| Pix2Struct zero-shot | 0.0973 | 0.0667 |
| Pix2Struct LoRA | 0.1676 | 0.0556 |

LoRA fine-tuning improved held-out ANLS by **+72.3% relative**, but exact match slightly decreased. This analysis is intentionally separate from the main 397-question leaderboard.

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
pip install streamlit pandas plotly jinja2 pytest
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

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard tabs:

- Overview
- Category Breakdown
- Failure Explorer
- Reliability
- Domain Adaptation
- Dataset Card
- Run Your Own Model

## Reproduce Qwen2.5-VL Baseline

The modern VLM baseline was run in Google Colab with a free T4 GPU.

Open:

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

After downloading the Qwen results into `results/track_a/qwen25vl_7b_results.json`, regenerate local artifacts with:

```bash
python scripts/enrich_results.py
python scripts/generate_reports.py
```

## Evaluation Metrics

- **ANLS:** Average Normalized Levenshtein Similarity, primary metric.
- **Exact Match:** Binary match after text/financial normalization.
- **Numeric Match:** 2% tolerance for numeric answers with percent-status agreement.
- **Bootstrap CIs:** 95% confidence intervals for scorecards.

## Dataset

- 397 questions
- 79 document images
- 10 companies: AAPL, AMZN, BAC, GS, JNJ, JPM, MSFT, TSLA, WMT, XOM
- Categories:
  - extractive: 243
  - layout understanding: 72
  - numerical reasoning: 65
  - chart interpretation: 17

## Limitations

- Single annotator; no inter-annotator agreement study.
- Chart interpretation has only 17 examples.
- Reliability paraphrases were model-generated and may introduce noise.
- Source images are screenshots from public filings, so visual quality varies.
- Pix2Struct LoRA is reported only as a held-out domain-adaptation analysis, not as a full leaderboard baseline.

## License

For non-commercial educational and portfolio evaluation purposes.
