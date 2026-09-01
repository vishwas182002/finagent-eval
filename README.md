# FinAgent-Eval

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)](#dashboard)
[![Colab](https://img.shields.io/badge/reproduce-Colab%20T4-orange)](https://colab.research.google.com/github/vishwas182002/finagent-eval/blob/main/notebooks/finagent_eval_qwen_vl_inference.ipynb)
[![CI](https://github.com/vishwas182002/finagent-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/vishwas182002/finagent-eval/actions/workflows/tests.yml)
[![Grader](https://img.shields.io/badge/grader-v2.0-informational)](CHANGELOG.md)

[**Live Dashboard**](https://finagent-eval.streamlit.app/) ·
[**Qwen Colab Notebook**](https://colab.research.google.com/github/vishwas182002/finagent-eval/blob/main/notebooks/finagent_eval_qwen_vl_inference.ipynb) ·
[**Changelog**](CHANGELOG.md)

**FinAgent-Eval** is a reproducible evaluation harness for visual question answering over financial document screenshots from SEC/company filings. It evaluates document-AI baselines and modern vision-language models on **397 hand-curated questions** across **79 document images** and **10 US public companies**, with unit-aware financial grading, an automatic failure taxonomy, bootstrap confidence intervals, a paraphrase reliability probe, a domain-adaptation analysis, a pluggable model-adapter runner, and an interactive Streamlit dashboard.

The project asks a practical question:

> How much do modern VLMs actually improve financial document understanding, and where do they still fail?

## Highlights

- Benchmarks **5 models** on the same 397-question Track A dataset, all re-graded by one grader.
- Adds a modern open-source VLM baseline: **Qwen2.5-VL-7B-Instruct** (4-bit, free Colab T4).
- **Financial-aware grading (grader v2.0):** `$85,269 million` matches a table value `85,269`; accounting negatives, currency symbols, unit suffixes, percent words, answer prefixes and unicode punctuation are all normalized. Numeric tolerance applies only where rounding is legitimate (numerical reasoning, charts).
- **Automatic failure taxonomy** with 13 labels; on the current results zero rows need manual diagnosis.
- **25-question paraphrase reliability probe** (4 phrasings each), graded with the same `grade()` as the leaderboard.
- **Pix2Struct LoRA domain-adaptation analysis** on 90 held-out questions, kept off the main leaderboard.
- **Adapter interface + runner:** implement `predict(image_path, question)`, get checkpointed predictions plus a reproducibility sidecar (prompt, decoding settings, dataset hash, grader version).
- **96 tests** including regression tests that pin every published number, and a CI job that regenerates all artifacts and fails on any diff.

## Key Results

All numbers are recomputed by the project grader (v2.0) from the raw predictions committed in `results/track_a/`.

| Model | ANLS | 95% CI | Exact Match | Project-correct | Notes |
|---|---:|---:|---:|---:|---|
| **Qwen2.5-VL-7B-Instruct** | **0.7378** | [0.695, 0.781] | **0.6877** | **277/397 (69.8%)** | Modern open-source VLM, 4-bit Colab T4 run |
| LayoutLMv3 | 0.1533 | [0.120, 0.188] | 0.1335 | 53/397 | Legacy document baseline |
| Pix2Struct | 0.1343 | [0.102, 0.165] | 0.1134 | 45/397 | Legacy document baseline |
| OCR+RoBERTa | 0.1132 | [0.085, 0.142] | 0.0856 | 34/397 | OCR + text QA baseline |
| Donut | 0.1061 | [0.079, 0.136] | 0.0756 | 30/397 | OCR-free document baseline |

*Project-correct* = exact match after financial normalization, or within 2% numeric tolerance on numerical-reasoning and chart questions.

**Headline finding:** Qwen2.5-VL-7B achieves a **4.8x relative ANLS gain** over the best legacy document baseline (0.1533 to 0.7378) and lifts exact match from **13.4%** to **68.8%**.

The improvement is large but uneven. Per category (Qwen, project-correct / ANLS):

| Category | n | Correct | ANLS |
|---|---:|---:|---:|
| Layout understanding | 72 | 67 (93%) | 0.966 |
| Extractive | 243 | 188 (77%) | 0.812 |
| Chart interpretation | 17 | 12 (71%) | 0.795 |
| **Numerical reasoning** | **65** | **10 (15%)** | **0.194** |

Qwen reads tables well and computes badly: on questions that require arithmetic over the document ("by what percentage did X increase from 2024 to 2025?") it is correct on **10 of 65**. Of its 120 remaining failures, 78 are `numeric_mismatch` (wrong value, beyond tolerance), 29 `text_mismatch` (wrong row/entity, e.g. "Net product sales" for "Net service sales"), 7 `near_miss_numeric` (a misread digit within 2% on a verbatim number), 3 `partial_text_match`, 3 `unit_mismatch`.

> **Grader revision.** These figures supersede the v1 numbers (Qwen ANLS 0.6368, 246/397). The v1 grader did not understand unit words and scored 35 correct Qwen answers as failures; it also accepted misread digits on extractive questions. See [CHANGELOG.md](CHANGELOG.md) for the full before/after table and rationale.

## Why This Matters

Financial documents are not ordinary OCR tasks. They combine dense tables, visual layout, accounting formats, temporal references, charts, and numeric reasoning. A model can read a number correctly and still answer the wrong year, wrong row, or wrong table section, and a grader can penalise a correct answer for writing "$85,269 million" instead of "85,269".

FinAgent-Eval is designed to make those failure modes visible rather than hiding them behind one aggregate score, and to make the grader itself auditable.

## Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

Tabs: **Overview** (leaderboard, ANLS vs EM, CIs), **Category Breakdown**, **Failure Explorer** (inspect failures with the source image), **Reliability**, **Domain Adaptation**, **Dataset Card**, **Run Your Own Model**.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"        # package + CLI + dashboard + test tooling
finagent-eval --version       # finagent-eval 0.2.0 (grader v2.0)
```

`pip install -r requirements.txt` installs the exact pinned versions used by CI and Streamlit Cloud.

## Run Pipeline

```bash
finagent-eval all            # every stage below, in order
```

or stage by stage:

```bash
finagent-eval build-tasks          # raw annotations -> data/track_a/tasks.json
finagent-eval normalize-results    # 2024 baseline files -> results/track_a/*.json
finagent-eval validate             # schema / coverage checks on every prediction file
finagent-eval enrich               # recompute grades + failure labels -> results/track_a/enriched/
finagent-eval report               # JSON + HTML scorecards and comparison -> reports/
finagent-eval domain-adaptation    # Pix2Struct zero-shot vs LoRA held-out analysis
finagent-eval regrade-reliability  # re-grade the paraphrase probe with the project grader
```

Every stage is a pure function of the project root (`finagent_eval/pipeline/`), outputs are deterministic (seeded bootstrap, no timestamps), and CI regenerates everything and fails if a committed artifact differs. The old `python scripts/<stage>.py` entry points still work as thin wrappers.

Grade a single answer from the shell:

```bash
finagent-eval grade '$85,269 million' '85,269' --category extractive
```

## Run Tests

```bash
python -m pytest tests -v          # 96 tests
python -m pytest -m "not slow"     # skip the ~10s dashboard smoke test
ruff check .
```

`tests/test_regression.py` pins every leaderboard number, the numerical-reasoning headline and the reliability summary, and checks the committed enriched files equal a fresh recomputation. A grader change that moves any number fails CI until `GRADER_VERSION` is bumped, the CHANGELOG is updated and artifacts are regenerated.

## Add Your Own Model

Implement one adapter:

```python
# my_model.py
from pathlib import Path
from finagent_eval.adapters import BaseAdapter

class MyAdapter(BaseAdapter):
    name = "MyModel"

    def setup(self):
        self.model = load_my_model()

    def predict(self, image_path: Path, question: str) -> str:
        return self.model.answer(image=str(image_path), question=question)

    def metadata(self):  # recorded in the .meta.json sidecar
        return {"prompt_template": "...", "decoding": {"temperature": 0}}
```

Run it:

```bash
finagent-eval run --adapter my_model:MyAdapter --out results/track_a/my_model.json --then-enrich
finagent-eval run --adapter qwen2.5-vl --out results/track_a/qwen_rerun.json   # needs a GPU: pip install -e ".[vlm]"
```

The runner checkpoints every 25 tasks, resumes on restart, records per-row latency, survives adapter exceptions (logged, empty prediction) and writes `results/track_a/my_model.meta.json` with the prompt, decoding settings, dataset hash, package/grader version and git sha. `--limit 10` runs a smoke test.

If you produced predictions elsewhere, save one row per task:

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

then `finagent-eval validate && finagent-eval enrich && finagent-eval report`. Raw prediction files never carry scores; every metric is recomputed at enrichment.

## Reproduce Qwen2.5-VL Baseline

The modern VLM baseline was run in Google Colab on a free T4. [Open the notebook](https://colab.research.google.com/github/vishwas182002/finagent-eval/blob/main/notebooks/finagent_eval_qwen_vl_inference.ipynb) (`notebooks/finagent_eval_qwen_vl_inference.ipynb`). It clones this repository, installs the package, loads Qwen2.5-VL-7B-Instruct in 4-bit (nf4, fp16 compute), runs all 397 questions greedily with `max_new_tokens=64` and checkpointing, and optionally runs the reliability probe. The exact prompt and settings are recorded in `results/track_a/qwen25vl_7b_results.meta.json` and in `finagent_eval/adapters/qwen_vl.py`.

## Evaluation Metrics

- **ANLS** (primary): Average Normalized Levenshtein Similarity with the DocVQA 0.5 threshold, computed on normalized text. Grader v2 short-circuits to 1.0 when both answers parse as equivalent financial numbers; the plain text variant is kept as `raw_anls`.
- **Exact Match:** equality after financial normalization for numbers, or after text normalization (case, whitespace, unicode punctuation, answer prefixes, leading article) for text.
- **Numeric tolerance:** 2% relative, percent-status must agree, never for year-like answers, and only counted as correct for `numerical_reasoning` and `chart_interpretation`.
- **Bootstrap CIs:** 95% percentile intervals, 1,000 resamples, seed 42.

Financial normalization (`finagent_eval.graders.parse_financial_number`) handles currency symbols and words (`$`, `US$`, `USD`, `dollars`), thousands separators, accounting negatives `(500)`, percent signs and words, unit suffixes and words (`1.5M`, `2.3B`, `85,269 million`, `53.8 billion`), answer prefixes (`Answer:`, `approximately`) and unicode dashes/quotes. An explicit unit on exactly one side is treated as the table's implicit unit, so `$85,269 million` equals `85,269` but `1.5 billion` does not equal `1.5 million`.

## Failure Taxonomy

Applied to every prediction that is not project-correct, in this order:

| Label | Meaning |
|---|---|
| `empty_output` | Model returned nothing |
| `abstention` | Explicit refusal / "not available" |
| `format_mismatch` | Same number, rejected only by a stricter grader (unreachable under the project grader) |
| `sign_mismatch` | Same magnitude, opposite sign (`39,371` for `(39,371)`) |
| `unit_mismatch` | Percent vs non-percent, or conflicting explicit units |
| `wrong_year` | References a year the question does not, or names the wrong year when the gold is a year |
| `hallucinated_number` | Number absent from the supplied source text (needs OCR text; not produced by the default pipeline) |
| `numeric_mismatch` | Both numeric, beyond tolerance |
| `near_miss_numeric` | Within 2% but tolerance does not apply to the category: a misread digit on a verbatim number |
| `type_mismatch` | Numeric vs text (`Yes` for `4,396`) |
| `partial_text_match` | One text answer contains the other (`Daily VaR` vs `Daily VaR (Value at Risk)`) |
| `text_mismatch` | Unrelated text answers (wrong row / entity) |
| `manual_diagnosis_required` | Fallback (currently 0 rows) |

## Reliability Probe

25 stratified questions, each asked 4 ways (original + 3 paraphrases generated by Qwen itself), graded with the project grader:

| Metric | Result |
|---|---:|
| Answer consistency (same canonical answer across all 4) | 11/25 = 44% |
| Correctness stability (all right or all wrong) | 18/25 = 72% |
| Original-question accuracy | 15/25 = 60% |
| All variants correct | 9/25 = 36% |
| Any variant correct | 16/25 = 64% |

| Category | Consistent | All correct |
|---|---:|---:|
| Extractive | 4/8 | 3/8 |
| Layout understanding | 5/6 | 5/6 |
| Numerical reasoning | **0/6** | **0/6** |
| Chart interpretation | 2/5 | 1/5 |

Qwen is stable on layout questions and brittle on numerical ones: rephrasing a percentage-change question changes the number it computes. Caveat: the paraphrases were generated by the model under test, and n=25.

## Domain Adaptation

Pix2Struct LoRA was trained on 307 QA pairs from 7 companies and tested on 90 held-out questions from JPM, MSFT and WMT. Reported separately, not as a leaderboard entry.

| Model (held-out, project grader) | ANLS | Exact Match |
|---|---:|---:|
| Pix2Struct zero-shot | 0.1000 | 0.0778 |
| Pix2Struct LoRA | 0.1348 | 0.0556 |

The original run's stored evaluator reports 0.0973 → 0.1676 (+72% relative). Under both graders LoRA moves answers closer on average (+0.035 ANLS under the project grader) without improving exact match: ANLS improves on 13 held-out questions and worsens on 10, but only 5 questions become correct while 7 become wrong, and all 2 correct chart answers are lost.

## Dataset

397 questions over 79 screenshots from 10 companies (AAPL, AMZN, BAC, GS, JNJ, JPM, MSFT, TSLA, WMT, XOM). Categories: extractive 243, layout understanding 72, numerical reasoning 65, chart interpretation 17. JNJ (97) and GS (70) account for 42% of questions.

## Limitations

- Single annotator; no inter-annotator agreement study. A few gold answers are informal (`Around 115 million`, `First half of the year`), which the normalizer handles case by case.
- Chart interpretation has 17 examples and the reliability probe 25, so those comparisons are directional.
- Reliability paraphrases were model-generated by the model under test.
- Two companies contribute 42% of questions; per-company results are not balanced.
- `hallucinated_number` requires OCR text, which the default pipeline does not produce.
- Qwen was run once, greedily, in 4-bit; no seed or precision sweep.
- Unit equivalence is context-free: an explicit unit on one side is assumed to be the table's implicit unit, so `1.5 million` and `1.5 billion` both match a bare gold `1.5`. Task-level `answer_unit` metadata is the proper fix and is on the roadmap.
- Text answers are compared syntactically. Semantic equivalents such as `First Half` vs `First half of the year` or `JPMorganChase` vs `JPMorgan Chase` are labelled `partial_text_match`/`text_mismatch`, not credited; a reviewed answer-alias set would raise the true accuracy above 277/397.
- Source images are screenshots from public filings; visual quality varies.

## Project Structure

```text
finagent-eval/
├── finagent_eval/
│   ├── graders/          # normalize, anls, exact_match, numeric, grade()
│   ├── failure_modes.py  # 13-label failure classifier
│   ├── report.py         # scorecards, bootstrap CIs, comparison table
│   ├── pipeline/         # build_tasks, normalize_results, validate, enrich, reports, domain_adaptation, reliability
│   ├── adapters/         # BaseAdapter, dummy adapters, QwenVLAdapter, registry
│   ├── runner.py         # run an adapter with checkpointing + metadata sidecar
│   ├── paths.py          # ProjectPaths
│   └── cli.py            # finagent-eval
├── scripts/              # thin wrappers around the CLI
├── dashboard/app.py
├── notebooks/finagent_eval_qwen_vl_inference.ipynb
├── templates/scorecard.html.j2
├── tests/                # 96 tests incl. regression pins
├── data/track_a/         # raw annotations, tasks.json, images
├── results/              # track_a predictions (+ .meta.json), enriched, reliability, domain_adaptation
├── reports/              # generated scorecards + comparison
├── CHANGELOG.md · CONTRIBUTING.md · CITATION.cff · LICENSE · DATA_LICENSE.md
```

## License

Code is MIT (`LICENSE`). Annotations, model outputs and screenshots have separate usage notes in `DATA_LICENSE.md`.
