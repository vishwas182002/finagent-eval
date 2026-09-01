# Changelog

All notable changes to FinAgent-Eval. Grader changes that move published numbers
are always listed here with before/after figures.

## 0.2.0 - grader v2.0

### Grader revision (changes published numbers)

The v1 grader under-credited the headline model. `normalize_financial_number`
did not recognise unit words, so a prediction such as `$85,269 million` for a
gold answer `85,269` (from a table headed "in millions") returned `None`, fell
back to plain-text comparison, scored ANLS 0 and was filed under
`manual_diagnosis_required`. Curly vs. straight apostrophes
(`STOCKHOLDERS’ EQUITY` vs `STOCKHOLDERS' EQUITY`) and a leading article
(`The Goldman Sachs Group, Inc.`) were also counted as wrong. An audit of
Qwen2.5-VL's 151 v1 "failures" found 35 that were correct answers.

Grader v2.0 therefore:

- parses unit words and suffixes (`million`, `bn`, `1.5M`, `USD`, `percent`),
  answer prefixes (`Answer:`, `approximately`), unicode punctuation and
  trailing periods, and treats an explicit unit on exactly one side as the
  table's implicit unit (`financial_numbers_equivalent`);
- makes ANLS numeric-aware: equivalent financial numbers score 1.0, everything
  else uses the standard DocVQA Levenshtein rule (`anls_score(...,
  numeric_aware=False)` gives the v1 text metric, reported as `raw_anls`);
- restricts the 2% numeric tolerance to `numerical_reasoning` and
  `chart_interpretation`. On extractive/layout questions the number is printed
  verbatim, and v1 was accepting misread digits (`37,586` for `37,855`,
  `1,086,031` for `1,095,835`) as correct. These are now `near_miss_numeric`
  failures;
- replaces the mostly-unused failure taxonomy (94 of Qwen's 151 v1 failures were
  `manual_diagnosis_required`) with `sign_mismatch`, `unit_mismatch`,
  `near_miss_numeric`, `type_mismatch`, `partial_text_match`, `text_mismatch`
  and a `wrong_year` rule for year-valued gold answers. Zero rows now need
  manual diagnosis;
- re-grades the paraphrase reliability probe with the same `grade()` function
  used for the leaderboard (the notebook had its own exact-only copy).

| Model | v1 ANLS | v2 ANLS | v1 EM | v2 EM | v1 project-correct | v2 project-correct |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | 0.6368 | **0.7378** | 0.5945 | **0.6877** | 246/397 | **277/397** |
| LayoutLMv3 | 0.1472 | 0.1533 | 0.1335 | 0.1335 | 53/397 | 53/397 |
| Pix2Struct | 0.1284 | 0.1343 | 0.1134 | 0.1134 | 49/397 | 45/397 |
| OCR+RoBERTa | 0.1108 | 0.1132 | 0.0856 | 0.0856 | 34/397 | 34/397 |
| Donut | 0.1021 | 0.1061 | 0.0756 | 0.0756 | 34/397 | 30/397 |

Qwen's numerical-reasoning result moves from 4/65 to 10/65 correct (six
questions in that category had gold answers like `32.3` for a `$32.3 billion`
read); it remains the weakest category by a wide margin (ANLS 0.1938 vs 0.81+
elsewhere). Reliability probe: correctness stability 20/25 -> 18/25,
original-question accuracy 13/25 -> 15/25; consistency and all-variants-correct
unchanged (11/25, 9/25). Pix2Struct LoRA held-out (project grader): ANLS
0.0973 -> 0.1348 becomes 0.1000 -> 0.1348.

### Packaging and tooling

- `pyproject.toml`; `pip install -e .` gives a `finagent-eval` CLI with
  `build-tasks`, `normalize-results`, `validate`, `enrich`, `report`,
  `domain-adaptation`, `regrade-reliability`, `all`, `run` and `grade`.
  The old `scripts/*.py` entry points remain as thin wrappers.
- Pipeline stages are functions in `finagent_eval/pipeline/` that take an
  explicit project root, so they can be tested on synthetic data.
- `finagent_eval.adapters.BaseAdapter` + `finagent_eval.runner.run_adapter`:
  implement `predict(image_path, question)` and the runner writes the prediction
  file with checkpoint/resume, per-row latency and a `.meta.json` sidecar
  (prompt, decoding settings, quantization, dataset hash, package/grader
  version, git sha). `QwenVLAdapter` reproduces the notebook run; a
  reconstructed sidecar for the original Colab run ships in
  `results/track_a/qwen25vl_7b_results.meta.json`.
- Scorecards gain `overall_project_correct` (+CI), `overall_raw_anls`,
  `mean_latency_ms`, `grader_version` and `run_metadata`.
- Tests: 12 -> 96, including a synthetic end-to-end pipeline test, runner and
  CLI tests, bootstrap-CI tests, and regression tests that pin every published
  leaderboard number and check the committed enriched files match a fresh
  recomputation.
- CI: ruff lint, tests on Python 3.10/3.11/3.12, a dashboard smoke test, and a
  reproducibility job that regenerates every artifact and fails on any diff.
  HTML reports no longer embed a timestamp so they are byte-stable.
- Dashboard: Streamlit `use_container_width` deprecation removed, project-correct
  column, grader caption, new failure-label colours, real adapter workflow in
  "Run Your Own Model".
- Notebook: clones this repository and imports the package grader instead of a
  copy-pasted normalizer.
- Pinned `requirements.txt` (pandas 2.3.x on Python 3.10, 3.0.x on 3.11+), `CITATION.cff`, `CONTRIBUTING.md`, `Makefile`.

### Post-review fixes

- `finagent-eval grade` requires `--category` so CLI grading agrees with the leaderboard's category-aware tolerance.
- `finagent-eval validate` honours `--no-strict` (strict is the default and fails on dataset-shape errors).
- `QwenVLAdapter` reproduces the notebook's OOM recovery (retry at 32 new tokens) and accepts a `revision` pin; the notebook installs released transformers instead of git `main` and exposes `MODEL_REVISION` / `REPO_REF` pins.
- README documents two known grader limitations: context-free unit equivalence and syntactic-only text matching.

## 0.1.0

Initial release: 397-question Track A dataset, four 2024 document-AI baselines,
Qwen2.5-VL-7B-Instruct Colab run, ANLS / exact-match / numeric-tolerance
grading, failure labels, bootstrap CIs, reliability mini-probe, Pix2Struct LoRA
domain-adaptation analysis, Streamlit dashboard.
