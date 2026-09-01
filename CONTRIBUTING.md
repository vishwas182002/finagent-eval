# Contributing

## Setup

```bash
git clone https://github.com/vishwas182002/finagent-eval.git
cd finagent-eval
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
make test
```

## Adding a model

1. Implement `finagent_eval.adapters.BaseAdapter` (see `adapters/qwen_vl.py`).
2. `finagent-eval run --adapter my_module:MyAdapter --out results/track_a/my_model.json --then-enrich`
3. Commit the prediction file, its `.meta.json` sidecar, the enriched file and
   the regenerated `reports/`. CI's reproducibility job checks they match.

If you produced predictions elsewhere, drop the JSON (one row per task, see
README "Add Your Own Model") into `results/track_a/` and run
`finagent-eval validate && finagent-eval enrich && finagent-eval report`.

## Changing the grader

Any change that moves a score must:

- bump `GRADER_VERSION` in `finagent_eval/graders/normalize.py`;
- add a before/after table to `CHANGELOG.md`;
- regenerate artifacts with `finagent-eval all`;
- update the pinned numbers in `tests/test_regression.py` and `README.md`.

`tests/test_regression.py::test_every_gold_answer_is_gradable_against_itself`
guards against normalization rules that would reject a gold answer.

## Adding questions

Edit `data/track_a/raw/combined_financial_vqa_dataset_fixed_types.json`, drop the
screenshot under `data/track_a/images/<TICKER>/`, then update the expected
counts in `finagent_eval/paths.py` and rerun the pipeline. Every baseline
result file must cover every task, so new questions need predictions for all
models before `validate` passes (or use `--no-strict` while iterating).

## Style

`ruff check .` must pass. Keep pipeline stages as pure functions of a
`ProjectPaths`; no module-level side effects, no `assert` for validation.
