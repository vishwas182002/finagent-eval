.PHONY: install test test-fast lint pipeline validate dashboard clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests -v

test-fast:
	python -m pytest tests -q -m "not slow"

lint:
	ruff check .

pipeline:
	finagent-eval all

validate:
	finagent-eval validate

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build *.egg-info
