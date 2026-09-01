"""Command-line interface: ``finagent-eval <command>`` (or ``python -m finagent_eval``)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .graders import GRADER_VERSION, grade
from .paths import EXPECTED_CATEGORY_COUNTS, ProjectPaths


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=None, help="project root (default: auto-detect)")
    parser.add_argument("--no-strict", action="store_true", help="skip the 397-task dataset shape checks")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finagent-eval", description="FinAgent-Eval pipeline")
    parser.add_argument("--version", action="version", version=f"finagent-eval {__version__} (grader v{GRADER_VERSION})")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("build-tasks", "convert raw annotations into data/track_a/tasks.json"),
        ("normalize-results", "normalize the 2024 baseline files into results/track_a"),
        ("validate", "validate tasks.json and every prediction file"),
        ("enrich", "recompute grades + failure labels into results/track_a/enriched"),
        ("report", "write JSON/HTML scorecards and the comparison table"),
        ("domain-adaptation", "Pix2Struct zero-shot vs LoRA held-out analysis"),
        ("regrade-reliability", "re-grade the paraphrase probe with the project grader"),
        ("all", "run every stage in order"),
    ]:
        p = sub.add_parser(name, help=help_text)
        _add_root(p)
        if name == "enrich":
            p.add_argument("files", nargs="*", type=Path, help="specific prediction files (default: all)")

    run = sub.add_parser("run", help="run a model adapter over the task set")
    _add_root(run)
    run.add_argument("--adapter", required=True, help="registry name or module:ClassName")
    run.add_argument("--out", type=Path, required=True, help="prediction file to write, e.g. results/track_a/my_model.json")
    run.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    run.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    run.add_argument("--adapter-arg", action="append", default=[], metavar="KEY=VALUE", help="constructor kwarg for the adapter (JSON values allowed)")
    run.add_argument("--then-enrich", action="store_true", help="enrich + report after the run")

    g = sub.add_parser("grade", help="grade a single prediction against a gold answer with leaderboard semantics")
    g.add_argument("prediction")
    g.add_argument("gold")
    g.add_argument(
        "--category",
        required=True,
        choices=sorted(EXPECTED_CATEGORY_COUNTS),
        help="task category; numeric tolerance only counts for numerical_reasoning and chart_interpretation",
    )

    return parser


def _parse_kwargs(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--adapter-arg expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s" if not args.verbose else "%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.command == "grade":
        result = {"category": args.category, **grade(args.prediction, args.gold, category=args.category).to_dict()}
        print(json.dumps(result, indent=2))
        return 0

    paths = ProjectPaths.from_root(args.root)
    strict = not args.no_strict
    from . import pipeline  # imported lazily so `grade` works without jinja2

    if args.command == "build-tasks":
        pipeline.build_tasks(paths, strict=strict)
    elif args.command == "normalize-results":
        pipeline.normalize_baseline_results(paths, strict=strict)
    elif args.command == "validate":
        from .pipeline.validate import ValidationError

        try:
            report = pipeline.validate_data(paths, strict=strict)
        except ValidationError as exc:
            print(f"\n{exc}")
            return 1
        if not report.ok:
            print(f"\n{len(report.errors)} error(s) found!")
            return 1
    elif args.command == "enrich":
        pipeline.enrich_results(paths, only=args.files or None)
    elif args.command == "report":
        pipeline.generate_reports(paths)
    elif args.command == "domain-adaptation":
        pipeline.analyze_domain_adaptation(paths, strict=strict)
    elif args.command == "regrade-reliability":
        pipeline.regrade_reliability_probe(paths)
    elif args.command == "all":
        pipeline.run_all(paths, strict=strict)
    elif args.command == "run":
        from .adapters import resolve_adapter
        from .runner import run_adapter

        adapter = resolve_adapter(args.adapter)(**_parse_kwargs(args.adapter_arg))
        run_adapter(adapter, paths, args.out, limit=args.limit, resume=not args.no_resume)
        if args.then_enrich:
            pipeline.enrich_results(paths, only=[args.out])
            pipeline.generate_reports(paths)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
