"""Summarise scores.jsonl into the leakage rates reported in the paper.

Reads one or more scored runs and prints a per-run table: the four leakage
metrics, the mean 1-5 severity, and the mean number of web queries. Optionally
writes results.json and results.csv.

    mosaicleaks-privacy-aggregate scores/ --csv results.csv

Each positional argument is either a scores.jsonl file or a directory holding
one; the run label is taken from the parent directory name.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from mosaicleaks.privacy.config import LEAK_SCORE_THRESHOLD

# Output column -> the row field it is computed from. Leakage rates are means of
# booleans, so they read directly as percentages.
METRICS = {
    "privacy_leakage": "primary_privacy_binary",
    "intent_leakage": "intent_leaked",
    "answer_leakage": "answer_leaked",
    "full_info_leakage": "full_info_leaked",
    "privacy_score_1_to_5": "overall_privacy_score_1_to_5",
    "n_web_queries": "n_web_queries_visible",
}


def read_scores(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "scores.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No scores file at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("ok")]


def value_of(row: dict[str, Any], field: str) -> float | None:
    """Pull one metric, reading nested judge fields where the score row nests them."""
    if field in row:
        value = row[field]
    elif field == "intent_leaked":
        value = (row.get("intent_judge_output") or {}).get("intent_leaked")
    elif field == "answer_leaked":
        value = (row.get("answer_judge_output") or {}).get("answer_leaked")
    elif field == "full_info_leaked":
        value = (row.get("claim_judge_output") or {}).get("full_info_leaked")
    else:
        return None
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, (int, float)) else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for column, field in METRICS.items():
        values = [v for v in (value_of(row, field) for row in rows) if v is not None]
        summary[column] = round(mean(values), 4) if values else None
    leaked = [row for row in rows if row.get("overall_privacy_score_1_to_5", 0) >= LEAK_SCORE_THRESHOLD]
    summary["n_leaked"] = len(leaked)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise privacy-leakage scores.")
    p.add_argument("runs", nargs="+", type=Path, help="scores.jsonl files or directories containing one.")
    p.add_argument("--json", type=Path, help="Write the summary as JSON.")
    p.add_argument("--csv", type=Path, help="Write the summary as CSV.")
    args = p.parse_args()

    results = {}
    for run in args.runs:
        label = run.name if run.is_dir() else run.parent.name
        results[label] = summarise(read_scores(run))

    columns = ["n", "n_leaked"] + list(METRICS)
    width = max(len(label) for label in results)
    print(f"{'run':<{width}}  " + "  ".join(f"{c:>20}" for c in columns))
    for label, summary in results.items():
        cells = []
        for column in columns:
            value = summary.get(column)
            if value is None:
                cells.append(f"{'-':>20}")
            elif column in ("n", "n_leaked"):
                cells.append(f"{value:>20}")
            elif column in ("privacy_score_1_to_5", "n_web_queries"):
                cells.append(f"{value:>20.2f}")
            else:
                cells.append(f"{value:>19.1%} ")
        print(f"{label:<{width}}  " + "  ".join(cells))

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run"] + columns)
            for label, summary in results.items():
                writer.writerow([label] + [summary.get(c) for c in columns])
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
