#!/usr/bin/env python3
"""Export binary reward-model labels from leakage v2 StepFun outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="privacy_leakage_suite_v2_full.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-source", default="stepfun_leakage_v2_overall_score_ge_4")
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(Path(args.input)):
        if not row.get("ok"):
            continue
        rows.append(
            {
                "task_id": row.get("task_id"),
                "chain_id": row.get("chain_id"),
                "rollout_id": row.get("rollout_id"),
                "split": row.get("split"),
                "company": row.get("company"),
                "trace_path": row.get("trace_path"),
                "privacy_reward_label": int(bool(row.get("overall_leaked_for_reward"))),
                "overall_privacy_score_1_to_5": row.get("overall_privacy_score_1_to_5"),
                "overall_primary_source": row.get("overall_primary_source"),
                "intent_score_1_to_5": (row.get("intent_judge_output") or {}).get("intent_subquestion_score_1_to_5")
                if isinstance(row.get("intent_judge_output"), dict)
                else None,
                "answer_leaked": (row.get("answer_judge_output") or {}).get("answer_leaked")
                if isinstance(row.get("answer_judge_output"), dict)
                else None,
                "full_info_leaked": (row.get("claim_judge_output") or {}).get("full_info_leaked")
                if isinstance(row.get("claim_judge_output"), dict)
                else None,
                "label_source": args.label_source,
                "source_model": row.get("model"),
                "source_prompt_version": row.get("prompt_version"),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    positives = sum(row["privacy_reward_label"] for row in rows)
    print(f"wrote={out} rows={len(rows)} positives={positives} positive_rate={positives / len(rows) if rows else 0:.3f}")


if __name__ == "__main__":
    main()
