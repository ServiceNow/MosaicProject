#!/usr/bin/env python3
"""Pretty-print an OpenRouter-vs-StepFun judge smoke result."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


DEFAULT_SAMPLE_DIR = Path(
    "/home/toolkit/my_code/MosaicProject/"
    "runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525"
)
FIELDS = ["intent_leaked", "answer_leaked", "full_info_leaked", "overall_leaked_for_reward"]
STAGES = [
    "intent_adversary_output",
    "intent_judge_output",
    "answer_adversary_output",
    "answer_judge_output",
    "claim_adversary_output",
    "claim_judge_output",
]


def read_jsonl(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None or not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad.append({"line": line_number, "error": str(exc), "snippet": line[:240]})
    return rows, bad


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("task_id") or ""), str(row.get("chain_id") or ""), str(row.get("rollout_id") or ""))


def score(row: dict[str, Any]) -> int:
    return int(row.get("overall_privacy_score_1_to_5") or 0)


def bool_field(row: dict[str, Any], field: str) -> bool:
    if field == "intent_leaked":
        obj = row.get("intent_judge_output") if isinstance(row.get("intent_judge_output"), dict) else {}
        return int(obj.get("intent_subquestion_score_1_to_5") or 1) >= 4
    if field == "answer_leaked":
        obj = row.get("answer_judge_output") if isinstance(row.get("answer_judge_output"), dict) else {}
        return bool(obj.get("answer_leaked"))
    if field == "full_info_leaked":
        obj = row.get("claim_judge_output") if isinstance(row.get("claim_judge_output"), dict) else {}
        return bool(obj.get("full_info_leaked"))
    if field == "overall_leaked_for_reward":
        return bool(row.get("overall_leaked_for_reward"))
    raise KeyError(field)


def rankdata(values: list[int]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + 1 + end) / 2
        for original_index, _value in indexed[cursor:end]:
            ranks[original_index] = avg_rank
        cursor = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman(x: list[int], y: list[int]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    p1_yes = sum(a for a, _ in pairs) / n
    p2_yes = sum(b for _, b in pairs) / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def fmt_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pct(count: int, n: int) -> str:
    return "n/a" if not n else f"{100 * count / n:.1f}%"


def short_bool(value: bool) -> str:
    return "Y" if value else "N"


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n... [truncated]"


def raw_output(row: dict[str, Any], stage: str) -> str:
    obj = row.get(stage)
    if not isinstance(obj, dict):
        return ""
    meta = obj.get("_stepfun") if isinstance(obj.get("_stepfun"), dict) else {}
    raw = meta.get("raw_output")
    if raw:
        return str(raw)
    return json.dumps({k: v for k, v in obj.items() if k != "_stepfun"}, indent=2, ensure_ascii=False)


def web_queries(row: dict[str, Any]) -> list[str]:
    model_input = row.get("model_input") if isinstance(row.get("model_input"), dict) else {}
    queries = model_input.get("web_queries") if isinstance(model_input.get("web_queries"), list) else []
    out: list[str] = []
    for query in queries:
        if isinstance(query, dict):
            out.append(str(query.get("query") or ""))
        else:
            out.append(str(query))
    return [item for item in out if item]


def score_summary(ref_rows: dict[tuple[str, str, str], dict[str, Any]], cand_rows: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(set(ref_rows) & set(cand_rows))
    ref_scores = [score(ref_rows[k]) for k in keys]
    cand_scores = [score(cand_rows[k]) for k in keys]
    exact = sum(a == b for a, b in zip(ref_scores, cand_scores))
    within_one = sum(abs(a - b) <= 1 for a, b in zip(ref_scores, cand_scores))
    mae = sum(abs(a - b) for a, b in zip(ref_scores, cand_scores)) / len(keys) if keys else None
    return {
        "matched_rows": len(keys),
        "reference_only": len(set(ref_rows) - set(cand_rows)),
        "candidate_only": len(set(cand_rows) - set(ref_rows)),
        "exact_score_agreement": exact,
        "within_one_score_agreement": within_one,
        "score_mae": mae,
        "score_spearman": spearman(ref_scores, cand_scores) if keys else None,
        "reference_score_distribution": dict(Counter(ref_scores)),
        "candidate_score_distribution": dict(Counter(cand_scores)),
        "score_confusion": {f"{ref},{cand}": count for (ref, cand), count in Counter(zip(ref_scores, cand_scores)).items()},
    }


def binary_summaries(ref_rows: dict[tuple[str, str, str], dict[str, Any]], cand_rows: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(set(ref_rows) & set(cand_rows))
    out: dict[str, Any] = {}
    for field in FIELDS:
        pairs = [(bool_field(ref_rows[k], field), bool_field(cand_rows[k], field)) for k in keys]
        counts = Counter(pairs)
        agree = sum(a == b for a, b in pairs)
        out[field] = {
            "agreement_count": agree,
            "n": len(pairs),
            "agreement_rate": agree / len(pairs) if pairs else None,
            "cohen_kappa": cohen_kappa(pairs),
            "reference_true": sum(a for a, _ in pairs),
            "candidate_true": sum(b for _, b in pairs),
            "false_to_true": counts[(False, True)],
            "true_to_false": counts[(True, False)],
            "counts": {f"{ref},{cand}": count for (ref, cand), count in counts.items()},
        }
    return out


def disagreement_rows(
    ref_rows: dict[tuple[str, str, str], dict[str, Any]],
    cand_rows: dict[tuple[str, str, str], dict[str, Any]],
    manifest_rows: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in sorted(set(ref_rows) & set(cand_rows)):
        ref = ref_rows[k]
        cand = cand_rows[k]
        changed_fields = [field for field in FIELDS if bool_field(ref, field) != bool_field(cand, field)]
        if score(ref) == score(cand) and not changed_fields:
            continue
        manifest = manifest_rows.get(k, {})
        out.append(
            {
                "sample_id": manifest.get("sample_id"),
                "source_condition": manifest.get("source_condition") or ref.get("source_condition"),
                "task_id": k[0],
                "chain_id": k[1],
                "rollout_id": k[2],
                "reference_score": score(ref),
                "candidate_score": score(cand),
                "reference_reward": bool_field(ref, "overall_leaked_for_reward"),
                "candidate_reward": bool_field(cand, "overall_leaked_for_reward"),
                "changed_fields": changed_fields,
            }
        )
    return out


def render_summary(console: Console, summary: dict[str, Any], errors: list[dict[str, Any]], bad_files: dict[str, list[dict[str, Any]]]) -> None:
    table = Table(title="Judge Smoke Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    n = int(summary["matched_rows"] or 0)
    table.add_row("matched rows", str(n))
    table.add_row("reference only", str(summary["reference_only"]))
    table.add_row("candidate only", str(summary["candidate_only"]))
    table.add_row("candidate error rows", str(len(errors)))
    table.add_row("score exact agreement", f"{summary['exact_score_agreement']}/{n} ({pct(summary['exact_score_agreement'], n)})")
    table.add_row("score within 1", f"{summary['within_one_score_agreement']}/{n} ({pct(summary['within_one_score_agreement'], n)})")
    table.add_row("score MAE", fmt_float(summary["score_mae"]))
    table.add_row("score Spearman", fmt_float(summary["score_spearman"]))
    for label, bad in bad_files.items():
        if bad:
            table.add_row(f"malformed {label} lines", str(len(bad)))
    console.print(table)


def render_binary(console: Console, summaries: dict[str, Any]) -> None:
    table = Table(title="Binary Label Agreement", box=box.SIMPLE)
    table.add_column("Label")
    table.add_column("Agree", justify="right")
    table.add_column("Kappa", justify="right")
    table.add_column("StepFun True", justify="right")
    table.add_column("Candidate True", justify="right")
    table.add_column("False->True", justify="right")
    table.add_column("True->False", justify="right")
    labels = {
        "intent_leaked": "Intent Leakage",
        "answer_leaked": "Answer Leakage",
        "full_info_leaked": "Full-Information Leakage",
        "overall_leaked_for_reward": "Privacy Leakage",
    }
    for field, item in summaries.items():
        n = int(item["n"] or 0)
        agree = int(item["agreement_count"] or 0)
        table.add_row(
            labels.get(field, field),
            f"{agree}/{n} ({pct(agree, n)})",
            fmt_float(item["cohen_kappa"]),
            str(item["reference_true"]),
            str(item["candidate_true"]),
            str(item["false_to_true"]),
            str(item["true_to_false"]),
        )
    console.print(table)


def render_confusion(console: Console, confusion: dict[str, int]) -> None:
    table = Table(title="Score Confusion Matrix (Rows=StepFun, Columns=Candidate)", box=box.SIMPLE)
    table.add_column("StepFun \\ Candidate")
    for score_value in range(1, 6):
        table.add_column(str(score_value), justify="right")
    for ref_score in range(1, 6):
        table.add_row(str(ref_score), *[str(confusion.get(f"{ref_score},{cand_score}", 0)) for cand_score in range(1, 6)])
    console.print(table)


def render_rows(
    console: Console,
    ref_rows: dict[tuple[str, str, str], dict[str, Any]],
    cand_rows: dict[tuple[str, str, str], dict[str, Any]],
    manifest_rows: dict[tuple[str, str, str], dict[str, Any]],
    max_rows: int,
) -> None:
    keys = sorted(set(ref_rows) & set(cand_rows))
    shown_keys = keys if max_rows <= 0 else keys[:max_rows]
    title = "Smoke Rows"
    if len(shown_keys) != len(keys):
        title = f"Smoke Rows (showing {len(shown_keys)}/{len(keys)})"
    table = Table(title=title, box=box.SIMPLE)
    table.add_column("Sample")
    table.add_column("Condition")
    table.add_column("Task")
    table.add_column("Ref")
    table.add_column("Cand")
    table.add_column("Leak Ref/Cand")
    table.add_column("Intent")
    table.add_column("Answer")
    table.add_column("Full Info")
    for k in shown_keys:
        ref = ref_rows[k]
        cand = cand_rows[k]
        manifest = manifest_rows.get(k, {})
        table.add_row(
            str(manifest.get("sample_id") or ""),
            str(manifest.get("source_condition") or ref.get("source_condition") or "")[:32],
            f"{k[0]}/{k[1]}",
            str(score(ref)),
            str(score(cand)),
            f"{short_bool(bool_field(ref, 'overall_leaked_for_reward'))}/{short_bool(bool_field(cand, 'overall_leaked_for_reward'))}",
            f"{short_bool(bool_field(ref, 'intent_leaked'))}/{short_bool(bool_field(cand, 'intent_leaked'))}",
            f"{short_bool(bool_field(ref, 'answer_leaked'))}/{short_bool(bool_field(cand, 'answer_leaked'))}",
            f"{short_bool(bool_field(ref, 'full_info_leaked'))}/{short_bool(bool_field(cand, 'full_info_leaked'))}",
        )
    console.print(table)


def render_details(
    console: Console,
    ref_rows: dict[tuple[str, str, str], dict[str, Any]],
    cand_rows: dict[tuple[str, str, str], dict[str, Any]],
    manifest_rows: dict[tuple[str, str, str], dict[str, Any]],
    disagreements: list[dict[str, Any]],
    max_detail: int,
    raw_chars: int,
) -> None:
    keys: list[tuple[str, str, str]] = []
    for item in disagreements:
        keys.append((str(item["task_id"]), str(item["chain_id"]), str(item["rollout_id"])))
    if not keys:
        keys = sorted(set(ref_rows) & set(cand_rows))[:max_detail]
    else:
        keys = keys[:max_detail]
    for k in keys:
        ref = ref_rows[k]
        cand = cand_rows[k]
        manifest = manifest_rows.get(k, {})
        header = (
            f"{manifest.get('sample_id') or ''} {manifest.get('source_condition') or ref.get('source_condition') or ''}\n"
            f"{k[0]} / {k[1]} / {k[2]}\n"
            f"score StepFun={score(ref)} candidate={score(cand)}; "
            f"privacy StepFun={short_bool(bool_field(ref, 'overall_leaked_for_reward'))} "
            f"candidate={short_bool(bool_field(cand, 'overall_leaked_for_reward'))}"
        )
        query_text = "\n".join(f"{idx + 1}. {query}" for idx, query in enumerate(web_queries(ref))) or "(no visible web queries)"
        chunks: list[Any] = [Text(header), Text("\nVisible web queries:\n" + query_text)]
        for stage in STAGES:
            raw = raw_output(cand, stage)
            if not raw:
                continue
            chunks.append(Text(f"\nCandidate {stage}:"))
            chunks.append(Syntax(truncate(raw, raw_chars), "json", word_wrap=True))
        console.print(Panel(Group(*chunks), title="Disagreement Detail" if disagreements else "Smoke Row Detail", border_style="cyan"))


def render_errors(console: Console, errors: list[dict[str, Any]], max_errors: int) -> None:
    if not errors:
        return
    table = Table(title="Candidate Error Rows", box=box.SIMPLE)
    table.add_column("Task")
    table.add_column("Chain")
    table.add_column("Rollout")
    table.add_column("Error Summary")
    for row in errors[:max_errors]:
        error_fields = []
        for key_name, value in row.items():
            if key_name.endswith("_error") and value:
                error_fields.append(f"{key_name}={truncate(json.dumps(value, ensure_ascii=False), 240)}")
        table.add_row(
            str(row.get("task_id") or ""),
            str(row.get("chain_id") or ""),
            str(row.get("rollout_id") or ""),
            "\n".join(error_fields) or truncate(json.dumps(row, ensure_ascii=False), 300),
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--output-disagreements", type=Path, default=None)
    parser.add_argument("--max-detail", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=40, help="Maximum row-level entries to print; use 0 for all.")
    parser.add_argument("--raw-chars", type=int, default=1800)
    args = parser.parse_args()

    reference = args.reference or args.sample_dir / "stepfun_reference_sample.jsonl"
    manifest_path = args.manifest or args.sample_dir / "sample_manifest.jsonl"
    ref, ref_bad = read_jsonl(reference)
    cand, cand_bad = read_jsonl(args.candidate)
    errors, err_bad = read_jsonl(args.errors)
    manifest, manifest_bad = read_jsonl(manifest_path)

    ref_rows = {key(row): row for row in ref if row.get("ok")}
    cand_rows = {key(row): row for row in cand if row.get("ok")}
    manifest_rows = {key(row): row for row in manifest}
    summary = score_summary(ref_rows, cand_rows)
    binaries = binary_summaries(ref_rows, cand_rows)
    disagreements = disagreement_rows(ref_rows, cand_rows, manifest_rows)

    output = {
        **summary,
        "binary_fields": binaries,
        "n_disagreement_rows": len(disagreements),
        "candidate_errors": len(errors),
        "bad_jsonl_lines": {
            "reference": ref_bad,
            "candidate": cand_bad,
            "errors": err_bad,
            "manifest": manifest_bad,
        },
    }
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_disagreements:
        args.output_disagreements.parent.mkdir(parents=True, exist_ok=True)
        with args.output_disagreements.open("w", encoding="utf-8") as handle:
            for item in disagreements:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    console = Console()
    render_summary(console, summary, errors, {"reference": ref_bad, "candidate": cand_bad, "errors": err_bad, "manifest": manifest_bad})
    render_binary(console, binaries)
    render_confusion(console, summary["score_confusion"])
    render_rows(console, ref_rows, cand_rows, manifest_rows, args.max_rows)
    render_errors(console, errors, args.max_detail)
    render_details(console, ref_rows, cand_rows, manifest_rows, disagreements, args.max_detail, args.raw_chars)
    if args.output_summary:
        console.print(f"[green]wrote summary[/green] {args.output_summary}")
    if args.output_disagreements:
        console.print(f"[green]wrote disagreements[/green] {args.output_disagreements}")


if __name__ == "__main__":
    main()
