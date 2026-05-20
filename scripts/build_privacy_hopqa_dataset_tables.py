#!/usr/bin/env python3
"""Build LaTeX dataset summary tables for Privacy HopQA paper assets."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data_files/privacy_hopqa_final_dataset"
DEFAULT_DATASET_BASENAME = "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review"
DEFAULT_PRIVATE_QA = Path(
    "/home/toolkit/nice_code/drbench/making_dataset_2/outputs/secret_inventory_all_filtered.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_assets/privacy_hopqa_dataset_tables"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_float(value: float) -> str:
    return f"{value:.2f}"


def mean_min_max(summary: dict[str, Any]) -> str:
    return f"{fmt_float(float(summary['mean']))} [{summary['min']}, {summary['max']}]"


def doc_key(hop: dict[str, Any]) -> str:
    return str(hop.get("doc_id") or hop.get("doc_ref") or "")


def summarize_private_qa(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    docs: set[str] = set()
    question_strings: set[str] = set()
    question_counts: list[int] = []
    secret_types: Counter[str] = Counter()
    tasks: set[str] = set()

    for row in rows:
        doc_id = str(row.get("doc_id") or row.get("chunk_id") or "")
        docs.add(doc_id)
        parts = doc_id.split("/")
        if len(parts) > 1 and parts[1].startswith("DR"):
            tasks.add(parts[1])

        secrets = row.get("secrets") or []
        question_counts.append(len(secrets))
        for secret in secrets:
            question = secret.get("question")
            if isinstance(question, str):
                question_strings.add(question)
            secret_types[str(secret.get("secret_type") or "unknown")] += 1

    return {
        "source": str(path),
        "local_documents": len(docs),
        "private_qa_pairs": sum(question_counts),
        "unique_private_questions": len(question_strings),
        "private_questions_per_doc": {
            "mean": statistics.mean(question_counts) if question_counts else 0.0,
            "median": statistics.median(question_counts) if question_counts else 0.0,
            "min": min(question_counts) if question_counts else 0,
            "max": max(question_counts) if question_counts else 0,
        },
        "secret_types": dict(sorted(secret_types.items())),
        "n_secret_types": len(secret_types),
        "task_count": len(tasks),
    }


def summarize_dataset_split(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    local_doc_counts: Counter[str] = Counter()
    web_doc_counts: Counter[str] = Counter()
    hop_counts: list[int] = []
    local_hops_per_question: list[int] = []
    web_hops_per_question: list[int] = []
    local_hops = 0
    web_hops = 0
    companies: Counter[str] = Counter()
    local_to_web_chains = 0
    web_to_local_chains = 0

    for row in rows:
        hops = row.get("hops") or []
        hop_counts.append(len(hops))
        row_local_hops = 0
        row_web_hops = 0
        companies[str(row.get("company") or "unknown")] += 1
        local_to_web_chains += int(bool(row.get("has_local_to_web_transition")))
        web_to_local_chains += int(bool(row.get("has_web_to_local_transition")))

        for hop in hops:
            hop_type = str(hop.get("hop_type") or "").upper()
            key = doc_key(hop)
            if hop_type == "L":
                local_hops += 1
                row_local_hops += 1
                if key:
                    local_doc_counts[key] += 1
            elif hop_type == "W":
                web_hops += 1
                row_web_hops += 1
                if key:
                    web_doc_counts[key] += 1
        local_hops_per_question.append(row_local_hops)
        web_hops_per_question.append(row_web_hops)

    local_doc_values = list(local_doc_counts.values())
    return {
        "source": str(path),
        "multi_hop_questions": len(rows),
        "hop_questions": sum(hop_counts),
        "private_local_questions": local_hops,
        "public_web_questions": web_hops,
        "unique_local_documents": len(local_doc_counts),
        "unique_web_documents": len(web_doc_counts),
        "private_questions_per_local_doc": {
            "mean": statistics.mean(local_doc_values) if local_doc_values else 0.0,
            "median": statistics.median(local_doc_values) if local_doc_values else 0.0,
            "min": min(local_doc_values) if local_doc_values else 0,
            "max": max(local_doc_values) if local_doc_values else 0,
        },
        "hops_per_question": {
            "mean": statistics.mean(hop_counts) if hop_counts else 0.0,
            "median": statistics.median(hop_counts) if hop_counts else 0.0,
            "min": min(hop_counts) if hop_counts else 0,
            "max": max(hop_counts) if hop_counts else 0,
        },
        "local_hops_per_question": {
            "mean": statistics.mean(local_hops_per_question) if local_hops_per_question else 0.0,
            "median": statistics.median(local_hops_per_question) if local_hops_per_question else 0.0,
            "min": min(local_hops_per_question) if local_hops_per_question else 0,
            "max": max(local_hops_per_question) if local_hops_per_question else 0,
        },
        "web_hops_per_question": {
            "mean": statistics.mean(web_hops_per_question) if web_hops_per_question else 0.0,
            "median": statistics.median(web_hops_per_question) if web_hops_per_question else 0.0,
            "min": min(web_hops_per_question) if web_hops_per_question else 0,
            "max": max(web_hops_per_question) if web_hops_per_question else 0,
        },
        "local_to_web_chains": local_to_web_chains,
        "web_to_local_chains": web_to_local_chains,
        "companies": dict(sorted(companies.items())),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def compact_summary_table(private_qa: dict[str, Any], full: dict[str, Any]) -> str:
    private_q_doc = private_qa["private_questions_per_doc"]
    local_hops_q = full["local_hops_per_question"]
    web_hops_q = full["web_hops_per_question"]
    hops_q = full["hops_per_question"]
    rows = [
        ("Private QA Set", "Local documents with private QA", fmt_int(private_qa["local_documents"])),
        ("Private QA Set", "Private question--answer pairs", fmt_int(private_qa["private_qa_pairs"])),
        ("Private QA Set", "Private QAs per local document", f"{fmt_float(private_q_doc['mean'])} mean"),
        ("Final Privacy HopQA", "Multi-hop questions", fmt_int(full["multi_hop_questions"])),
        ("Final Privacy HopQA", "Hop-level questions", fmt_int(full["hop_questions"])),
        ("Final Privacy HopQA", "Local hops per question, mean [min, max]", mean_min_max(local_hops_q)),
        ("Final Privacy HopQA", "Web hops per question, mean [min, max]", mean_min_max(web_hops_q)),
        ("Final Privacy HopQA", "Total hops per question, mean [min, max]", mean_min_max(hops_q)),
        ("Final Privacy HopQA", "Unique local supporting documents", fmt_int(full["unique_local_documents"])),
        ("Final Privacy HopQA", "Unique web supporting documents", fmt_int(full["unique_web_documents"])),
    ]
    body: list[str] = []
    previous_group = ""
    for group, stat, value in rows:
        if previous_group and group != previous_group:
            body.append(r"\midrule")
        group_text = group if group != previous_group else ""
        body.append(
            f"{latex_escape(group_text)} & {latex_escape(stat)} & {latex_escape(str(value))} \\\\"
        )
        previous_group = group

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{p{0.25\linewidth}p{0.47\linewidth}p{0.20\linewidth}}",
            r"\toprule",
            r"\textbf{Dataset component} & \textbf{Statistic} & \textbf{Value} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Privacy HopQA dataset summary. Bracketed values are [min, max] over multi-hop questions.}",
            r"\label{tab:privacy_hopqa_dataset_summary}",
            r"\end{table}",
            "",
        ]
    )


def split_summary_table(split_stats: dict[str, dict[str, Any]]) -> str:
    rows: list[str] = []
    for split in ("Full", "Train", "Val", "Test"):
        stat = split_stats[split]
        local_q_doc = stat["private_questions_per_local_doc"]
        rows.append(
            " & ".join(
                [
                    split,
                    fmt_int(stat["multi_hop_questions"]),
                    fmt_int(stat["hop_questions"]),
                    fmt_int(stat["private_local_questions"]),
                    fmt_int(stat["public_web_questions"]),
                    fmt_int(stat["unique_local_documents"]),
                    fmt_float(local_q_doc["mean"]),
                    fmt_float(stat["hops_per_question"]["mean"]),
                ]
            )
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"\textbf{Split} & \textbf{Questions} & \textbf{Hops} & \textbf{Local} & \textbf{Web} & \textbf{Local docs} & \textbf{Local q/doc} & \textbf{Hops/q} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Final Privacy HopQA split statistics. Questions are multi-hop chain questions; Local and Web are hop-level subquestions. The held-out test split contains MediConn Solutions only.}",
            r"\label{tab:privacy_hopqa_split_summary}",
            r"\end{table*}",
            "",
        ]
    )


def secret_type_table(private_qa: dict[str, Any]) -> str:
    rows = [
        f"{latex_escape(secret_type)} & {fmt_int(count)} \\\\"
        for secret_type, count in sorted(private_qa["secret_types"].items())
    ]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"\textbf{Secret type} & \textbf{Private QA pairs} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Private QA Set composition by secret type.}",
            r"\label{tab:privacy_hopqa_secret_type_counts}",
            r"\end{table}",
            "",
        ]
    )


def build(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    basename = args.dataset_basename
    split_paths = {
        "Full": dataset_dir / f"{basename}.jsonl",
        "Train": dataset_dir / f"{basename}_train.jsonl",
        "Val": dataset_dir / f"{basename}_val.jsonl",
        "Test": dataset_dir / f"{basename}_test.jsonl",
    }

    private_qa = summarize_private_qa(Path(args.private_qa))
    split_stats = {split: summarize_dataset_split(path) for split, path in split_paths.items()}
    stats = {
        "private_qa": private_qa,
        "final_privacy_hopqa": split_stats,
    }

    out_dir = Path(args.output_dir)
    write_text(out_dir / "dataset_summary_compact.tex", compact_summary_table(private_qa, split_stats["Full"]))
    write_text(out_dir / "dataset_split_summary.tex", split_summary_table(split_stats))
    write_text(out_dir / "private_qa_secret_type_counts.tex", secret_type_table(private_qa))
    write_text(
        out_dir / "all_dataset_tables.tex",
        "\n".join(
            [
                r"% Requires \usepackage{booktabs}.",
                r"\input{paper_assets/privacy_hopqa_dataset_tables/dataset_summary_compact}",
                r"\input{paper_assets/privacy_hopqa_dataset_tables/dataset_split_summary}",
                r"\input{paper_assets/privacy_hopqa_dataset_tables/private_qa_secret_type_counts}",
                "",
            ]
        ),
    )
    write_text(out_dir / "dataset_stats.json", json.dumps(stats, indent=2, sort_keys=True))
    write_text(
        out_dir / "README.md",
        "\n".join(
            [
                "# Privacy HopQA Dataset Tables",
                "",
                "LaTeX snippets generated by `scripts/build_privacy_hopqa_dataset_tables.py`.",
                "",
                "Primary paper-body candidate:",
                "",
                "- `dataset_summary_compact.tex`",
                "",
                "Useful appendix/backmatter variants:",
                "",
                "- `dataset_split_summary.tex`",
                "- `private_qa_secret_type_counts.tex`",
                "",
                "The tables require `\\usepackage{booktabs}`.",
                "",
            ]
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--dataset-basename", default=DEFAULT_DATASET_BASENAME)
    parser.add_argument("--private-qa", default=str(DEFAULT_PRIVATE_QA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
