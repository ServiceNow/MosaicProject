#!/usr/bin/env python3
"""Build paper-viewer assets for the OpenRouter-vs-StepFun judge validation."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "runtime_analysis" / "privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525"
VIEWER = ROOT / "runtime_analysis" / "privacy_hopqa_q30_paper_viewer_s100"
REFERENCE = SAMPLE_DIR / "stepfun_reference_sample_full300_20260525.jsonl"
CANDIDATE = SAMPLE_DIR / "openrouter_smoke_anthropic_claude-opus-4.7_full300_combined_20260525.jsonl"

LABELS = {
    "privacy": "Privacy Leakage",
    "intent": "Intent Leakage",
    "answer": "Answer Leakage",
    "full_info": "Full-Information Leakage",
}
COLORS = {
    "blue": "#3c6e91",
    "green": "#4f7f5f",
    "orange": "#c07a3a",
    "red": "#b85b5b",
    "gray": "#6b7772",
    "grid": "#dfe4df",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("task_id")), str(row.get("chain_id")), str(row.get("rollout_id")))


def score(row: dict[str, Any]) -> int:
    return int(row.get("overall_privacy_score_1_to_5") or 0)


def binary(row: dict[str, Any], field: str) -> bool:
    if field == "privacy":
        return bool(row.get("overall_leaked_for_reward"))
    if field == "intent":
        judge = row.get("intent_judge_output") if isinstance(row.get("intent_judge_output"), dict) else {}
        return int(judge.get("intent_subquestion_score_1_to_5") or 1) >= 4
    if field == "answer":
        judge = row.get("answer_judge_output") if isinstance(row.get("answer_judge_output"), dict) else {}
        return bool(judge.get("answer_leaked"))
    if field == "full_info":
        judge = row.get("claim_judge_output") if isinstance(row.get("claim_judge_output"), dict) else {}
        return bool(judge.get("full_info_leaked"))
    raise KeyError(field)


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def ranks(values: list[int]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        avg = (cursor + 1 + end) / 2
        for idx in order[cursor:end]:
            out[idx] = avg
        cursor = end
    return out


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    p1 = sum(a for a, _ in pairs) / n
    p2 = sum(b for _, b in pairs) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def exact_mcnemar_pvalue(false_to_true: int, true_to_false: int) -> float | None:
    n = false_to_true + true_to_false
    if n == 0:
        return None
    observed = min(false_to_true, true_to_false)
    cdf = sum(math.comb(n, k) for k in range(observed + 1)) / (2**n)
    return min(1.0, 2 * cdf)


def weighted_kappa(score_pairs: list[tuple[int, int]], weight: str = "quadratic") -> float:
    labels = [1, 2, 3, 4, 5]
    n = len(score_pairs)
    obs = np.zeros((5, 5), dtype=float)
    for a, b in score_pairs:
        obs[a - 1, b - 1] += 1
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col) / n
    weights = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            dist = abs(labels[i] - labels[j])
            weights[i, j] = (dist / 4) ** 2 if weight == "quadratic" else dist / 4
    observed_disagreement = float((weights * obs).sum() / n)
    expected_disagreement = float((weights * exp).sum() / n)
    if expected_disagreement == 0:
        return 1.0
    return 1 - observed_disagreement / expected_disagreement


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def fmt_float(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def pvalue_text(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 1e-4:
        return "$<10^{-4}$"
    return f"{value:.4f}"


def tp_fp_fn_text(row: dict[str, Any]) -> str:
    return f"{row['tp']}/{row['fp']}/{row['fn']}"


def main() -> None:
    VIEWER.joinpath("plots").mkdir(parents=True, exist_ok=True)
    VIEWER.joinpath("tables").mkdir(parents=True, exist_ok=True)

    ref_rows = {key(row): row for row in read_jsonl(REFERENCE) if row.get("ok")}
    cand_rows = {key(row): row for row in read_jsonl(CANDIDATE) if row.get("ok")}
    keys = sorted(set(ref_rows) & set(cand_rows))
    if not keys:
        raise RuntimeError("no matched validation rows found")

    ref_scores = [score(ref_rows[k]) for k in keys]
    cand_scores = [score(cand_rows[k]) for k in keys]
    score_pairs = list(zip(ref_scores, cand_scores))
    confusion = np.zeros((5, 5), dtype=int)
    for a, b in score_pairs:
        confusion[a - 1, b - 1] += 1

    binary_rows: list[dict[str, Any]] = []
    for field in ["privacy", "intent", "answer", "full_info"]:
        ref_vals = [binary(ref_rows[k], field) for k in keys]
        cand_vals = [binary(cand_rows[k], field) for k in keys]
        pairs = list(zip(ref_vals, cand_vals))
        tn = sum((not a) and (not b) for a, b in pairs)
        fp = sum((not a) and b for a, b in pairs)
        fn = sum(a and (not b) for a, b in pairs)
        tp = sum(a and b for a, b in pairs)
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if tp + fn == 0 else tp / (tp + fn)
        binary_rows.append(
            {
                "metric": LABELS[field],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "stepfun_positive": tp + fn,
                "opus_positive": tp + fp,
                "agreement": (tn + tp) / len(keys),
                "precision_vs_stepfun": precision,
                "recall_vs_stepfun": recall,
                "mcc": pearson([int(v) for v in ref_vals], [int(v) for v in cand_vals]),
                "kappa": cohen_kappa(pairs),
                "mcnemar_p": exact_mcnemar_pvalue(fp, fn),
            }
        )

    score_summary = {
        "n": len(keys),
        "exact_agreement": sum(a == b for a, b in score_pairs) / len(keys),
        "within_one": sum(abs(a - b) <= 1 for a, b in score_pairs) / len(keys),
        "mae": sum(abs(a - b) for a, b in score_pairs) / len(keys),
        "pearson": pearson([float(v) for v in ref_scores], [float(v) for v in cand_scores]),
        "spearman": pearson(ranks(ref_scores), ranks(cand_scores)),
        "weighted_kappa_linear": weighted_kappa(score_pairs, "linear"),
        "weighted_kappa_quadratic": weighted_kappa(score_pairs, "quadratic"),
        "stepfun_distribution": dict(Counter(ref_scores)),
        "opus_distribution": dict(Counter(cand_scores)),
        "confusion": confusion.tolist(),
    }

    table_json = {"score_summary": score_summary, "binary_rows": binary_rows}
    (VIEWER / "tables" / "judge_validation_summary.json").write_text(
        json.dumps(table_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    latex_lines = [
        "\\begin{table}[t]",
        "\\small",
        "\\centering",
        "\\begin{tabular}{@{}llccc@{}}",
        "\\toprule",
        "\\textbf{Outcome} & \\textbf{Type} & \\textbf{Agreement} & \\textbf{$\\kappa$} & \\textbf{Secondary} \\\\",
        "\\midrule",
        (
            "Overall Privacy Score & 1--5 ordinal & "
            f"{fmt_pct(score_summary['exact_agreement'])} exact; {fmt_pct(score_summary['within_one'])} within-one & "
            f"QWK {fmt_float(score_summary['weighted_kappa_quadratic'])} & "
            f"Spearman {fmt_float(score_summary['spearman'])} \\\\"
        ),
        "\\midrule",
    ]
    for row in binary_rows:
        latex_lines.append(
            f"{row['metric']} & binary & {fmt_pct(row['agreement'])} & "
            f"{fmt_float(row['kappa'])} & TP/FP/FN {tp_fp_fn_text(row)} \\\\"
        )
    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{OpenRouter Opus 4.7 versus StepFun privacy-judge validation on {len(keys)} paired held-out rollouts. For the ordinal 1--5 privacy score, $\\kappa$ is quadratic weighted Cohen's kappa. For binary labels, $\\kappa$ is ordinary Cohen's kappa. TP/FP/FN treat StepFun as the reference label. Spearman is reported as an ordinal association diagnostic rather than the primary agreement statistic.}}",
            "\\label{tab:judge_validation_summary}",
            "\\end{table}",
            "",
        ]
    )
    (VIEWER / "tables" / "judge_validation_summary.tex").write_text("\n".join(latex_lines), encoding="utf-8")

    # Score confusion matrix.
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(confusion, cmap="YlGnBu", vmin=0)
    ax.set_xlabel("Opus Score")
    ax.set_ylabel("StepFun Score")
    ax.set_xticks(range(5), [str(i) for i in range(1, 6)])
    ax.set_yticks(range(5), [str(i) for i in range(1, 6)])
    ax.set_title("Judge Score Agreement")
    for i in range(5):
        for j in range(5):
            value = confusion[i, j]
            color = "white" if value > confusion.max() * 0.45 else "#1f2933"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rollouts")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(VIEWER / "plots" / f"judge_validation_score_confusion.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Binary agreement plot.
    labels = [row["metric"].replace("Full-Information", "Full-Info") for row in binary_rows]
    agreement = [100 * row["agreement"] for row in binary_rows]
    recall = [100 * (row["recall_vs_stepfun"] or 0) for row in binary_rows]
    precision = [100 * (row["precision_vs_stepfun"] or 0) for row in binary_rows]
    mcc = [row["mcc"] or 0 for row in binary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), gridspec_kw={"width_ratios": [1.2, 1.0]})
    x = np.arange(len(labels))
    width = 0.25
    axes[0].bar(x - width, agreement, width, label="Agreement", color=COLORS["blue"])
    axes[0].bar(x, precision, width, label="Precision", color=COLORS["green"])
    axes[0].bar(x + width, recall, width, label="Recall", color=COLORS["orange"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("%")
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].set_yticks(range(0, 101, 10), minor=True)
    axes[0].grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.8)
    axes[0].grid(axis="y", which="minor", color=COLORS["grid"], linewidth=0.4, alpha=0.65)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(x, mcc, color=COLORS["red"], width=0.55)
    axes[1].axhline(0, color="#303633", linewidth=0.8)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("MCC")
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].set_yticks(np.linspace(0, 1.0, 6))
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Binary Privacy-Label Agreement", y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(VIEWER / "plots" / f"judge_validation_binary_agreement.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote assets under {VIEWER}")


if __name__ == "__main__":
    main()
