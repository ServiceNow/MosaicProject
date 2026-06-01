#!/usr/bin/env python3
"""Build a slim step-100 paper viewer for Privacy HopQA q30 results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_q30_paper_viewer as base
from scripts import plot_privacy_hopqa_q30_training_efficiency as efficiency


DEFAULT_OUTPUT_DIR = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_paper_viewer_s100")
DEFAULT_LATEST_OUTPUT = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/latest_privacy_hopqa_q30_paper_viewer_s100.html")
TRAJECTORY_DIR = Path("../privacy_hopqa_q30_training_trajectory_20260524")
EFFICIENCY_DIR = Path("../privacy_hopqa_q30_training_efficiency_20260527")
PADR_S100_NORMAL_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_missing_ckpts_16kgen_32kctx_helperfe_20260524"
)
PADR_S100_PROMPT_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_privreward_tau05_s100_privacy_prompt_16kgen_32kctx_helperfe_20260524"
)
DRBENCH_EVAL_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/drbench_step100_eval")

BASE_LABEL = "qwen3_4b_instruct_2507"
TASK_S50_LABEL = "qwen3_4b_situational_reward_s50"
TASK_S100_LABEL = "qwen3_4b_situational_reward_s100"
PADR_S50_LABEL = "qwen3_4b_privreward_tau05_s50"
PADR_S100_LABEL = "qwen3_4b_privreward_tau05_s100"
STEP100_QUERY_EXAMPLE_MODELS = [BASE_LABEL, TASK_S100_LABEL, PADR_S100_LABEL]
STEP100_QUERY_EXAMPLE_PRIORITY_CASES = [
    ("2", "DR0058", "144d8a03"),  # password policy -> malicious domains
    ("1", "DR0046", "7ad0fe34"),  # cloud migration -> nation-state attack
    ("2", "DR0100", "c6d82d2b"),  # interoperability APIs -> zero-trust adoption
]

SUMMARY_METRICS = {
    "strict_chain": "strict_chain_success_rate",
    "final_accuracy": "final_correct_rate",
    "per_hop_accuracy": "mean_hop_accuracy",
    "privacy_leakage": "privacy_leakage_rate",
    "strict_privacy": "strict_privacy_rate",
    "intent": "intent4_rate",
    "answer": "answer_rate",
    "full": "full_info_rate",
    "lenient": "lenient_full_info_rate",
}
BEHAVIOR_METRICS = ["web_queries", "web_reads", "local_searches", "local_reads"]
UNTRAINED_MODEL_ORDER = [
    BASE_LABEL,
    "qwen3_8b",
    "step35_flash",
    "gpt_oss_20b",
    "chroma_context_1",
    "gemma4_31b_it",
]
UNTRAINED_MODEL_LATEX_LABELS = {
    BASE_LABEL: r"\qwenthreeb",
    "qwen3_8b": "Qwen3 8B",
    "step35_flash": "Step-3.5 Flash",
    "gpt_oss_20b": "GPT-OSS 20B",
    "chroma_context_1": "Chroma Context-1",
    "gemma4_31b_it": "Gemma 4 31B IT",
}
UNTRAINED_MODEL_HTML_LABELS = {
    BASE_LABEL: "Qwen3 4B",
    "qwen3_8b": "Qwen3 8B",
    "step35_flash": "Step-3.5 Flash",
    "gpt_oss_20b": "GPT-OSS 20B",
    "chroma_context_1": "Chroma Context-1",
    "gemma4_31b_it": "Gemma 4 31B IT",
}
UNTRAINED_METRICS = [
    ("strict_chain", "max", "strict_chain_success_rate"),
    ("per_hop_accuracy", "max", "hop_accuracy"),
    ("privacy_leakage", "min", "reward_rate"),
    ("intent", "min", "intent4_rate"),
    ("answer", "min", "answer_rate"),
    ("full", "min", "full_info_rate"),
]
DRBENCH_RUN_SPECS = [
    ("base", "Qwen3 4B", "r1", "minival_qwen3_4b_base_fewshot_20260528"),
    ("base", "Qwen3 4B", "r2", "minival_qwen3_4b_base_fewshot_rep2_20260529"),
    ("base", "Qwen3 4B", "r3", "minival_qwen3_4b_base_fewshot_rep3_20260529"),
    ("privacy", "PA-DR step100", "r1", "minival_privreward_tau05_s100_fewshot_rep1_20260529"),
    ("privacy", "PA-DR step100", "r2", "minival_privreward_tau05_s100_fewshot_rep2_20260529"),
    ("privacy", "PA-DR step100", "r3", "minival_privreward_tau05_s100_fewshot_rep3_20260529"),
]
DRBENCH_METRICS = [
    ("score", "Aggregate Score"),
    ("insights_recall", "Insight Recall"),
    ("factuality", "Factuality"),
    ("distractor_avoidance", "Distractor Avoidance"),
    ("report_quality", "Report Quality"),
]


def collect_metric_row(root: Path, model_label: str, condition: str, display: str) -> dict[str, Any]:
    rows = base.final_situational_metric_rows(root, condition, model_labels=[model_label])
    row = rows[0] if rows else {"model_label": model_label, "condition": condition, "status": "missing"}
    row = dict(row)
    row["model"] = display
    return row


def collect_rows() -> dict[str, dict[str, Any]]:
    return {
        "base_normal": collect_metric_row(
            base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT,
            BASE_LABEL,
            "Normal Prompt",
            "Qwen3 4B",
        ),
        "base_prompt": collect_metric_row(
            base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            BASE_LABEL,
            "Privacy Prompt",
            "Qwen3 4B + privacy prompt",
        ),
        "task_s50_normal": collect_metric_row(
            base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT,
            TASK_S50_LABEL,
            "Normal Prompt",
            "Task step50",
        ),
        "task_s50_prompt": collect_metric_row(
            base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            TASK_S50_LABEL,
            "Privacy Prompt",
            "Task step50 + privacy prompt",
        ),
        "task_s100_normal": collect_metric_row(
            base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT,
            TASK_S100_LABEL,
            "Normal Prompt",
            "Task step100",
        ),
        "task_s100_prompt": collect_metric_row(
            base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            TASK_S100_LABEL,
            "Privacy Prompt",
            "Task step100 + privacy prompt",
        ),
        "padr_s50_normal": collect_metric_row(
            base.DEFAULT_PRIVACY_AWARE_FINAL_EVAL_ARTIFACT_ROOT,
            PADR_S50_LABEL,
            "Normal Prompt",
            "PA-DR step50",
        ),
        "padr_s50_prompt": collect_metric_row(
            base.DEFAULT_PRIVACY_AWARE_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            PADR_S50_LABEL,
            "Privacy Prompt",
            "PA-DR step50 + privacy prompt",
        ),
        "padr_s100_normal": collect_metric_row(
            PADR_S100_NORMAL_ROOT,
            PADR_S100_LABEL,
            "Normal Prompt",
            "PA-DR step100",
        ),
        "padr_s100_prompt": collect_metric_row(
            PADR_S100_PROMPT_ROOT,
            PADR_S100_LABEL,
            "Privacy Prompt",
            "PA-DR step100 + privacy prompt",
        ),
    }


def row_with_raw_metrics(label: str, row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "condition": label,
        "attempts": f"{row.get('attempted_unique', 0)}/{row.get('expected', 1032)}",
        "judged": f"{row.get('privacy_unique', 0)}/{row.get('privacy_expected', 0)}",
        "status": row.get("status", "missing"),
    }
    for output_key, source_key in SUMMARY_METRICS.items():
        out[output_key] = base.pct_pm_text(row, source_key)
        out[f"{output_key}_value"] = base.live.as_float(row.get(source_key))
        out[f"{output_key}_sem"] = base.live.as_float(row.get(f"{source_key}_sem"))
    return out


def main_summary_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Qwen3 4B", rows["base_normal"]),
        ("Qwen3 4B + privacy prompt", rows["base_prompt"]),
        ("Task Training", rows["task_s100_normal"]),
        ("Task Training + privacy prompt", rows["task_s100_prompt"]),
        ("Task+PA-DR Training", rows["padr_s100_normal"]),
        ("Task+PA-DR Training + privacy prompt", rows["padr_s100_prompt"]),
    ]
    return [row_with_raw_metrics(label, row) for label, row in specs]


def checkpoint_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Task step50", rows["task_s50_normal"]),
        ("Task step100", rows["task_s100_normal"]),
        ("Task step50 + privacy prompt", rows["task_s50_prompt"]),
        ("Task step100 + privacy prompt", rows["task_s100_prompt"]),
        ("PA-DR step50", rows["padr_s50_normal"]),
        ("PA-DR step100", rows["padr_s100_normal"]),
        ("PA-DR step50 + privacy prompt", rows["padr_s50_prompt"]),
        ("PA-DR step100 + privacy prompt", rows["padr_s100_prompt"]),
    ]
    return [row_with_raw_metrics(label, row) for label, row in specs]


def status_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Base", rows["base_normal"]),
        ("Base + privacy prompt", rows["base_prompt"]),
        ("Task step100", rows["task_s100_normal"]),
        ("Task step100 + privacy prompt", rows["task_s100_prompt"]),
        ("PA-DR step100", rows["padr_s100_normal"]),
        ("PA-DR step100 + privacy prompt", rows["padr_s100_prompt"]),
    ]
    out: list[dict[str, Any]] = []
    for label, row in specs:
        privacy_unique = int(row.get("privacy_unique") or 0)
        privacy_expected = int(row.get("privacy_expected") or 0)
        out.append(
            {
                "condition": label,
                "attempts": f"{row.get('attempted_unique', 0)}/{row.get('expected', 1032)}",
                "successful_rollouts": row.get("ok_rollout_unique", 0),
                "privacy_judgments": f"{privacy_unique}/{privacy_expected}",
                "missing_privacy_judgments": max(0, privacy_expected - privacy_unique),
                "status": row.get("status", "missing"),
            }
        )
    return out


def behavior_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Base", "Qwen3 4B", "Normal", rows["base_normal"]),
        ("Base", "Qwen3 4B + privacy prompt", "Privacy Prompt", rows["base_prompt"]),
        ("Task", "Task", "Normal", rows["task_s100_normal"]),
        ("Task", "Task + privacy prompt", "Privacy Prompt", rows["task_s100_prompt"]),
        ("PA-DR", "PA-DR", "Normal", rows["padr_s100_normal"]),
        ("PA-DR", "PA-DR + privacy prompt", "Privacy Prompt", rows["padr_s100_prompt"]),
    ]
    out: list[dict[str, Any]] = []
    for training_state, condition, prompt_state, row in specs:
        item = {
            "training_state": training_state,
            "condition": condition,
            "prompt_state": prompt_state,
            "successful_rollouts": row.get("ok_rollout_unique"),
        }
        for key in BEHAVIOR_METRICS:
            item[key] = row.get(key)
            item[f"{key}_sem"] = row.get(f"{key}_sem")
        out.append(item)
    return out


def latex_training_summary(rows: list[dict[str, Any]]) -> str:
    metric_specs = [
        ("strict_chain", "max"),
        ("per_hop_accuracy", "max"),
        ("privacy_leakage", "min"),
        ("intent", "min"),
        ("answer", "min"),
        ("full", "min"),
    ]
    best: dict[str, float] = {}
    for key, direction in metric_specs:
        values = [value for row in rows if (value := base.live.as_float(row.get(f"{key}_value"))) is not None]
        if values:
            best[key] = max(values) if direction == "max" else min(values)
    labels = [
        r"\qwenthreeb",
        r"\quad + privacy prompt",
        "Task Training",
        r"\quad + privacy prompt",
        "Task+PA-DR Training",
        r"\quad + privacy prompt",
    ]
    body: list[str] = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        if index in {2, 4}:
            body.append(r"\midrule")
        cells = [label]
        for key, _direction in metric_specs:
            value = base.live.as_float(row.get(f"{key}_value"))
            is_best = value is not None and key in best and math.isclose(value, best[key], abs_tol=1e-12)
            cells.append(base.latex_pct_pm_cell(row, key, bold=is_best))
        body.append(" & ".join(cells) + r" \\")
    body_text = "\n".join(body)
    return rf"""\begin{{table*}}[t]
\small
\centering
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
\textbf{{Model}} & \textbf{{\shortstack{{Strict Chain\\Success ($\uparrow$)}}}} & \textbf{{\shortstack{{Hop-Level\\Accuracy ($\uparrow$)}}}} & \textbf{{\shortstack{{Privacy\\Leakage ($\downarrow$)}}}} & \textbf{{\shortstack{{Intent\\Leakage ($\downarrow$)}}}} & \textbf{{\shortstack{{Answer\\Leakage ($\downarrow$)}}}} & \textbf{{\shortstack{{Full-Information\\Leakage ($\downarrow$)}}}} \\
\midrule
{body_text}
\bottomrule
\end{{tabular}}
\caption{{Effect of RL training on \qwenthreeb, evaluated with and without the privacy-aware prompt, using checkpoint 100 for both training runs. We train one model purely on task performance and one with our Privacy Aware-Deep Research (PA-DR) method that also penalizes privacy leakage. Values after $\pm$ are SEMs in percentage points.}}
\label{{tab:training_effect_summary_s100}}
\end{{table*}}
"""


def untrained_model_summary_rows(current_viewer_data_path: Path) -> list[dict[str, Any]]:
    data = json.loads(current_viewer_data_path.read_text(encoding="utf-8"))
    by_model = {row.get("model_label"): row for row in data.get("privacy_prompt_summary", [])}
    out: list[dict[str, Any]] = []
    for model_label in UNTRAINED_MODEL_ORDER:
        row = by_model.get(model_label)
        if not row:
            continue
        for prefix, latex_label, html_label in [
            (
                "baseline",
                UNTRAINED_MODEL_LATEX_LABELS[model_label],
                UNTRAINED_MODEL_HTML_LABELS[model_label],
            ),
            ("prompt", r"\quad + privacy prompt", "+ privacy prompt"),
        ]:
            item: dict[str, Any] = {
                "model_label": model_label,
                "condition": prefix,
                "latex_label": latex_label,
                "html_label": html_label,
            }
            for output_key, _direction, source_key in UNTRAINED_METRICS:
                item[f"{output_key}_value"] = row.get(f"{prefix}_{source_key}")
                item[f"{output_key}_sem"] = row.get(f"{prefix}_{source_key}_sem")
            out.append(item)
    return out


def latex_untrained_model_summary(rows: list[dict[str, Any]]) -> str:
    best: dict[str, float] = {}
    for key, direction, _source in UNTRAINED_METRICS:
        values = [value for row in rows if (value := base.live.as_float(row.get(f"{key}_value"))) is not None]
        if values:
            best[key] = max(values) if direction == "max" else min(values)
    body: list[str] = []
    for index, row in enumerate(rows):
        if index and row.get("condition") == "baseline":
            body.append(r"\midrule")
        cells = [str(row.get("latex_label") or "")]
        for key, _direction, _source in UNTRAINED_METRICS:
            value = base.live.as_float(row.get(f"{key}_value"))
            is_best = value is not None and key in best and math.isclose(value, best[key], abs_tol=1e-12)
            cells.append(base.latex_pct_pm_cell(row, key, bold=is_best))
        body.append(" & ".join(cells) + r" \\")
    body_text = "\n".join(body)
    return rf"""\begin{{table*}}[t]
\small
\centering
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
\textbf{{Model}} & \textbf{{\shortstack{{Strict Chain\\Success ($\uparrow$)}}}} & \textbf{{\shortstack{{Hop-Level\\Accuracy ($\uparrow$)}}}} & \textbf{{\shortstack{{Privacy\\Leakage ($\downarrow$)}}}} & \textbf{{\shortstack{{Intent\\Leakage ($\downarrow$)}}}} & \textbf{{\shortstack{{Answer\\Leakage ($\downarrow$)}}}} & \textbf{{\shortstack{{Full-Information\\Leakage ($\downarrow$)}}}} \\
\midrule
{body_text}
\bottomrule
\end{{tabular}}
\caption{{Comparison of untrained models on the held-out Privacy HopQA test set, evaluated with and without the privacy-aware prompt. Values after $\pm$ are SEMs in percentage points. Chroma Context-1 uses the medium OpenRouter configuration.}}
\label{{tab:untrained_model_summary}}
\end{{table*}}
"""


def untrained_model_rows_for_html(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "model": row.get("html_label"),
                "strict_chain": base.pct_pm_text(row, "strict_chain"),
                "per_hop_accuracy": base.pct_pm_text(row, "per_hop_accuracy"),
                "privacy_leakage": base.pct_pm_text(row, "privacy_leakage"),
                "intent": base.pct_pm_text(row, "intent"),
                "answer": base.pct_pm_text(row, "answer"),
                "full": base.pct_pm_text(row, "full"),
            }
        )
    return out


def latex_behavior_summary(rows: list[dict[str, Any]]) -> str:
    labels = [
        r"\qwenthreeb",
        r"\quad + privacy prompt",
        "Task",
        r"\quad + privacy prompt",
        "PA-DR",
        r"\quad + privacy prompt",
    ]
    body: list[str] = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        if index in {2, 4}:
            body.append(r"\midrule")
        cells = [label, *(base.latex_num_pm_cell(row, key) for key in BEHAVIOR_METRICS)]
        body.append(" & ".join(cells) + r" \\")
    body_text = "\n".join(body)
    return rf"""\begin{{table*}}[t]
\small
\centering
\begin{{tabular}}{{lllll}}
\toprule
\textbf{{Model}} & \textbf{{Mean Web Queries}} & \textbf{{Mean Web Reads}} & \textbf{{Mean Local Searches}} & \textbf{{Mean Local Reads}} \\
\midrule
{body_text}
\bottomrule
\end{{tabular}}
\caption{{Effect of step-100 training on query and retrieval behavior for \qwenthreeb. Values are means per successful rollout; values after $\pm$ are SEMs over repeat means.}}
\label{{tab:final_training_behavior_s100}}
\end{{table*}}
"""


def latex_details(title: str, latex_text: str) -> str:
    return (
        f"<details class=\"latex-block\"><summary>{base.esc(title)}</summary>"
        f"<pre><code>{base.esc(latex_text)}</code></pre></details>"
    )


def step100_query_example_rows(*, limit: int = 14) -> list[dict[str, Any]]:
    problem_lookup = base.live.load_problem_lookup(base.live.DEFAULT_DATASET_ROOT, base.live.DEFAULT_DATASET_BASENAME)
    standard_rollouts = base.live.load_rollouts(base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT, problem_lookup)
    standard_stepfun = base.live.copy_rollout_identity_from_trace(
        base.live.load_leakage(base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT, "stepfun"),
        standard_rollouts,
    )
    padr_rollouts = base.live.load_rollouts(PADR_S100_NORMAL_ROOT, problem_lookup)
    padr_stepfun = base.live.copy_rollout_identity_from_trace(
        base.live.load_leakage(PADR_S100_NORMAL_ROOT, "stepfun"),
        padr_rollouts,
    )
    rollout_by_identity = {
        base.live.rollout_identity(row): row
        for row in [*standard_rollouts, *padr_rollouts]
        if row.get("model_label") in STEP100_QUERY_EXAMPLE_MODELS and row.get("split") == "test"
    }
    by_case: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in [*standard_stepfun, *padr_stepfun]:
        model_label = str(row.get("model_label") or "")
        if model_label not in STEP100_QUERY_EXAMPLE_MODELS or row.get("split") != "test":
            continue
        case_key = (str(row.get("repeat") or 0), str(row.get("task_id") or ""), str(row.get("chain_id") or ""))
        by_case.setdefault(case_key, {})[model_label] = row

    candidates: list[dict[str, Any]] = []
    for case_key, rows_by_model in by_case.items():
        if not all(model in rows_by_model for model in STEP100_QUERY_EXAMPLE_MODELS):
            continue
        base_row, task_row, privacy_row = [rows_by_model[model] for model in STEP100_QUERY_EXAMPLE_MODELS]
        problem = problem_lookup.get((case_key[1], case_key[2])) or {}
        pattern = str(problem.get("pattern") or base_row.get("pattern") or "")
        if "W" not in pattern:
            continue
        strict_successes = sum(
            1
            for model in STEP100_QUERY_EXAMPLE_MODELS
            if rollout_by_identity.get((model, "test", case_key[1], case_key[2], case_key[0]), {}).get(
                "strict_chain_success"
            )
        )
        if strict_successes < 3:
            continue
        if not (
            base.live.reward_positive(base_row)
            and base.live.reward_positive(task_row)
            and not base.live.reward_positive(privacy_row)
        ):
            continue

        base_rollout = rollout_by_identity.get((BASE_LABEL, "test", case_key[1], case_key[2], case_key[0]), {})
        task_rollout = rollout_by_identity.get((TASK_S100_LABEL, "test", case_key[1], case_key[2], case_key[0]), {})
        privacy_rollout = rollout_by_identity.get((PADR_S100_LABEL, "test", case_key[1], case_key[2], case_key[0]), {})

        _base_hits, base_unique_hits = base.paired_example_query_hits(base_row)
        _task_hits, task_unique_hits = base.paired_example_query_hits(task_row)
        _privacy_hits, privacy_unique_hits = base.paired_example_query_hits(privacy_row)
        base_leakage_types = base.leakage_type_labels(base_row)
        task_leakage_types = base.leakage_type_labels(task_row)
        privacy_leakage_types = base.leakage_type_labels(privacy_row)
        score = 0.0
        score += 18.0 * strict_successes
        score += 8.0 * len(base_unique_hits) + 8.0 * len(task_unique_hits) - 8.0 * len(privacy_unique_hits)
        score += 5.0 * len(base_leakage_types) + 6.0 * len(task_leakage_types) - 3.0 * len(privacy_leakage_types)
        score += 3.5 * (base.live.score(base_row) - 3) + 4.0 * (base.live.score(task_row) - 3)
        score += 2.0 * max(0, base.live.web_query_count(task_row) - base.live.web_query_count(privacy_row))
        score -= 0.35 * max(
            base.live.web_query_count(base_row),
            base.live.web_query_count(task_row),
            base.live.web_query_count(privacy_row),
        )
        if base_leakage_types and task_leakage_types:
            score += 8.0
        if int(problem.get("n_hops") or base_row.get("n_hops") or 0) <= 3:
            score += 4.0
        candidates.append(
            {
                "score": score,
                "case_key": case_key,
                "task_id": case_key[1],
                "chain_id": case_key[2],
                "repeat": case_key[0],
                "topic": f"{case_key[1]}/{case_key[2]}",
                "pattern": pattern,
                "n_hops": problem.get("n_hops") or base_row.get("n_hops"),
                "hops": base.paired_example_problem_hops(problem),
                "questions": base.paired_example_problem_questions(problem),
                "base_score": base.live.score(base_row),
                "task_score": base.live.score(task_row),
                "privacy_score": base.live.score(privacy_row),
                "base_rating": base.paired_example_leakage_label(base_row),
                "task_rating": base.paired_example_leakage_label(task_row),
                "privacy_rating": base.paired_example_leakage_label(privacy_row),
                "base_leakage_types": base_leakage_types,
                "task_leakage_types": task_leakage_types,
                "privacy_leakage_types": privacy_leakage_types,
                "base_queries": base.paired_example_web_queries(base_row),
                "task_queries": base.paired_example_web_queries(task_row),
                "privacy_queries": base.paired_example_web_queries(privacy_row),
                "base_selected_queries": base.paired_example_selected_queries(base_row, limit=3),
                "task_selected_queries": base.paired_example_selected_queries(task_row, limit=3),
                "privacy_selected_queries": base.paired_example_selected_queries(privacy_row, limit=3),
                "base_web_query_count": base.live.web_query_count(base_row),
                "task_web_query_count": base.live.web_query_count(task_row),
                "privacy_web_query_count": base.live.web_query_count(privacy_row),
                "base_hop_accuracy": base.live.task_accuracy(base_rollout),
                "task_hop_accuracy": base.live.task_accuracy(task_rollout),
                "privacy_hop_accuracy": base.live.task_accuracy(privacy_rollout),
                "base_strict_chain_success": bool(base_rollout.get("strict_chain_success")),
                "task_strict_chain_success": bool(task_rollout.get("strict_chain_success")),
                "privacy_strict_chain_success": bool(privacy_rollout.get("strict_chain_success")),
                "base_final_correct": bool(base_rollout.get("final_correct")),
                "task_final_correct": bool(task_rollout.get("final_correct")),
                "privacy_final_correct": bool(privacy_rollout.get("final_correct")),
            }
        )
    candidates.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)

    candidate_by_case = {tuple(str(part) for part in row.get("case_key") or ()): row for row in candidates}
    selected: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    seen_chains: set[tuple[str, str]] = set()

    def maybe_select(row: dict[str, Any]) -> bool:
        chain_key = (str(row.get("task_id") or ""), str(row.get("chain_id") or ""))
        if chain_key in seen_chains:
            return False
        if task_counts[str(row.get("task_id") or "")] >= 2:
            return False
        seen_chains.add(chain_key)
        task_counts[str(row.get("task_id") or "")] += 1
        row["selection_source"] = row.get("selection_source") or "ranked"
        row["rank"] = len(selected) + 1
        selected.append(row)
        return True

    for case_key in STEP100_QUERY_EXAMPLE_PRIORITY_CASES:
        row = candidate_by_case.get(tuple(case_key))
        if row is None:
            continue
        row["selection_source"] = "review-prioritized"
        maybe_select(row)
        if len(selected) >= limit:
            break
    for row in candidates:
        if len(selected) >= limit:
            break
        maybe_select(row)
    return selected


def latex_step100_query_examples(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        cells = [
            base.paired_examples_latex_cell(base.curated_chain_summary_lines(row), max_lines=5),
            base.paired_examples_latex_query_bundle([str(item) for item in row.get("base_queries") or []], max_lines=12),
            base.paired_examples_latex_query_bundle([str(item) for item in row.get("task_queries") or []], max_lines=12),
            base.paired_examples_latex_query_bundle([str(item) for item in row.get("privacy_queries") or []], max_lines=12),
            base.leakage_type_latex(row),
        ]
        body.append(" & ".join(cells) + r" \\")
    body_text = "\n\\midrule\n".join(body)
    return rf"""\begin{{table*}}[t]
\scriptsize
\centering
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}p{{0.24\textwidth}}p{{0.18\textwidth}}p{{0.18\textwidth}}p{{0.18\textwidth}}p{{0.16\textwidth}}@{{}}}}
\toprule
\textbf{{Question Chain}} & \textbf{{Base \qwenthreeb Web Queries}} & \textbf{{Task Web Queries}} & \textbf{{PA-DR Web Queries}} & \textbf{{Leakage Types}} \\
\midrule
{body_text}
\bottomrule
\end{{tabular}}
\caption{{Representative held-out test examples from checkpoint 100 where the base and task-trained \qwenthreeb rollouts are judged privacy-leaking, while the PA-DR rollout for the same test-chain repeat has no judged leakage. All three rollouts have strict chain success; web-query columns show every web query made by that rollout. Leakage types are not mutually exclusive.}}
\label{{tab:paired_query_leakage_examples_s100}}
\end{{table*}}
"""


def query_example_accuracy_text(row: dict[str, Any], prefix: str) -> str:
    hop_accuracy = base.live.as_float(row.get(f"{prefix}_hop_accuracy"))
    hop_text = f"{100.0 * hop_accuracy:.0f}% hop accuracy" if hop_accuracy is not None else "hop accuracy n/a"
    chain_text = "strict chain success" if row.get(f"{prefix}_strict_chain_success") else "strict chain failure"
    final_text = "final correct" if row.get(f"{prefix}_final_correct") else "final incorrect"
    return f"{hop_text}; {chain_text}; {final_text}"


def step100_query_model_cell_html(row: dict[str, Any], prefix: str) -> str:
    leakage = base.leakage_type_text([str(item) for item in row.get(f"{prefix}_leakage_types") or []])
    queries = [str(item) for item in row.get(f"{prefix}_queries") or []]
    count = int(row.get(f"{prefix}_web_query_count") or len(queries))
    return (
        f"<div class=\"rating\">{base.esc(leakage)}</div>"
        f"<div class=\"muted\">{base.esc(query_example_accuracy_text(row, prefix))}; {count} web queries total</div>"
        f"{base.paired_example_query_list_html(queries)}"
    )


def step100_query_examples_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No step100 query examples found.</p>"
    body: list[str] = []
    for row in rows:
        case = (
            f"<strong>{base.esc(row.get('rank'))}. {base.esc(row.get('task_id'))}/{base.esc(row.get('chain_id'))}</strong>"
            f"<div class=\"muted\">r{base.esc(row.get('repeat'))}; {base.esc(row.get('pattern'))}; "
            f"{base.esc(row.get('n_hops'))} hops</div>"
        )
        leakage_types = (
            f"{base.leakage_type_html(row, 'base_leakage_types', 'Base')}<br>"
            f"{base.leakage_type_html(row, 'task_leakage_types', 'Task')}<br>"
            f"{base.leakage_type_html(row, 'privacy_leakage_types', 'PA-DR')}"
        )
        body.append(
            "<tr>"
            f"<td>{case}</td>"
            f"<td>{base.curated_chain_summary_html(row)}</td>"
            f"<td>{step100_query_model_cell_html(row, 'base')}</td>"
            f"<td>{step100_query_model_cell_html(row, 'task')}</td>"
            f"<td>{step100_query_model_cell_html(row, 'privacy')}</td>"
            f"<td>{leakage_types}</td>"
            "</tr>"
        )
    return (
        "<table class=\"paper-example-table\"><thead><tr>"
        "<th>Case</th><th>Question Chain</th><th>Base Qwen3 4B Queries</th><th>Task Queries</th><th>PA-DR Queries</th><th>Leakage Types</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_training_effect(rows: dict[str, dict[str, Any]], path: Path) -> None:
    specs = [
        ("base_normal", "Base", "#1f2421", "o", True, (-7, 7), "right", "bottom"),
        ("base_prompt", "Base + prompt", "#1f2421", "D", False, (7, -7), "left", "top"),
        ("task_s100_normal", "Task", "#D55E00", "s", True, (8, 8), "left", "bottom"),
        ("task_s100_prompt", "Task + prompt", "#D55E00", "D", False, (-14, 0), "right", "center"),
        ("padr_s100_normal", "PA-DR", "#009E73", "s", True, (8, 7), "left", "bottom"),
        ("padr_s100_prompt", "PA-DR + prompt", "#009E73", "D", False, (0, -11), "center", "top"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    points: dict[str, tuple[float, float]] = {}
    for key, label, color, marker, filled, label_offset, label_ha, label_va in specs:
        row = rows[key]
        x = base.live.as_float(row.get("strict_chain_success_rate"))
        y = base.live.as_float(row.get("privacy_leakage_rate"))
        if x is None or y is None:
            continue
        xy = (100 * x, 100 * y)
        points[key] = xy
        alpha = 1.0 if row.get("complete") else 0.62
        ax.errorbar(
            xy[0],
            xy[1],
            xerr=100 * (base.live.as_float(row.get("strict_chain_success_rate_sem")) or 0.0),
            yerr=100 * (base.live.as_float(row.get("privacy_leakage_rate_sem")) or 0.0),
            fmt=marker,
            markersize=6.2,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=1.35,
            color=color,
            ecolor=color,
            elinewidth=1.15,
            capsize=3.0,
            alpha=alpha,
            zorder=4,
        )
        point_label = f"{label}*" if not row.get("complete") else label
        ax.annotate(
            point_label,
            xy=xy,
            xytext=label_offset,
            textcoords="offset points",
            fontsize=9.2,
            color=color,
            ha=label_ha,
            va=label_va,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.90},
            zorder=5,
        )
    arrows = [
        ("base_normal", "task_s100_normal", "#D55E00", "-"),
        ("base_normal", "padr_s100_normal", "#009E73", "-"),
        ("base_normal", "base_prompt", "#1f2421", "--"),
    ]
    for start_key, end_key, color, linestyle in arrows:
        if start_key not in points or end_key not in points:
            continue
        ax.annotate(
            "",
            xy=points[end_key],
            xytext=points[start_key],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.25 if linestyle == "--" else 1.6,
                "linestyle": linestyle,
                "alpha": 0.9,
                "shrinkA": 6,
                "shrinkB": 6,
            },
            zorder=2,
        )
    ax.set_xlabel("Strict Chain Success (%)")
    ax.set_ylabel("Privacy Leakage (%)")
    ax.set_xlim(40, 65)
    ax.set_ylim(0, 60)
    ax.set_xticks([40, 45, 50, 55, 60, 65])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    base.add_minor_grid(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f2421", markeredgecolor="#1f2421", markersize=6.2, label="Base Qwen3 4B"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#D55E00", markeredgecolor="#D55E00", markersize=6.2, label="Task training"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#009E73", markeredgecolor="#009E73", markersize=6.2, label="Task+PA-DR training"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#5F6F7A", markersize=6.2, label="+ privacy prompt"),
        Line2D([0], [0], color="#5F6F7A", lw=1.6, label="training effect"),
    ]
    ax.legend(handles=handles, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    base.save_figure(fig, path)


def plot_core_leakage_types(rows: list[dict[str, Any]], path: Path) -> None:
    if len(rows) < 6:
        return
    metrics = [
        ("Intent Leakage", "intent"),
        ("Answer Leakage", "answer"),
        ("Full-Information Leakage", "full"),
    ]
    normal_rows = [rows[0], rows[2], rows[4]]
    prompt_rows = [rows[1], rows[3], rows[5]]
    labels = ["Base", "Task", "PA-DR"]
    all_percent_values: list[float] = []
    for _title, key in metrics:
        for row in [*normal_rows, *prompt_rows]:
            value = base.live.as_float(row.get(f"{key}_value"))
            sem = base.live.as_float(row.get(f"{key}_sem")) or 0.0
            if value is not None:
                all_percent_values.append(100 * (value + sem))
    y_upper = base.nice_percent_upper_from_values(all_percent_values)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharey=True)
    x = list(range(len(labels)))
    width = 0.36
    normal_pos = [item - width / 2 for item in x]
    prompt_pos = [item + width / 2 for item in x]
    error_kw = {"elinewidth": 1.1, "ecolor": "#303633", "capsize": 3, "capthick": 1.0}
    for ax, (title, key) in zip(axes, metrics):
        normal_vals = [100 * (base.live.as_float(row.get(f"{key}_value")) or 0.0) for row in normal_rows]
        prompt_vals = [100 * (base.live.as_float(row.get(f"{key}_value")) or 0.0) for row in prompt_rows]
        normal_yerr = [100 * (base.live.as_float(row.get(f"{key}_sem")) or 0.0) for row in normal_rows]
        prompt_yerr = [100 * (base.live.as_float(row.get(f"{key}_sem")) or 0.0) for row in prompt_rows]
        ax.bar(
            normal_pos,
            normal_vals,
            width=width,
            color=base.BASELINE_COLOR,
            label="Baseline",
            yerr=normal_yerr,
            error_kw=error_kw,
        )
        ax.bar(
            prompt_pos,
            prompt_vals,
            width=width,
            color=base.PROMPT_COLOR,
            label="Privacy Prompt",
            yerr=prompt_yerr,
            error_kw=error_kw,
        )
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.xaxis.set_minor_locator(NullLocator())
        base.set_percent_y_axis(ax, y_upper)
    axes[0].set_ylabel("% Occurrence")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.10))
    base.save_figure(fig, path)


def plot_behavior(rows: list[dict[str, Any]], path: Path) -> None:
    states = ["Base", "Task", "PA-DR"]
    row_map = {(row.get("training_state"), row.get("prompt_state")): row for row in rows}
    normal_rows = [row_map.get((state, "Normal")) or {} for state in states]
    prompt_rows = [row_map.get((state, "Privacy Prompt")) or {} for state in states]
    metrics = [
        ("Mean Web Queries", "web_queries", "web"),
        ("Mean Web Reads", "web_reads", "web"),
        ("Mean Local Searches", "local_searches", "local"),
        ("Mean Local Reads", "local_reads", "local"),
    ]
    group_max: dict[str, float] = {}
    for _title, key, group in metrics:
        values = [
            value
            for row in [*normal_rows, *prompt_rows]
            if (value := base.live.as_float(row.get(key))) is not None
        ]
        group_max[group] = max(group_max.get(group, 1.0), *values, 1.0) * 1.16
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8))
    x = list(range(len(states)))
    width = 0.34
    for ax, (title, key, group) in zip(axes.flat, metrics):
        normal_vals = [base.live.as_float(row.get(key)) or math.nan for row in normal_rows]
        prompt_vals = [base.live.as_float(row.get(key)) or math.nan for row in prompt_rows]
        ax.bar(
            [item - width / 2 for item in x],
            normal_vals,
            width=width,
            color=base.BASELINE_COLOR,
            label="Normal Prompt",
            yerr=base.count_yerr(normal_rows, key),
            error_kw={"elinewidth": 1.0, "ecolor": "#303633", "capsize": 2.6, "capthick": 0.9},
        )
        ax.bar(
            [item + width / 2 for item in x],
            prompt_vals,
            width=width,
            color=base.PROMPT_COLOR,
            label="Privacy-Aware Prompt",
            yerr=base.count_yerr(prompt_rows, key),
            error_kw={"elinewidth": 1.0, "ecolor": "#303633", "capsize": 2.6, "capthick": 0.9},
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(states)
        ax.xaxis.set_minor_locator(NullLocator())
        base.set_nice_count_y_axis(ax, group_max[group], target_ticks=5)
    axes[0, 0].set_ylabel("Mean Count per Rollout")
    axes[1, 0].set_ylabel("Mean Count per Rollout")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.supxlabel("Training State", y=0.02)
    base.save_figure(fig, path)


def table_rows_for_html(rows: list[dict[str, Any]], include_status: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "condition": row["condition"],
            "strict_chain": row["strict_chain"],
            "final_accuracy": row["final_accuracy"],
            "per_hop_accuracy": row["per_hop_accuracy"],
            "privacy_leakage": row["privacy_leakage"],
            "intent": row["intent"],
            "answer": row["answer"],
            "full": row["full"],
        }
        if include_status:
            item["judged"] = row["judged"]
            item["status"] = row["status"]
        out.append(item)
    return out


def behavior_rows_for_html(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition": row["condition"],
            "web_queries": base.num_pm_text(row, "web_queries"),
            "web_reads": base.num_pm_text(row, "web_reads"),
            "local_searches": base.num_pm_text(row, "local_searches"),
            "local_reads": base.num_pm_text(row, "local_reads"),
        }
        for row in rows
    ]


TOKEN_USAGE_CONDITIONS = [
    ("Base", base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT, BASE_LABEL),
    ("Base + privacy prompt", base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT, BASE_LABEL),
    ("Task s100", base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT, TASK_S100_LABEL),
    ("Task s100 + privacy prompt", base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT, TASK_S100_LABEL),
    ("PA-DR s100", PADR_S100_NORMAL_ROOT, PADR_S100_LABEL),
    ("PA-DR s100 + privacy prompt", PADR_S100_PROMPT_ROOT, PADR_S100_LABEL),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sum_stepfun_usage_once(row: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    seen_usage_ids: set[int] = set()

    def add_usage(usage: dict[str, Any]) -> None:
        nonlocal prompt_tokens, completion_tokens
        usage_id = id(usage)
        if usage_id in seen_usage_ids:
            return
        seen_usage_ids.add(usage_id)
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            stepfun_meta = obj.get("_stepfun") if isinstance(obj.get("_stepfun"), dict) else None
            if stepfun_meta and isinstance(stepfun_meta.get("usage"), dict):
                add_usage(stepfun_meta["usage"])
                for key, child in obj.items():
                    if key != "_stepfun":
                        walk(child)
                return
            if isinstance(obj.get("usage"), dict):
                add_usage(obj["usage"])
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(row)
    return prompt_tokens, completion_tokens


def mean_or_none(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def token_usage_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_agent_input: list[int] = []
    all_agent_output: list[int] = []
    all_eval_input: list[int] = []
    all_eval_output: list[int] = []
    for condition, artifact_root, model_label in TOKEN_USAGE_CONDITIONS:
        model_root = artifact_root / "model_runs" / model_label
        rollout_rows = read_jsonl(model_root / "tables" / "rollout_index.jsonl")
        successful_rollouts = [
            row
            for row in rollout_rows
            if row.get("ok") and (row.get("split") in (None, "test"))
        ]
        successful_ids = {
            (row.get("split"), row.get("repeat"), row.get("chain_id"))
            for row in successful_rollouts
        }
        agent_input = [int(row.get("prompt_tokens") or 0) for row in successful_rollouts]
        agent_output = [int(row.get("output_tokens") or 0) for row in successful_rollouts]

        raw_privacy_rows = read_jsonl(model_root / "tables" / "privacy_leakage_suite_v2_stepfun_full.jsonl")
        privacy_rows = base.live.dedupe_rows_by_latest(
            base.live.copy_rollout_identity_from_trace(raw_privacy_rows, rollout_rows),
            "stepfun",
        )
        judged_rows = [
            row
            for row in privacy_rows
            if (row.get("split"), row.get("repeat"), row.get("chain_id")) in successful_ids
        ]
        eval_input: list[int] = []
        eval_output: list[int] = []
        for row in judged_rows:
            prompt_tokens, completion_tokens = sum_stepfun_usage_once(row)
            eval_input.append(prompt_tokens)
            eval_output.append(completion_tokens)

        all_agent_input.extend(agent_input)
        all_agent_output.extend(agent_output)
        all_eval_input.extend(eval_input)
        all_eval_output.extend(eval_output)
        rows.append(
            {
                "condition": condition,
                "agent_rollouts": len(agent_input),
                "agent_input_mean": mean_or_none(agent_input),
                "agent_output_mean": mean_or_none(agent_output),
                "agent_total_mean": mean_or_none([x + y for x, y in zip(agent_input, agent_output)]),
                "agent_input_total": sum(agent_input),
                "agent_output_total": sum(agent_output),
                "agent_total": sum(agent_input) + sum(agent_output),
                "privacy_judgments": len(eval_input),
                "eval_input_mean": mean_or_none(eval_input),
                "eval_output_mean": mean_or_none(eval_output),
                "eval_total_mean": mean_or_none([x + y for x, y in zip(eval_input, eval_output)]),
                "eval_input_total": sum(eval_input),
                "eval_output_total": sum(eval_output),
                "eval_total": sum(eval_input) + sum(eval_output),
            }
        )

    combined = {
        "condition": "All six step100 conditions",
        "agent_rollouts": len(all_agent_input),
        "agent_input_mean": mean_or_none(all_agent_input),
        "agent_output_mean": mean_or_none(all_agent_output),
        "agent_total_mean": mean_or_none([x + y for x, y in zip(all_agent_input, all_agent_output)]),
        "agent_input_total": sum(all_agent_input),
        "agent_output_total": sum(all_agent_output),
        "agent_total": sum(all_agent_input) + sum(all_agent_output),
        "privacy_judgments": len(all_eval_input),
        "eval_input_mean": mean_or_none(all_eval_input),
        "eval_output_mean": mean_or_none(all_eval_output),
        "eval_total_mean": mean_or_none([x + y for x, y in zip(all_eval_input, all_eval_output)]),
        "eval_input_total": sum(all_eval_input),
        "eval_output_total": sum(all_eval_output),
        "eval_total": sum(all_eval_input) + sum(all_eval_output),
    }
    return rows, combined


def format_token_mean(value: Any) -> str:
    number = base.live.as_float(value)
    if number is None:
        return "n/a"
    return f"{number / 1000:.1f}k"


def format_token_total(value: Any) -> str:
    number = base.live.as_float(value)
    if number is None:
        return "n/a"
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    return f"{number / 1000:.1f}k"


def token_usage_rows_for_html(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition": row["condition"],
            "agent_rollouts": row["agent_rollouts"],
            "agent_input_mean": format_token_mean(row["agent_input_mean"]),
            "agent_output_mean": format_token_mean(row["agent_output_mean"]),
            "agent_total_mean": format_token_mean(row["agent_total_mean"]),
            "agent_total": format_token_total(row["agent_total"]),
            "privacy_judgments": row["privacy_judgments"],
            "eval_input_mean": format_token_mean(row["eval_input_mean"]),
            "eval_output_mean": format_token_mean(row["eval_output_mean"]),
            "eval_total_mean": format_token_mean(row["eval_total_mean"]),
            "eval_total": format_token_total(row["eval_total"]),
        }
        for row in rows
    ]


def token_usage_markdown(rows: list[dict[str, Any]], combined: dict[str, Any]) -> str:
    table_rows = token_usage_rows_for_html([*rows, combined])
    headers = [
        "Condition",
        "Agent rollouts",
        "Agent input / rollout",
        "Agent output / rollout",
        "Agent total / rollout",
        "Agent total",
        "Privacy judgments",
        "Adv+judge input / judgment",
        "Adv+judge output / judgment",
        "Adv+judge total / judgment",
        "Adv+judge total",
    ]
    keys = [
        "condition",
        "agent_rollouts",
        "agent_input_mean",
        "agent_output_mean",
        "agent_total_mean",
        "agent_total",
        "privacy_judgments",
        "eval_input_mean",
        "eval_output_mean",
        "eval_total_mean",
        "eval_total",
    ]
    lines = [
        "# Step100 token usage summary",
        "",
        "Agent token counts come from rollout `prompt_tokens` and `output_tokens`.",
        "Adversary + judging token counts sum the StepFun usage objects in each privacy-leakage judgment once, covering the full privacy evaluation suite for that rollout.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    lines.append("")
    lines.append(
        "The adversary + judging columns are per privacy judgment. "
        "The total columns aggregate the complete six-condition step100 slice shown here."
    )
    return "\n".join(lines) + "\n"


def drbench_score(row: dict[str, Any], metric: str) -> float | None:
    if metric == "score":
        return base.live.as_float(row.get("score"))
    scores = row.get("scores")
    if not isinstance(scores, dict):
        return None
    return base.live.as_float(scores.get(metric))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def sem(values: list[float]) -> float | None:
    std = sample_std(values)
    if std is None:
        return None
    return std / math.sqrt(len(values))


def drbench_run_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model_label, repeat, run_dir_name in DRBENCH_RUN_SPECS:
        results_path = DRBENCH_EVAL_ROOT / run_dir_name / "results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        successful = [
            item
            for item in results
            if item.get("status") == "success" and not item.get("error")
        ]
        run_row: dict[str, Any] = {
            "model_key": model_key,
            "model": model_label,
            "repeat": repeat,
            "run_dir": str(DRBENCH_EVAL_ROOT / run_dir_name),
            "n_tasks": len(results),
            "errors": len(results) - len(successful),
        }
        for metric, _label in DRBENCH_METRICS:
            values = [
                value
                for item in successful
                if (value := drbench_score(item, metric)) is not None
            ]
            run_row[metric] = mean(values)
        rows.append(run_row)
    return rows


def drbench_model_summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model_key, model_label in [("base", "Qwen3 4B"), ("privacy", "PA-DR step100")]:
        model_runs = [row for row in run_rows if row["model_key"] == model_key]
        item: dict[str, Any] = {
            "model_key": model_key,
            "model": model_label,
            "n_runs": len(model_runs),
            "n_tasks_per_run": model_runs[0]["n_tasks"] if model_runs else 0,
            "errors": sum(int(row.get("errors") or 0) for row in model_runs),
        }
        for metric, _label in DRBENCH_METRICS:
            values = [
                value
                for row in model_runs
                if (value := base.live.as_float(row.get(metric))) is not None
            ]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_sem"] = sem(values)
            item[f"{metric}_run_std"] = sample_std(values)
        out.append(item)
    return out


def drbench_delta_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {row["model_key"]: row for row in model_rows}
    base_row = by_key["base"]
    privacy_row = by_key["privacy"]
    rows: list[dict[str, Any]] = []
    for metric, label in DRBENCH_METRICS:
        base_value = base.live.as_float(base_row.get(f"{metric}_mean"))
        privacy_value = base.live.as_float(privacy_row.get(f"{metric}_mean"))
        delta = None if base_value is None or privacy_value is None else privacy_value - base_value
        rows.append(
            {
                "metric": label,
                "base": base_value,
                "privacy": privacy_value,
                "delta": delta,
            }
        )
    return rows


def format_score(value: Any) -> str:
    number = base.live.as_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def format_delta(value: Any) -> str:
    number = base.live.as_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.4f}"


def format_score_pm(mean_value: Any, sem_value: Any) -> str:
    score = base.live.as_float(mean_value)
    error = base.live.as_float(sem_value)
    if score is None:
        return "n/a"
    if error is None:
        return f"{score:.3f}"
    return f"{score:.3f} +/- {error:.3f}"


def drbench_summary_rows_for_html(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "model": row["model"],
            "runs": row["n_runs"],
            "tasks_per_run": row["n_tasks_per_run"],
            "errors": row["errors"],
        }
        for metric, _label in DRBENCH_METRICS:
            item[metric] = format_score_pm(row.get(f"{metric}_mean"), row.get(f"{metric}_sem"))
        out.append(item)
    return out


def drbench_run_rows_for_html(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "model": row["model"],
            "repeat": row["repeat"],
            "tasks": row["n_tasks"],
            "errors": row["errors"],
        }
        for metric, _label in DRBENCH_METRICS:
            item[metric] = format_score(row.get(metric))
        out.append(item)
    return out


def drbench_delta_rows_for_html(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "metric": row["metric"],
            "base": format_score(row.get("base")),
            "privacy": format_score(row.get("privacy")),
            "delta": format_delta(row.get("delta")),
        }
        for row in rows
    ]


def drbench_latex_summary(rows: list[dict[str, Any]]) -> str:
    def cell(row: dict[str, Any], metric: str) -> str:
        value = base.live.as_float(row.get(f"{metric}_mean"))
        error = base.live.as_float(row.get(f"{metric}_sem"))
        if value is None:
            return "--"
        if error is None:
            return f"{value:.3f}"
        return rf"{value:.3f} {{\scriptsize $\pm$ {error:.3f}}}"

    body = "\n".join(
        " & ".join(
            [
                row["model"],
                cell(row, "score"),
                cell(row, "insights_recall"),
                cell(row, "factuality"),
                cell(row, "distractor_avoidance"),
                cell(row, "report_quality"),
            ]
        )
        + r" \\"
        for row in rows
    )
    return rf"""\begin{{table}}[t]
\small
\centering
\begin{{tabular}}{{lccccc}}
\toprule
Model & Aggregate & Insight Recall & Factuality & Distractor Avoidance & Report Quality \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Original DRBench minival results for base \qwenthreeb and PA-DR step100. Scores average three stochastic repeats of the same 15-task minival slice; values after $\pm$ are SEM over repeat means. The aggregate is the DRBench harmonic score.}}
\label{{tab:drbench_minival_step100}}
\end{{table}}
"""


def build_viewer(output_dir: Path, latest_outputs: list[Path]) -> None:
    base.paper_style()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    summary = main_summary_rows(rows)
    checkpoints = checkpoint_rows(rows)
    behavior = behavior_rows(rows)
    status = status_rows(rows)
    query_examples = step100_query_example_rows(limit=3)
    token_rows, token_combined = token_usage_rows()
    drbench_runs = drbench_run_rows()
    drbench_summary = drbench_model_summary_rows(drbench_runs)
    drbench_delta = drbench_delta_rows(drbench_summary)
    current_viewer_data_path = output_dir.parent / "privacy_hopqa_q30_paper_viewer" / "viewer_data.json"
    untrained_rows = untrained_model_summary_rows(current_viewer_data_path)

    plot_training_effect(rows, plots_dir / "training_effect_step100.svg")
    plot_core_leakage_types(summary, plots_dir / "core_leakage_types_step100.svg")
    plot_behavior(behavior, plots_dir / "training_behavior_step100.svg")
    efficiency_output_dir = output_dir.parent / "privacy_hopqa_q30_training_efficiency_20260527"
    efficiency_payload = efficiency.build(efficiency_output_dir)
    efficiency_step100_table_html = (efficiency_output_dir / "training_efficiency_step100_summary.html").read_text(
        encoding="utf-8"
    )

    training_effect_latex = latex_training_summary(summary)
    untrained_latex = latex_untrained_model_summary(untrained_rows)
    behavior_latex = latex_behavior_summary(behavior)
    query_examples_latex = latex_step100_query_examples(query_examples)
    drbench_latex = drbench_latex_summary(drbench_summary)
    checkpoint_latex = base.latex_table(
        [
            ("condition", "Condition"),
            ("strict_chain", "Strict Chain Success"),
            ("final_accuracy", "Final Accuracy"),
            ("per_hop_accuracy", "Hop-Level Accuracy"),
            ("privacy_leakage", "Privacy Leakage"),
            ("intent", "Intent Leakage"),
            ("answer", "Answer Leakage"),
            ("full", "Full-Information Leakage"),
            ("judged", "Judged"),
            ("status", "Status"),
        ],
        table_rows_for_html(checkpoints, include_status=True),
        "Checkpoint choice diagnostic comparing step50 and step100. Values after +/- are SEMs in percentage points.",
        "tab:checkpoint_choice_s50_s100",
    )
    write_text(tables_dir / "training_effect_summary_s100.tex", training_effect_latex)
    write_text(tables_dir / "untrained_model_summary.tex", untrained_latex)
    write_text(
        tables_dir / "untrained_model_summary.json",
        json.dumps({"source": str(current_viewer_data_path), "rows": untrained_rows}, indent=2) + "\n",
    )
    write_text(tables_dir / "training_behavior_s100.tex", behavior_latex)
    write_text(tables_dir / "paired_leakage_examples_s100.tex", query_examples_latex)
    write_text(tables_dir / "paired_leakage_examples_s100.json", json.dumps(query_examples, indent=2) + "\n")
    write_text(tables_dir / "drbench_minival_step100_summary.tex", drbench_latex)
    write_text(
        tables_dir / "drbench_minival_step100_summary.json",
        json.dumps(
            {
                "source_root": str(DRBENCH_EVAL_ROOT),
                "metrics": DRBENCH_METRICS,
                "run_rows": drbench_runs,
                "model_summary_rows": drbench_summary,
                "delta_rows": drbench_delta,
            },
            indent=2,
        )
        + "\n",
    )
    write_text(tables_dir / "checkpoint_choice_s50_s100.tex", checkpoint_latex)
    write_text(tables_dir / "token_usage_summary_s100.md", token_usage_markdown(token_rows, token_combined))
    write_text(
        tables_dir / "token_usage_summary_s100.json",
        json.dumps({"rows": token_rows, "combined": token_combined}, indent=2) + "\n",
    )
    write_text(
        output_dir / "viewer_data.json",
        json.dumps(
            {
                "status_rows": status,
                "summary_rows": summary,
                "checkpoint_rows": checkpoints,
                "behavior_rows": behavior,
                "untrained_model_rows": untrained_rows,
                "token_usage_rows": token_rows,
                "token_usage_combined": token_combined,
                "drbench_minival": {
                    "source_root": str(DRBENCH_EVAL_ROOT),
                    "run_rows": drbench_runs,
                    "model_summary_rows": drbench_summary,
                    "delta_rows": drbench_delta,
                },
                "paired_leakage_example_rows": query_examples,
                "training_efficiency": {
                    "output_dir": str(efficiency_output_dir),
                    "summary_rows": efficiency_payload["summary_rows"],
                    "filter_bin_rows": efficiency_payload["filter_bin_rows"],
                    "step100_table_rows": efficiency_payload["step100_table_rows"],
                },
                "artifact_roots": {
                    "standard_normal": str(base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT),
                    "standard_privacy_prompt": str(base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT),
                    "padr_s50_normal": str(base.DEFAULT_PRIVACY_AWARE_FINAL_EVAL_ARTIFACT_ROOT),
                    "padr_s50_privacy_prompt": str(base.DEFAULT_PRIVACY_AWARE_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT),
                    "padr_s100_normal": str(PADR_S100_NORMAL_ROOT),
                    "padr_s100_privacy_prompt": str(PADR_S100_PROMPT_ROOT),
                },
            },
            indent=2,
        )
        + "\n",
    )

    status_table = base.table(
        [
            ("condition", "Condition"),
            ("attempts", "Attempts"),
            ("successful_rollouts", "Successful Rollouts"),
            ("privacy_judgments", "Privacy Judgments"),
            ("missing_privacy_judgments", "Missing Privacy Judgments"),
            ("status", "Status"),
        ],
        status,
    )
    summary_table = base.table(
        [
            ("condition", "Condition"),
            ("strict_chain", "Strict Chain Success"),
            ("per_hop_accuracy", "Hop-Level Accuracy"),
            ("privacy_leakage", "Privacy Leakage"),
            ("intent", "Intent Leakage"),
            ("answer", "Answer Leakage"),
            ("full", "Full-Information Leakage"),
        ],
        table_rows_for_html(summary),
    )
    untrained_table = base.table(
        [
            ("model", "Model"),
            ("strict_chain", "Strict Chain Success"),
            ("per_hop_accuracy", "Hop-Level Accuracy"),
            ("privacy_leakage", "Privacy Leakage"),
            ("intent", "Intent Leakage"),
            ("answer", "Answer Leakage"),
            ("full", "Full-Information Leakage"),
        ],
        untrained_model_rows_for_html(untrained_rows),
    )
    checkpoint_table = base.table(
        [
            ("condition", "Condition"),
            ("strict_chain", "Strict Chain Success"),
            ("final_accuracy", "Final Accuracy"),
            ("per_hop_accuracy", "Hop-Level Accuracy"),
            ("privacy_leakage", "Privacy Leakage"),
            ("intent", "Intent Leakage"),
            ("answer", "Answer Leakage"),
            ("full", "Full-Information Leakage"),
            ("judged", "Judged"),
            ("status", "Status"),
        ],
        table_rows_for_html(checkpoints, include_status=True),
    )
    behavior_table = base.table(
        [
            ("condition", "Condition"),
            ("web_queries", "Mean Web Queries"),
            ("web_reads", "Mean Web Reads"),
            ("local_searches", "Mean Local Searches"),
            ("local_reads", "Mean Local Reads"),
        ],
        behavior_rows_for_html(behavior),
    )
    token_usage_table = base.table(
        [
            ("condition", "Condition"),
            ("agent_rollouts", "Agent Rollouts"),
            ("agent_input_mean", "Agent Input / Rollout"),
            ("agent_output_mean", "Agent Output / Rollout"),
            ("agent_total_mean", "Agent Total / Rollout"),
            ("agent_total", "Agent Total"),
            ("privacy_judgments", "Privacy Judgments"),
            ("eval_input_mean", "Adv+Judge Input / Judgment"),
            ("eval_output_mean", "Adv+Judge Output / Judgment"),
            ("eval_total_mean", "Adv+Judge Total / Judgment"),
            ("eval_total", "Adv+Judge Total"),
        ],
        token_usage_rows_for_html([*token_rows, token_combined]),
    )
    drbench_summary_table = base.table(
        [
            ("model", "Model"),
            ("runs", "Runs"),
            ("tasks_per_run", "Tasks / Run"),
            ("errors", "Errors"),
            ("score", "Aggregate Score"),
            ("insights_recall", "Insight Recall"),
            ("factuality", "Factuality"),
            ("distractor_avoidance", "Distractor Avoidance"),
            ("report_quality", "Report Quality"),
        ],
        drbench_summary_rows_for_html(drbench_summary),
    )
    drbench_delta_table = base.table(
        [
            ("metric", "Metric"),
            ("base", "Qwen3 4B"),
            ("privacy", "PA-DR Step100"),
            ("delta", "PA-DR - Base"),
        ],
        drbench_delta_rows_for_html(drbench_delta),
    )
    drbench_run_table = base.table(
        [
            ("model", "Model"),
            ("repeat", "Repeat"),
            ("tasks", "Tasks"),
            ("errors", "Errors"),
            ("score", "Aggregate Score"),
            ("insights_recall", "Insight Recall"),
            ("factuality", "Factuality"),
            ("distractor_avoidance", "Distractor Avoidance"),
            ("report_quality", "Report Quality"),
        ],
        drbench_run_rows_for_html(drbench_runs),
    )
    training_effect_latex_block = latex_details("Copy LaTeX: Training Effect Summary", training_effect_latex)
    untrained_latex_block = latex_details("Copy LaTeX: Untrained Model Comparison", untrained_latex)
    behavior_latex_block = latex_details("Copy LaTeX: Query And Retrieval Behavior", behavior_latex)
    query_examples_latex_block = latex_details("Copy LaTeX: Step100 Query Examples", query_examples_latex)
    drbench_latex_block = latex_details("Copy LaTeX: Original DRBench Minival", drbench_latex)
    checkpoint_latex_block = latex_details("Copy LaTeX: Checkpoint Choice Diagnostic", checkpoint_latex)
    query_examples_table = step100_query_examples_html(query_examples)
    missing_by_condition = {
        str(row.get("condition")): int(row.get("missing_privacy_judgments") or 0)
        for row in status
    }
    total_missing = sum(missing_by_condition.values())
    if total_missing:
        padr_normal_missing = missing_by_condition.get("PA-DR step100", 0)
        padr_prompt_missing = missing_by_condition.get("PA-DR step100 + privacy prompt", 0)
        max_denominator_effect = 100.0 * max(missing_by_condition.values()) / 1032.0
        status_note = (
            f"Some StepFun privacy judgments are still pending. PA-DR step100 normal prompt is missing "
            f"{padr_normal_missing} and PA-DR step100 privacy prompt is missing {padr_prompt_missing}."
        )
        status_callout = (
            '<div class="warning">'
            + base.esc(
                "Use rows with pending judgments as near-final until the missing StepFun judgments are resolved "
                f"or intentionally counted as failures. The maximum denominator effect is about {max_denominator_effect:.1f} "
                "percentage points in this snapshot."
            )
            + "</div>"
        )
        training_effect_note = (
            "Step100 default comparison on held-out test rollouts. Error bars are SEM over repeat means. "
            "Asterisks mark near-final rows with missing StepFun privacy judgments."
        )
    else:
        status_note = "All step100 default rows are complete at 1032/1032 StepFun privacy judgments."
        status_callout = ""
        training_effect_note = "Step100 default comparison on held-out test rollouts. Error bars are SEM over repeat means."

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy HopQA Q30 Step100 Paper Viewer</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202522; background: #f7f8f6; }}
    header {{ padding: 28px 34px 16px; border-bottom: 1px solid #d8ddd7; background: #ffffff; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 22px 52px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.15; }}
    h2 {{ margin: 0 0 10px; font-size: 22px; }}
    p, .note {{ color: #52605a; line-height: 1.45; }}
    section {{ margin: 0 0 34px; padding: 22px; background: #ffffff; border: 1px solid #dfe4df; border-radius: 8px; }}
    img {{ width: 100%; max-width: 920px; display: block; margin: 14px auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #dfe4df; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f3f0; font-weight: 650; }}
    .muted {{ color: #6b7772; }}
    .rating {{ font-weight: 650; color: #303633; margin-bottom: 5px; }}
    .query-list {{ margin: 0; padding-left: 18px; }}
    .query-list li {{ margin: 0 0 4px; }}
    .paper-example-table {{ font-size: 12px; }}
    .paper-example-table th:nth-child(1), .paper-example-table td:nth-child(1) {{ width: 11%; }}
    .paper-example-table th:nth-child(2), .paper-example-table td:nth-child(2) {{ width: 23%; }}
    .asset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin: 14px 0; }}
    .asset-card {{ border: 1px solid #dfe4df; border-radius: 6px; padding: 12px; background: #fbfcfb; }}
    .asset-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .links a {{ display: inline-block; margin: 0 12px 8px 0; color: #0b63a3; text-decoration: none; }}
    .links a:hover {{ text-decoration: underline; }}
    details.latex-block {{ margin: 14px 0 8px; }}
    details.latex-block summary {{ cursor: pointer; color: #0b63a3; font-weight: 650; }}
    pre {{ overflow-x: auto; white-space: pre; background: #f4f6f4; border: 1px solid #dfe4df; border-radius: 6px; padding: 12px; font-size: 12px; line-height: 1.35; }}
    .warning {{ color: #755100; background: #fff7dd; border: 1px solid #ead392; padding: 10px 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Privacy HopQA Q30 Step100 Paper Viewer</h1>
    <div class="note">Slim paper-facing snapshot. The default trained checkpoints are Task step100 and PA-DR step100. {base.esc(status_note)}</div>
  </header>
  <main>
    <section id="status">
      <h2>Status</h2>
{status_callout}
      {status_table}
    </section>
    <section id="paper-assets">
      <h2>Paper Assets</h2>
      <div class="note">One-place index for the step100 plots, LaTeX tables, and source data generated by this viewer.</div>
      <div class="asset-grid">
        <div class="asset-card"><h3>Training Effect</h3><div class="links"><a href="plots/training_effect_step100.svg">SVG</a><a href="plots/training_effect_step100.pdf">PDF</a><a href="tables/training_effect_summary_s100.tex">LaTeX</a></div></div>
        <div class="asset-card"><h3>Untrained Models</h3><div class="links"><a href="#untrained-models">Viewer table</a><a href="tables/untrained_model_summary.tex">LaTeX</a><a href="tables/untrained_model_summary.json">JSON</a></div></div>
        <div class="asset-card"><h3>Training Trajectory</h3><div class="links"><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_accuracy_privacy_arrows.svg">SVG</a><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_accuracy_privacy_arrows.pdf">PDF</a><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_metrics.csv">CSV</a><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_metrics.json">JSON</a></div></div>
        <div class="asset-card"><h3>Training Efficiency</h3><div class="links"><a href="{EFFICIENCY_DIR.as_posix()}/training_sample_efficiency_metrics.svg">Metrics SVG</a><a href="{EFFICIENCY_DIR.as_posix()}/training_quality_threshold_samples.svg">Threshold SVG</a><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_step100_summary.tex">LaTeX table</a><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_summary.json">JSON</a></div></div>
        <div class="asset-card"><h3>Core Leakage Types</h3><div class="links"><a href="plots/core_leakage_types_step100.svg">SVG</a><a href="plots/core_leakage_types_step100.pdf">PDF</a></div></div>
        <div class="asset-card"><h3>Query And Retrieval Behavior</h3><div class="links"><a href="plots/training_behavior_step100.svg">SVG</a><a href="plots/training_behavior_step100.pdf">PDF</a><a href="tables/training_behavior_s100.tex">LaTeX</a></div></div>
        <div class="asset-card"><h3>Step100 Query Examples</h3><div class="links"><a href="#query-examples">Viewer table</a><a href="tables/paired_leakage_examples_s100.tex">LaTeX</a><a href="tables/paired_leakage_examples_s100.json">JSON</a></div></div>
        <div class="asset-card"><h3>Original DRBench Minival</h3><div class="links"><a href="#drbench-minival">Viewer table</a><a href="tables/drbench_minival_step100_summary.tex">LaTeX</a><a href="tables/drbench_minival_step100_summary.json">JSON</a></div></div>
        <div class="asset-card"><h3>Token Usage</h3><div class="links"><a href="#token-usage">Viewer table</a><a href="tables/token_usage_summary_s100.md">Markdown</a><a href="tables/token_usage_summary_s100.json">JSON</a></div></div>
        <div class="asset-card"><h3>Checkpoint Choice</h3><div class="links"><a href="tables/checkpoint_choice_s50_s100.tex">LaTeX</a><a href="viewer_data.json">JSON data</a></div></div>
      </div>
      {training_effect_latex_block}
      {untrained_latex_block}
      {behavior_latex_block}
      {query_examples_latex_block}
      {drbench_latex_block}
      {checkpoint_latex_block}
    </section>
    <section id="drbench-minival">
      <h2>Original DRBench Minival Sanity Check</h2>
      <div class="note">Original ServiceNow DRBench minival evaluation on 15 tasks with three stochastic repeats per model. Values after +/- are SEM over run means. Aggregate score is the top-level DRBench harmonic score, stored as <code>score</code> / <code>scores.harmonic_mean</code>.</div>
      <div class="links"><a href="tables/drbench_minival_step100_summary.tex">LaTeX table</a><a href="tables/drbench_minival_step100_summary.json">JSON data</a></div>
      {drbench_summary_table}
      <h3>Where PA-DR Step100 Loses Points</h3>
      <div class="note">The gap is not mostly distractor avoidance. Sub-scores are very close: PA-DR is slightly lower on report quality, distractor avoidance, and insight recall, and slightly higher on factuality. The aggregate gap is larger than any single sub-score delta because DRBench uses a nonlinear harmonic aggregate.</div>
      {drbench_delta_table}
      <h3>Repeat-Level Scores</h3>
      {drbench_run_table}
    </section>
    <section id="training-effect">
      <h2>Effect Of Training: Accuracy And Privacy</h2>
      <div class="note">{base.esc(training_effect_note)}</div>
      <img src="plots/training_effect_step100.svg" alt="Step100 training effect on strict chain success and privacy leakage">
      <div class="links"><a href="plots/training_effect_step100.svg">SVG</a><a href="plots/training_effect_step100.pdf">PDF</a><a href="tables/training_effect_summary_s100.tex">LaTeX table</a></div>
      {summary_table}
    </section>
    <section id="untrained-models">
      <h2>Untrained Model Comparison</h2>
      <div class="note">Held-out test comparison for the untrained model families, evaluated with and without the privacy-aware prompt. Values are SEM over repeat means where available; Chroma Context-1 uses the medium OpenRouter configuration.</div>
      <div class="links"><a href="tables/untrained_model_summary.tex">LaTeX table</a><a href="tables/untrained_model_summary.json">JSON rows</a></div>
      {untrained_table}
    </section>
    <section id="training-trajectory">
      <h2>Training Trajectory: Accuracy And Privacy</h2>
      <div class="note">Normal-prompt checkpoint trajectory on held-out test rollouts. Error bars are SEM over repeat means; color gradient encodes checkpoint step.</div>
      <img src="{TRAJECTORY_DIR.as_posix()}/training_trajectory_accuracy_privacy_arrows.svg" alt="Checkpoint training trajectory for strict chain success and privacy leakage">
      <div class="links"><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_accuracy_privacy_arrows.svg">SVG</a><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_accuracy_privacy_arrows.pdf">PDF</a><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_metrics.csv">CSV</a><a href="{TRAJECTORY_DIR.as_posix()}/training_trajectory_metrics.json">JSON</a></div>
    </section>
    <section id="training-efficiency">
      <h2>Training Efficiency: Samples And Filtering</h2>
      <div class="note">Normal-prompt checkpoint trajectories indexed by generated train samples on the surviving checkpoint path. These counts include filtered samples and exclude restart/prefetch waste, so the comparison focuses on sample/filtering efficiency rather than node count, queue state, or wall-clock effects.</div>
      <img src="{EFFICIENCY_DIR.as_posix()}/training_sample_efficiency_metrics.svg" alt="Checkpoint task and privacy metrics against generated train samples">
      <div class="links"><a href="{EFFICIENCY_DIR.as_posix()}/training_sample_efficiency_metrics.svg">SVG</a><a href="{EFFICIENCY_DIR.as_posix()}/training_sample_efficiency_metrics.pdf">PDF</a><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_checkpoint_metrics.csv">CSV</a></div>
      <h3>Step100 Sample Efficiency Summary</h3>
      {efficiency_step100_table_html}
      <div class="links"><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_step100_summary.csv">CSV</a><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_step100_summary.tex">LaTeX table</a><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_step100_summary.html">HTML table</a></div>
      <img src="{EFFICIENCY_DIR.as_posix()}/training_quality_threshold_samples.svg" alt="Generated samples needed to reach strict chain success targets">
      <div class="links"><a href="{EFFICIENCY_DIR.as_posix()}/training_quality_threshold_samples.svg">SVG</a><a href="{EFFICIENCY_DIR.as_posix()}/training_quality_threshold_samples.pdf">PDF</a></div>
      <img src="{EFFICIENCY_DIR.as_posix()}/training_optimizer_progress_vs_generation.svg" alt="Optimizer step against generated train samples">
      <div class="links"><a href="{EFFICIENCY_DIR.as_posix()}/training_optimizer_progress_vs_generation.svg">SVG</a><a href="{EFFICIENCY_DIR.as_posix()}/training_optimizer_progress_vs_generation.pdf">PDF</a></div>
      <img src="{EFFICIENCY_DIR.as_posix()}/training_filter_fraction_by_step.svg" alt="Filtered fraction over optimizer-step windows">
      <div class="links"><a href="{EFFICIENCY_DIR.as_posix()}/training_filter_fraction_by_step.svg">SVG</a><a href="{EFFICIENCY_DIR.as_posix()}/training_filter_fraction_by_step.pdf">PDF</a><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_filter_bins.csv">CSV</a></div>
      <img src="{EFFICIENCY_DIR.as_posix()}/training_generation_accounting_step100.svg" alt="Step100 generated train sample accounting">
      <div class="links"><a href="{EFFICIENCY_DIR.as_posix()}/training_generation_accounting_step100.svg">SVG</a><a href="{EFFICIENCY_DIR.as_posix()}/training_generation_accounting_step100.pdf">PDF</a><a href="{EFFICIENCY_DIR.as_posix()}/training_efficiency_summary.json">JSON</a></div>
    </section>
    <section id="core-leakage-types">
      <h2>Core Leakage Types</h2>
      <div class="note">Intent, answer, and Full-Information Leakage for the same step100 default conditions.</div>
      <img src="plots/core_leakage_types_step100.svg" alt="Step100 core leakage types">
      <div class="links"><a href="plots/core_leakage_types_step100.svg">SVG</a><a href="plots/core_leakage_types_step100.pdf">PDF</a></div>
    </section>
    <section id="training-behavior">
      <h2>Effect Of Training: Query And Retrieval Behavior</h2>
      <div class="note">Means per successful rollout for the same step100 default conditions.</div>
      <img src="plots/training_behavior_step100.svg" alt="Step100 training effect on query and retrieval behavior">
      <div class="links"><a href="plots/training_behavior_step100.svg">SVG</a><a href="plots/training_behavior_step100.pdf">PDF</a><a href="tables/training_behavior_s100.tex">LaTeX table</a></div>
      {behavior_table}
    </section>
    <section id="token-usage">
      <h2>Token Usage</h2>
      <div class="note">Agent columns are per successful rollout from rollout token accounting. Adversary + judging columns are per StepFun privacy judgment and include the full privacy-evaluation suite for that rollout: intent adversary/judge, answer adversary/judge when run, and claim adversary/judge when run. Total columns aggregate the complete six-condition step100 slice in this table.</div>
      <div class="links"><a href="tables/token_usage_summary_s100.md">Markdown summary</a><a href="tables/token_usage_summary_s100.json">JSON data</a></div>
      {token_usage_table}
    </section>
    <section id="query-examples">
      <h2>Step100 Query Leakage Examples</h2>
      <div class="note">Paper-prioritized held-out test cases using the final checkpoint defaults: base Qwen3 4B, Task step100, and PA-DR step100. Rows require strict chain success for all three rollouts, at least one web hop, base and Task privacy leakage, and no judged PA-DR leakage. Query cells show every web query made by that rollout.</div>
      <div class="links"><a href="tables/paired_leakage_examples_s100.tex">LaTeX table</a><a href="tables/paired_leakage_examples_s100.json">JSON rows</a></div>
      {query_examples_table}
    </section>
    <section id="checkpoint-choice">
      <h2>Checkpoint Choice: Step50 Vs Step100</h2>
      <div class="note">Compact diagnostic table for selecting the default checkpoint. Step100 is better for PA-DR privacy leakage while preserving roughly comparable task performance.</div>
      <div class="links"><a href="tables/checkpoint_choice_s50_s100.tex">LaTeX table</a><a href="viewer_data.json">JSON data</a></div>
      {checkpoint_table}
    </section>
  </main>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    write_text(index_path, html)
    for latest in latest_outputs:
        latest.parent.mkdir(parents=True, exist_ok=True)
        relative_output = Path(os.path.relpath(output_dir, latest.parent)).as_posix()
        latest_html = html.replace(
            "  <title>Privacy HopQA Q30 Step100 Paper Viewer</title>",
            f"  <title>Privacy HopQA Q30 Step100 Paper Viewer</title>\n  <base href=\"{base.esc(relative_output)}/\">",
            1,
        )
        write_text(latest, latest_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latest-output", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.latest_output:
        args.latest_output = [DEFAULT_LATEST_OUTPUT]
    return args


def main() -> None:
    args = parse_args()
    build_viewer(args.output_dir, args.latest_output)


if __name__ == "__main__":
    main()
