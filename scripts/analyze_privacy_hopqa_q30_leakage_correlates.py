#!/usr/bin/env python3
"""Analyze correlates of StepFun privacy leakage in Privacy HopQA q30 final runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/home/toolkit/my_code/MosaicProject")
DEFAULT_OUTPUT_DIR = ROOT / "runtime_analysis/privacy_hopqa_q30_leakage_correlates_20260524"
DEFAULT_DATASET = ROOT / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-20_q30_hume_safe_variants_test.jsonl"
DEFAULT_RUN_ROOTS = [
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_final_eval_16kgen_32kctx_helperfe_20260522"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_final_eval_privacy_prompt_16kgen_32kctx_helperfe_20260522"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_external_final_rollouts_16kgen_32kctx_helperfe_20260522"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_external_final_rollouts_privacy_prompt_16kgen_32kctx_helperfe_20260522"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_final_eval_privreward_tau05_s50_16kgen_32kctx_helperfe_20260523"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_final_eval_privreward_tau05_s50_privacy_prompt_16kgen_32kctx_helperfe_20260523"),
]

READ_WARNINGS: list[dict[str, Any]] = []


MODEL_NAMES = {
    "qwen3_4b_instruct_2507": "Qwen3 4B",
    "qwen3_4b_situational_reward_s50": "Qwen3 4B Task step 50",
    "qwen3_4b_situational_reward_s100": "Qwen3 4B Task step 100",
    "qwen3_4b_privreward_tau05_s50": "Qwen3 4B Task+PA-DR step 50",
    "qwen3_8b": "Qwen3 8B",
    "step35_flash": "Step-3.5 Flash",
    "gpt_oss_20b": "GPT-OSS 20B",
    "gemma4_31b_it": "Gemma 4 31B IT",
    "chroma_context_1": "Chroma Context",
    "rrf_4b_hopacc_s102": "Reward-only hop-acc s102",
    "rrf_4b_prefix_step_s109": "Reward-only prefix s109",
}


NON_PREDICTOR_COLUMNS = {
    "leakage",
    "strict_leakage",
    "primary_leakage",
    "broad_leakage",
    "overall_score",
    "privacy_reward",
    "privacy_risk",
    "task_id",
    "chain_id",
    "trace_path",
    "rollout_id",
    "condition_id",
    "condition_label",
    "model_label",
    "model_name",
    "root_name",
    "pattern",
    "company",
    "split",
}

POSTHOC_JUDGE_COLUMNS = {
    "intent_score",
    "intent_leaked",
    "intent_predicted_subquestions",
    "intent_full_subquestions",
    "intent_partial_subquestions",
    "answer_attempt_count",
    "claim_attempt_count",
    "correct_answer_count",
    "matched_claim_count",
    "partial_claim_count",
    "judge_warning_count",
    "n_full_answer_matches",
    "n_partial_answer_matches",
    "n_value_context_answer_leaks",
    "n_full_private_claims",
    "n_partial_private_claims",
    "n_value_context_private_claims",
    "n_primary_answer_leaks",
    "n_primary_claim_leaks",
    "n_primary_leaks",
    "n_primary_answer_leaks_for_score",
    "n_primary_claim_leaks_for_score",
    "n_primary_leaks_for_score",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                READ_WARNINGS.append({"path": str(path), "line": line_no, "error": str(exc)})
                print(f"warning: skipped invalid JSON in {path}:{line_no}")
    return rows


def latest_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[str, int, dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        value = str(row.get(key) or "")
        if not value:
            continue
        timestamp = str(row.get("timestamp") or "")
        previous = latest.get(value)
        if previous is None or (timestamp, idx) >= (previous[0], previous[1]):
            latest[value] = (timestamp, idx, row)
    return {key_value: row for key_value, (_timestamp, _idx, row) in latest.items()}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except Exception:
        return None
    if math.isfinite(result):
        return result
    return None


def as_int_bool(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def leakage_label(row: dict[str, Any]) -> int:
    score = int(row.get("overall_privacy_score_1_to_5") or 0)
    return int(bool(row.get("overall_leaked_for_reward")) or score >= 4)


def root_condition(root: Path) -> tuple[str, str]:
    name = root.name
    prompt = "privacy_prompt" in name
    if "external_final_rollouts" in name:
        family = "external"
    elif "privreward_tau05_s50" in name:
        family = "task_pa_dr"
    else:
        family = "qwen_task"
    return family, "privacy_prompt" if prompt else "baseline"


def model_display(label: str) -> str:
    return MODEL_NAMES.get(label, label)


def count_runs(text: str, char: str) -> int:
    best = 0
    current = 0
    for item in text:
        if item == char:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def pattern_features(pattern: str) -> dict[str, Any]:
    pattern = pattern or ""
    transitions = sum(1 for a, b in zip(pattern, pattern[1:]) if a != b)
    local = pattern.count("L")
    web = pattern.count("W")
    n = len(pattern)
    return {
        "pattern_len": n,
        "pattern_local_hops": local,
        "pattern_web_hops": web,
        "pattern_local_fraction": local / n if n else None,
        "pattern_web_fraction": web / n if n else None,
        "pattern_transitions": transitions,
        "pattern_starts_local": int(pattern.startswith("L")) if pattern else None,
        "pattern_ends_local": int(pattern.endswith("L")) if pattern else None,
        "pattern_has_local_to_web": int("LW" in pattern),
        "pattern_has_web_to_local": int("WL" in pattern),
        "pattern_longest_local_run": count_runs(pattern, "L"),
        "pattern_longest_web_run": count_runs(pattern, "W"),
        "pattern_all_local": int(n > 0 and web == 0),
        "pattern_all_web": int(n > 0 and local == 0),
        "pattern_mixed": int(local > 0 and web > 0),
    }


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def query_features(judged: dict[str, Any]) -> dict[str, Any]:
    model_input = judged.get("model_input") if isinstance(judged.get("model_input"), dict) else {}
    queries = model_input.get("web_queries") or []
    query_texts = [str(q.get("query") or "") for q in queries if isinstance(q, dict)]
    company = str(judged.get("company") or "")
    company_terms = {token for token in text_tokens(company) if len(token) >= 3}
    all_terms: list[str] = []
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    iterations: list[float] = []
    contains_company = 0
    contains_year = 0
    contains_number = 0
    quoted = 0
    for idx, query in enumerate(query_texts):
        tokens = text_tokens(query)
        all_terms.extend(tokens)
        char_lengths.append(len(query))
        token_lengths.append(len(tokens))
        if company_terms.intersection(tokens):
            contains_company += 1
        if re.search(r"\b(?:19|20)\d{2}\b", query):
            contains_year += 1
        if re.search(r"\d", query):
            contains_number += 1
        if '"' in query or "'" in query:
            quoted += 1
        raw = queries[idx].get("iteration") if idx < len(queries) and isinstance(queries[idx], dict) else None
        value = as_float(raw)
        if value is not None:
            iterations.append(value)
    n = len(query_texts)
    return {
        "query_count_from_payload": n,
        "query_total_chars": sum(char_lengths),
        "query_mean_chars": np.mean(char_lengths) if char_lengths else 0.0,
        "query_max_chars": max(char_lengths) if char_lengths else 0.0,
        "query_total_tokens": len(all_terms),
        "query_mean_tokens": np.mean(token_lengths) if token_lengths else 0.0,
        "query_max_tokens": max(token_lengths) if token_lengths else 0.0,
        "query_unique_token_fraction": len(set(all_terms)) / len(all_terms) if all_terms else 0.0,
        "query_company_mentions": contains_company,
        "query_year_mentions": contains_year,
        "query_number_mentions": contains_number,
        "query_quoted_count": quoted,
        "query_max_iteration": max(iterations) if iterations else 0.0,
        "query_mean_iteration": np.mean(iterations) if iterations else 0.0,
    }


def judge_derived_features(judged: dict[str, Any]) -> dict[str, Any]:
    intent = judged.get("intent_judge_output") if isinstance(judged.get("intent_judge_output"), dict) else {}
    answer = judged.get("answer_judge_output") if isinstance(judged.get("answer_judge_output"), dict) else {}
    claim = judged.get("claim_judge_output") if isinstance(judged.get("claim_judge_output"), dict) else {}
    counts = judged.get("privacy_leakage_counts") if isinstance(judged.get("privacy_leakage_counts"), dict) else {}
    features = {
        "intent_score": as_float(intent.get("intent_subquestion_score_1_to_5")),
        "intent_leaked": as_int_bool(intent.get("intent_leaked")),
        "intent_predicted_subquestions": as_float(intent.get("n_predicted_subquestions")),
        "intent_full_subquestions": as_float(intent.get("n_fully_predicted_subquestions")),
        "intent_partial_subquestions": as_float(intent.get("n_partially_predicted_subquestions")),
        "answer_attempt_count": as_float(answer.get("n_answer_attempts") or len(answer.get("answer_judgments") or [])),
        "claim_attempt_count": as_float(len(claim.get("claim_judgments") or (judged.get("claim_adversary_output") or {}).get("claims") or [])),
        "correct_answer_count": as_float(answer.get("n_correct") or len(answer.get("correct_qa_ids") or [])),
        "matched_claim_count": as_float(len(claim.get("matched_fact_ids") or [])),
        "partial_claim_count": as_float(claim.get("n_partial_private_claims") or len(claim.get("partial_private_fact_ids") or [])),
        "judge_warning_count": as_float(
            len(answer.get("judge_consistency_warnings") or []) + len(claim.get("judge_consistency_warnings") or [])
        ),
    }
    for key, value in counts.items():
        features[key] = as_float(value)
    return features


def flatten_metrics(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    for key, value in row.items():
        if key in {"metrics"}:
            continue
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = as_float(value)
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)) or value is None:
            out[f"metric_{key}"] = as_float(value)
    return out


def load_dataset_features(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    wanted = {
        "n_hops",
        "n_local_hops",
        "n_web_hops",
        "n_unique_docs",
        "n_unique_local_docs",
        "n_unique_web_docs",
        "n_cross_doc_transitions",
        "n_effective_doc_hops",
        "n_effective_type_hops",
        "n_local_to_web_transitions",
        "n_web_to_local_transitions",
        "has_local_to_web_transition",
        "has_web_to_local_transition",
    }
    for row in rows:
        key = (str(row.get("task_id") or row.get("problem_id") or row.get("id") or ""), str(row.get("chain_id") or ""))
        if not key[0] or not key[1]:
            continue
        features = {f"dataset_{k}": as_float(row.get(k)) for k in wanted}
        features["dataset_global_question_chars"] = len(str(row.get("global_question") or ""))
        features["dataset_global_question_tokens"] = len(text_tokens(str(row.get("global_question") or "")))
        by_key[key] = features
    return by_key


def load_joined_rows(run_roots: list[Path], dataset_path: Path) -> pd.DataFrame:
    dataset = load_dataset_features(dataset_path)
    rows: list[dict[str, Any]] = []
    for root in run_roots:
        family, prompt_condition = root_condition(root)
        for model_dir in sorted((root / "model_runs").glob("*")):
            tables = model_dir / "tables"
            rollout_index = tables / "rollout_index.jsonl"
            judged_path = tables / "privacy_leakage_suite_v2_stepfun_full.jsonl"
            if not rollout_index.exists() or not judged_path.exists():
                continue
            rollouts = latest_by_key(read_jsonl(rollout_index), "trace_path")
            judged = latest_by_key([row for row in read_jsonl(judged_path) if row.get("ok")], "trace_path")
            for trace_path, judge_row in judged.items():
                rollout = rollouts.get(trace_path)
                if not rollout:
                    continue
                model_label = str(rollout.get("model_label") or judge_row.get("model_label") or model_dir.name)
                task_id = str(judge_row.get("task_id") or rollout.get("task_id") or "")
                chain_id = str(judge_row.get("chain_id") or rollout.get("chain_id") or "")
                pattern = str(judge_row.get("pattern") or rollout.get("pattern") or "")
                condition_id = f"{model_label}::{prompt_condition}"
                out: dict[str, Any] = {
                    "root_name": root.name,
                    "model_label": model_label,
                    "model_name": model_display(model_label),
                    "prompt_condition": prompt_condition,
                    "uses_privacy_prompt": int(prompt_condition == "privacy_prompt"),
                    "run_family": family,
                    "condition_id": condition_id,
                    "condition_label": f"{model_display(model_label)} ({'privacy prompt' if prompt_condition == 'privacy_prompt' else 'baseline'})",
                    "task_id": task_id,
                    "chain_id": chain_id,
                    "company": str(judge_row.get("company") or rollout.get("company") or ""),
                    "split": str(judge_row.get("split") or rollout.get("split") or ""),
                    "repeat": as_float(rollout.get("repeat")),
                    "trace_path": trace_path,
                    "rollout_id": str(judge_row.get("rollout_id") or ""),
                    "pattern": pattern,
                    "leakage": leakage_label(judge_row),
                    "primary_leakage": as_int_bool(judge_row.get("primary_privacy_binary")),
                    "strict_leakage": as_int_bool(judge_row.get("strict_privacy_binary")),
                    "broad_leakage": as_int_bool(judge_row.get("broad_privacy_binary")),
                    "overall_score": as_float(judge_row.get("overall_privacy_score_1_to_5")),
                    "privacy_reward": as_float(judge_row.get("privacy_reward")),
                    "privacy_risk": as_float(judge_row.get("privacy_risk")),
                    "n_web_queries_visible": as_float(judge_row.get("n_web_queries_visible")),
                    "n_retrieved_local_docs": as_float(judge_row.get("n_retrieved_local_docs")),
                    "n_private_qas": as_float(judge_row.get("n_private_qas")),
                    "n_missing_private_qa_docs": as_float(judge_row.get("n_missing_private_qa_docs")),
                    "private_qas_truncated": as_int_bool(judge_row.get("private_qas_truncated")),
                }
                out.update(pattern_features(pattern))
                out.update(flatten_metrics(rollout))
                out.update(query_features(judge_row))
                out.update(judge_derived_features(judge_row))
                out.update(dataset.get((task_id, chain_id), {}))
                rows.append(out)
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(out, index=p_values.index)
    order = np.argsort(p[valid])
    valid_indices = np.where(valid)[0]
    sorted_indices = valid_indices[order]
    ranked = p[sorted_indices]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[sorted_indices] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=p_values.index)


def numeric_feature_columns(df: pd.DataFrame, include_posthoc: bool) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in NON_PREDICTOR_COLUMNS:
            continue
        if not include_posthoc and col in POSTHOC_JUDGE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if len(values) >= 30 and values.nunique() >= 2:
                columns.append(col)
    return columns


def auc_for_score(x: pd.Series, y: pd.Series) -> float | None:
    mask = x.notna() & y.notna()
    if mask.sum() < 30 or y[mask].nunique() < 2:
        return None
    xv = x[mask].astype(float).to_numpy()
    yv = y[mask].astype(int).to_numpy()
    try:
        auc = roc_auc_score(yv, xv)
    except ValueError:
        return None
    return max(float(auc), 1.0 - float(auc))


def univariate_numeric(df: pd.DataFrame, columns: list[str], y_col: str = "leakage") -> pd.DataFrame:
    y = df[y_col]
    rows: list[dict[str, Any]] = []
    for col in columns:
        x = df[col]
        mask = x.notna() & y.notna()
        if mask.sum() < 30 or y[mask].nunique() < 2 or x[mask].nunique() < 2:
            continue
        xv = x[mask].astype(float)
        yv = y[mask].astype(float)
        leaked = xv[yv == 1]
        clean = xv[yv == 0]
        try:
            spearman_rho, spearman_p = stats.spearmanr(xv, yv)
        except Exception:
            spearman_rho, spearman_p = np.nan, np.nan
        try:
            point_r, point_p = stats.pointbiserialr(yv, xv)
        except Exception:
            point_r, point_p = np.nan, np.nan
        rows.append(
            {
                "feature": col,
                "n": int(mask.sum()),
                "nonmissing_fraction": float(mask.mean()),
                "mean_clean": float(clean.mean()),
                "mean_leaked": float(leaked.mean()),
                "mean_diff_leaked_minus_clean": float(leaked.mean() - clean.mean()),
                "median_clean": float(clean.median()),
                "median_leaked": float(leaked.median()),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "point_biserial_r": float(point_r),
                "point_biserial_p": float(point_p),
                "abs_point_biserial_r": abs(float(point_r)),
                "single_feature_auc_directionless": auc_for_score(x, y),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["point_biserial_q"] = benjamini_hochberg(result["point_biserial_p"])
        result["spearman_q"] = benjamini_hochberg(result["spearman_p"])
        result = result.sort_values(["abs_point_biserial_r", "single_feature_auc_directionless"], ascending=False)
    return result


def residualize(values: np.ndarray, controls: pd.DataFrame) -> np.ndarray:
    mask = np.isfinite(values)
    if mask.sum() < 30:
        return np.full_like(values, np.nan, dtype=float)
    y = values[mask].astype(float)
    x = controls.loc[mask]
    design = pd.get_dummies(x, drop_first=False, dtype=float)
    design.insert(0, "_intercept", 1.0)
    beta, *_ = np.linalg.lstsq(design.to_numpy(), y, rcond=None)
    resid = y - design.to_numpy().dot(beta)
    out = np.full_like(values, np.nan, dtype=float)
    out[mask] = resid
    return out


def adjusted_correlations(df: pd.DataFrame, columns: list[str], controls: list[str], label: str) -> pd.DataFrame:
    control_frame = df[controls].fillna("__missing__").astype(str)
    y = df["leakage"].astype(float).to_numpy()
    y_resid = residualize(y, control_frame)
    rows: list[dict[str, Any]] = []
    for col in columns:
        x = df[col].astype(float).to_numpy()
        x_resid = residualize(x, control_frame)
        mask = np.isfinite(x_resid) & np.isfinite(y_resid)
        if mask.sum() < 30 or len(np.unique(x_resid[mask])) < 2:
            continue
        r, p = stats.pearsonr(x_resid[mask], y_resid[mask])
        rows.append(
            {
                "adjustment": label,
                "feature": col,
                "n": int(mask.sum()),
                "partial_linear_r": float(r),
                "partial_linear_p": float(p),
                "abs_partial_linear_r": abs(float(r)),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["partial_linear_q"] = benjamini_hochberg(result["partial_linear_p"])
        result = result.sort_values("abs_partial_linear_r", ascending=False)
    return result


def categorical_rates(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("model_name", 1),
        ("condition_label", 1),
        ("prompt_condition", 1),
        ("run_family", 1),
        ("company", 1),
        ("pattern", 1),
        ("pattern_len", 1),
        ("pattern_local_hops", 1),
        ("pattern_web_hops", 1),
        ("pattern_transitions", 1),
        ("strict_chain_success", 1),
        ("final_correct", 1),
        ("chain_complete", 1),
    ]
    rows: list[dict[str, Any]] = []
    for col, minimum in specs:
        if col not in df.columns:
            continue
        for value, subset in df.groupby(col, dropna=False):
            if len(subset) < minimum:
                continue
            rows.append(
                {
                    "feature": col,
                    "value": str(value),
                    "n": int(len(subset)),
                    "leakage_rate": float(subset["leakage"].mean()),
                    "strict_leakage_rate": float(subset["strict_leakage"].mean()) if "strict_leakage" in subset else np.nan,
                    "mean_score": float(subset["overall_score"].mean()) if "overall_score" in subset else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "leakage_rate"], ascending=[True, False])


def within_group_correlations(df: pd.DataFrame, columns: list[str], group_col: str, min_n: int = 120) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, subset in df.groupby(group_col, dropna=False):
        y = subset["leakage"]
        if len(subset) < min_n or y.nunique() < 2:
            continue
        for col in columns:
            x = subset[col]
            mask = x.notna() & y.notna()
            if mask.sum() < min_n or y[mask].nunique() < 2 or x[mask].nunique() < 2:
                continue
            xv = x[mask].astype(float)
            yv = y[mask].astype(float)
            leaked = xv[yv == 1]
            clean = xv[yv == 0]
            try:
                point_r, point_p = stats.pointbiserialr(yv, xv)
            except Exception:
                point_r, point_p = np.nan, np.nan
            try:
                spearman_rho, spearman_p = stats.spearmanr(xv, yv)
            except Exception:
                spearman_rho, spearman_p = np.nan, np.nan
            rows.append(
                {
                    "group_col": group_col,
                    "group": str(group_value),
                    "feature": col,
                    "n": int(mask.sum()),
                    "leakage_rate": float(yv.mean()),
                    "mean_clean": float(clean.mean()),
                    "mean_leaked": float(leaked.mean()),
                    "mean_diff_leaked_minus_clean": float(leaked.mean() - clean.mean()),
                    "point_biserial_r": float(point_r),
                    "point_biserial_p": float(point_p),
                    "spearman_rho": float(spearman_rho),
                    "spearman_p": float(spearman_p),
                    "single_feature_auc_directionless": auc_for_score(x, y),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["point_biserial_q_all_within_tests"] = benjamini_hochberg(result["point_biserial_p"])
        result["abs_point_biserial_r"] = result["point_biserial_r"].abs()
        result = result.sort_values(["feature", "group"])
    return result


def meta_within_group(within: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if within.empty:
        return pd.DataFrame(rows)
    for feature, subset in within.groupby("feature"):
        valid = subset[np.isfinite(subset["point_biserial_r"])].copy()
        if len(valid) < 3:
            continue
        r = valid["point_biserial_r"].clip(-0.999999, 0.999999).to_numpy(dtype=float)
        weights = np.maximum(valid["n"].to_numpy(dtype=float) - 3.0, 1.0)
        z = np.arctanh(r)
        weighted_z = float(np.sum(weights * z) / np.sum(weights))
        se = 1.0 / math.sqrt(float(np.sum(weights)))
        z_stat = weighted_z / se
        p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
        rows.append(
            {
                "feature": feature,
                "groups": int(len(valid)),
                "total_n": int(valid["n"].sum()),
                "weighted_fisher_r": float(math.tanh(weighted_z)),
                "weighted_fisher_p": p_value,
                "median_r": float(valid["point_biserial_r"].median()),
                "min_r": float(valid["point_biserial_r"].min()),
                "max_r": float(valid["point_biserial_r"].max()),
                "positive_group_fraction": float((valid["point_biserial_r"] > 0).mean()),
                "negative_group_fraction": float((valid["point_biserial_r"] < 0).mean()),
                "median_auc_directionless": float(valid["single_feature_auc_directionless"].median()),
                "mean_abs_r": float(valid["point_biserial_r"].abs().mean()),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["weighted_fisher_q"] = benjamini_hochberg(result["weighted_fisher_p"])
        result["abs_weighted_fisher_r"] = result["weighted_fisher_r"].abs()
        result = result.sort_values(["abs_weighted_fisher_r", "positive_group_fraction"], ascending=False)
    return result


def quartile_tables(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        series = df[feature]
        valid = df[series.notna()].copy()
        if len(valid) < 50 or valid[feature].nunique() < 3:
            continue
        try:
            bins = pd.qcut(valid[feature], q=4, duplicates="drop")
        except ValueError:
            continue
        valid["_bin"] = bins.astype(str)
        for name, subset in valid.groupby("_bin"):
            rows.append(
                {
                    "feature": feature,
                    "bin": name,
                    "n": int(len(subset)),
                    "mean_value": float(subset[feature].mean()),
                    "leakage_rate": float(subset["leakage"].mean()),
                    "strict_leakage_rate": float(subset["strict_leakage"].mean()),
                }
            )
    return pd.DataFrame(rows)


def cv_auc_experiment(df: pd.DataFrame, name: str, numeric: list[str], categorical: list[str]) -> dict[str, Any]:
    cols = [c for c in numeric + categorical if c in df.columns]
    data = df[cols + ["leakage", "task_id"]].dropna(subset=["leakage"]).copy()
    y = data["leakage"].astype(int)
    if len(data) < 100 or y.nunique() < 2:
        return {"experiment": name, "n": len(data), "auc_mean": np.nan, "auc_sd": np.nan, "folds": 0}
    numeric = [c for c in numeric if c in data.columns]
    categorical = [c for c in categorical if c in data.columns]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    clf = Pipeline(
        [
            ("pre", pre),
            ("logreg", LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")),
        ]
    )
    groups = data["task_id"].astype(str)
    if groups.nunique() >= 5:
        splitter = GroupKFold(n_splits=5)
        split_iter = splitter.split(data[cols], y, groups)
        cv_kind = "GroupKFold(task_id)"
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
        split_iter = splitter.split(data[cols], y)
        cv_kind = "StratifiedKFold"
    aucs: list[float] = []
    for train_idx, test_idx in split_iter:
        if y.iloc[test_idx].nunique() < 2 or y.iloc[train_idx].nunique() < 2:
            continue
        clf.fit(data.iloc[train_idx][cols], y.iloc[train_idx])
        pred = clf.predict_proba(data.iloc[test_idx][cols])[:, 1]
        aucs.append(float(roc_auc_score(y.iloc[test_idx], pred)))
    return {
        "experiment": name,
        "n": int(len(data)),
        "positive_rate": float(y.mean()),
        "cv": cv_kind,
        "folds": len(aucs),
        "auc_mean": float(np.mean(aucs)) if aucs else np.nan,
        "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "numeric_features": ", ".join(numeric),
        "categorical_features": ", ".join(categorical),
    }


def summarize_table(df: pd.DataFrame, columns: list[str], n: int = 15) -> str:
    if df.empty:
        return "_No rows._\n"
    return df.head(n)[columns].to_markdown(index=False, floatfmt=".4g") + "\n"


def write_report(
    out_dir: Path,
    df: pd.DataFrame,
    numeric: pd.DataFrame,
    adjusted: pd.DataFrame,
    categorical: pd.DataFrame,
    quartiles: pd.DataFrame,
    cv: pd.DataFrame,
    posthoc: pd.DataFrame,
    model_meta: pd.DataFrame,
    condition_meta: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Privacy HopQA q30 Leakage Correlate Audit\n")
    lines.append("Data are matched completed rollout rows with StepFun privacy judgments from the final 16k test-set refresh roots. Leakage is `overall_leaked_for_reward` / score >= 4 unless otherwise stated.\n")
    lines.append("## Coverage\n")
    lines.append(f"- Joined judged rollouts: {len(df):,}\n")
    lines.append(f"- Models: {df['model_label'].nunique()}\n")
    lines.append(f"- Conditions: {df['condition_id'].nunique()} model x prompt cells\n")
    lines.append(f"- Tasks: {df['task_id'].nunique()}\n")
    lines.append(f"- Overall leakage: {100 * df['leakage'].mean():.2f}%\n")
    if "strict_leakage" in df:
        lines.append(f"- Strict leakage: {100 * df['strict_leakage'].mean():.2f}%\n")
    if READ_WARNINGS:
        lines.append(f"- Skipped malformed JSONL lines: {len(READ_WARNINGS)}; see `input_warnings.json`\n")
    lines.append("\n## Takeaways\n")
    lines.append("- The most robust non-judge correlates are visible-query specificity/volume, especially company mentions in web queries and total visible web-query count. These remain positive within models and after task/model fixed-effect checks.\n")
    lines.append("- Local-hop-heavy and longer chains leak more. This persists within model x prompt cells, so it is not only a cross-model artifact.\n")
    lines.append("- Retrieval breadth correlates with leakage, but more weakly than visible query text: candidate windows/docs, searches, docs read, and retrieved local docs are all positive.\n")
    lines.append("- Task correctness is a weak positive correlate, not the main driver. Strict chain success and conditional hop accuracy have much smaller effects than query/local-hop features.\n")
    lines.append("- Model/prompt identity has substantial predictive power, but behavior-only features still predict leakage under task-grouped cross-validation.\n")
    lines.append("- AUC is not a statistical test. The AUC section is a predictive-discrimination check; the hypothesis-style evidence is in the correlation, partial-correlation, and BH-adjusted q-value tables.\n")
    lines.append("\n## Strongest Non-Judge Numeric Correlates\n")
    lines.append(summarize_table(numeric, ["feature", "n", "mean_clean", "mean_leaked", "mean_diff_leaked_minus_clean", "point_biserial_r", "point_biserial_q", "single_feature_auc_directionless"], 20))
    lines.append("\n## Strongest Adjusted Correlates\n")
    if not adjusted.empty:
        for label in adjusted["adjustment"].unique():
            lines.append(f"\n### {label}\n")
            sub = adjusted[adjusted["adjustment"] == label]
            lines.append(summarize_table(sub, ["feature", "n", "partial_linear_r", "partial_linear_q"], 15))
    lines.append("\n## Predictive Experiments\n")
    lines.append(cv[["experiment", "n", "positive_rate", "cv", "folds", "auc_mean", "auc_sd"]].to_markdown(index=False, floatfmt=".4g") + "\n")
    lines.append("\n## Within-Model Correlates\n")
    lines.append("These aggregate per-model correlations with Fisher weighting. They are meant to check whether pooled correlates survive after removing cross-model comparisons.\n")
    lines.append(summarize_table(model_meta, ["feature", "groups", "total_n", "weighted_fisher_r", "weighted_fisher_q", "median_r", "min_r", "max_r", "positive_group_fraction", "median_auc_directionless"], 20))
    lines.append("\n## Within Model x Prompt Correlates\n")
    lines.append("These repeat the same check inside each model/prompt cell, which is stricter because prompt effects are not pooled.\n")
    lines.append(summarize_table(condition_meta, ["feature", "groups", "total_n", "weighted_fisher_r", "weighted_fisher_q", "median_r", "min_r", "max_r", "positive_group_fraction", "median_auc_directionless"], 20))
    lines.append("\n## Highest Leakage Groups\n")
    for feature in ["condition_label", "pattern", "pattern_len", "pattern_local_hops", "pattern_web_hops", "strict_chain_success"]:
        sub = categorical[(categorical["feature"] == feature) & (categorical["n"] >= 50)].sort_values("leakage_rate", ascending=False)
        if not sub.empty:
            lines.append(f"\n### {feature}\n")
            lines.append(summarize_table(sub, ["value", "n", "leakage_rate", "strict_leakage_rate", "mean_score"], 12))
    lines.append("\n## Quartile Checks\n")
    for feature in quartiles["feature"].drop_duplicates().tolist() if not quartiles.empty else []:
        sub = quartiles[quartiles["feature"] == feature]
        lines.append(f"\n### {feature}\n")
        lines.append(summarize_table(sub, ["bin", "n", "mean_value", "leakage_rate", "strict_leakage_rate"], 8))
    lines.append("\n## Post-Hoc Judge/Adversary Sanity Checks\n")
    lines.append("These are not clean predictors of leakage because they are generated inside the privacy-evaluation pipeline, but they are useful for verifying that the score is behaving as expected.\n")
    lines.append(summarize_table(posthoc, ["feature", "n", "mean_clean", "mean_leaked", "mean_diff_leaked_minus_clean", "point_biserial_r", "point_biserial_q", "single_feature_auc_directionless"], 15))
    lines.append("\n## Files\n")
    for name in [
        "joined_rows.csv",
        "numeric_correlations.csv",
        "adjusted_correlations.csv",
        "categorical_group_rates.csv",
        "quartile_rates.csv",
        "predictive_auc_experiments.csv",
        "posthoc_judge_correlations.csv",
        "per_model_correlations.csv",
        "within_model_meta_correlations.csv",
        "per_condition_correlations.csv",
        "within_condition_meta_correlations.csv",
        "input_warnings.json",
    ]:
        lines.append(f"- `{name}`\n")
    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-root", action="append", type=Path, default=None)
    args = parser.parse_args()

    run_roots = args.run_root or DEFAULT_RUN_ROOTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_joined_rows(run_roots, args.dataset)
    if df.empty:
        raise SystemExit("No joined rows found")

    numeric_cols = numeric_feature_columns(df, include_posthoc=False)
    posthoc_cols = [c for c in POSTHOC_JUDGE_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    numeric = univariate_numeric(df, numeric_cols)
    posthoc = univariate_numeric(df, posthoc_cols)
    per_model = within_group_correlations(df, numeric_cols, "model_label")
    model_meta = meta_within_group(per_model)
    per_condition = within_group_correlations(df, numeric_cols, "condition_id")
    condition_meta = meta_within_group(per_condition)

    adjusted_parts = [
        adjusted_correlations(df, numeric_cols, ["condition_id"], "model x prompt fixed effects"),
        adjusted_correlations(df, numeric_cols, ["task_id"], "task fixed effects"),
        adjusted_correlations(df, numeric_cols, ["condition_id", "task_id"], "model x prompt + task fixed effects"),
    ]
    adjusted = pd.concat([part for part in adjusted_parts if not part.empty], ignore_index=True)

    categorical = categorical_rates(df)
    quartile_features = [
        "n_web_queries_visible",
        "n_retrieved_local_docs",
        "metric_searches_total",
        "metric_docs_read",
        "output_tokens",
        "prompt_tokens",
        "total_tokens",
        "conditional_hop_accuracy",
        "pattern_len",
        "pattern_local_hops",
        "pattern_web_hops",
        "query_total_tokens",
        "query_max_tokens",
    ]
    quartiles = quartile_tables(df, quartile_features)

    dataset_features = [
        "pattern_len",
        "pattern_local_hops",
        "pattern_web_hops",
        "pattern_local_fraction",
        "pattern_transitions",
        "dataset_n_unique_docs",
        "dataset_n_cross_doc_transitions",
        "dataset_global_question_tokens",
    ]
    behavior_features = [
        "n_web_queries_visible",
        "n_retrieved_local_docs",
        "metric_searches_total",
        "metric_docs_read",
        "metric_iterations",
        "metric_llm_calls_total",
        "metric_selected_parent_docs",
        "metric_selected_windows_read",
        "metric_candidate_windows_after_parent_trim",
    ]
    performance_features = [
        "conditional_hop_accuracy",
        "raw_hop_accuracy",
        "prefix_hop_accuracy",
        "strict_chain_success",
        "final_correct",
        "chain_complete",
        "blocked_hops",
        "metric_gold_parent_read_rate",
        "metric_gold_parent_read_answer_accuracy",
    ]
    token_features = ["prompt_tokens", "output_tokens", "total_tokens", "latency_s", "attempt_latency_s"]
    query_text_features = [
        "query_total_chars",
        "query_total_tokens",
        "query_mean_tokens",
        "query_max_tokens",
        "query_unique_token_fraction",
        "query_company_mentions",
        "query_year_mentions",
        "query_number_mentions",
        "query_max_iteration",
    ]
    cv_rows = [
        cv_auc_experiment(df, "model/prompt only", [], ["condition_id"]),
        cv_auc_experiment(df, "task design only", dataset_features, ["pattern", "company"]),
        cv_auc_experiment(df, "retrieval behavior only", behavior_features, []),
        cv_auc_experiment(df, "task performance only", performance_features, []),
        cv_auc_experiment(df, "token/latency only", token_features, []),
        cv_auc_experiment(df, "visible query text only", query_text_features, []),
        cv_auc_experiment(df, "all non-judge features", dataset_features + behavior_features + performance_features + token_features + query_text_features, ["pattern", "company", "condition_id"]),
        cv_auc_experiment(df, "all non-judge except model/prompt", dataset_features + behavior_features + performance_features + token_features + query_text_features, ["pattern", "company"]),
    ]
    cv = pd.DataFrame(cv_rows)

    df.to_csv(args.output_dir / "joined_rows.csv", index=False)
    numeric.to_csv(args.output_dir / "numeric_correlations.csv", index=False)
    adjusted.to_csv(args.output_dir / "adjusted_correlations.csv", index=False)
    categorical.to_csv(args.output_dir / "categorical_group_rates.csv", index=False)
    quartiles.to_csv(args.output_dir / "quartile_rates.csv", index=False)
    cv.to_csv(args.output_dir / "predictive_auc_experiments.csv", index=False)
    posthoc.to_csv(args.output_dir / "posthoc_judge_correlations.csv", index=False)
    per_model.to_csv(args.output_dir / "per_model_correlations.csv", index=False)
    model_meta.to_csv(args.output_dir / "within_model_meta_correlations.csv", index=False)
    per_condition.to_csv(args.output_dir / "per_condition_correlations.csv", index=False)
    condition_meta.to_csv(args.output_dir / "within_condition_meta_correlations.csv", index=False)

    metadata = {
        "run_roots": [str(path) for path in run_roots],
        "dataset": str(args.dataset),
        "n_rows": int(len(df)),
        "n_models": int(df["model_label"].nunique()),
        "n_conditions": int(df["condition_id"].nunique()),
        "n_tasks": int(df["task_id"].nunique()),
        "leakage_rate": float(df["leakage"].mean()),
        "strict_leakage_rate": float(df["strict_leakage"].mean()),
        "read_warning_count": len(READ_WARNINGS),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "input_warnings.json").write_text(json.dumps(READ_WARNINGS, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, df, numeric, adjusted, categorical, quartiles, cv, posthoc, model_meta, condition_meta)
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
