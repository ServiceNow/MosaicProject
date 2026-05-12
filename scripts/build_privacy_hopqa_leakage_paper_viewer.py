#!/usr/bin/env python3
"""Build a paper-oriented leakage analysis viewer with static SVG plots."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


PALETTE = {
    "ink": "#1f2421",
    "muted": "#65706b",
    "line": "#d7d0c2",
    "panel": "#fffdf8",
    "bg": "#f6f3ec",
    "blue": "#315c9c",
    "green": "#14705e",
    "gold": "#b67812",
    "red": "#9f3428",
    "gray": "#8b918c",
    "lavender": "#7764a8",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("task_id") or ""), str(row.get("chain_id") or ""), str(row.get("rollout_id") or ""))


def discover_rollout_index(path: Path) -> Path | None:
    candidate = path.parent / "rollout_index.jsonl"
    return candidate if candidate.exists() else None


def load_rollout_metrics(path: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        trace_path = str(row.get("trace_path") or "")
        if trace_path:
            out[trace_path] = row
    return out


def metric_value(rollout_metrics: dict[str, dict[str, Any]], row: dict[str, Any], name: str) -> Any:
    metrics_row = rollout_metrics.get(str(row.get("trace_path") or "")) or {}
    metrics = metrics_row.get("metrics") if isinstance(metrics_row.get("metrics"), dict) else {}
    return metrics.get(name, metrics_row.get(name))


def intent_score(row: dict[str, Any]) -> int:
    obj = row.get("intent_judge_output") if isinstance(row.get("intent_judge_output"), dict) else {}
    return int(obj.get("intent_subquestion_score_1_to_5") or 0)


def answer_leaked(row: dict[str, Any]) -> bool:
    obj = row.get("answer_judge_output") if isinstance(row.get("answer_judge_output"), dict) else {}
    return bool(obj.get("answer_leaked"))


def full_info_leaked(row: dict[str, Any]) -> bool:
    obj = row.get("claim_judge_output") if isinstance(row.get("claim_judge_output"), dict) else {}
    return bool(obj.get("full_info_leaked"))


def score(row: dict[str, Any]) -> int:
    return int(row.get("overall_privacy_score_1_to_5") or 0)


def reward(row: dict[str, Any]) -> bool:
    return bool(row.get("overall_leaked_for_reward") or score(row) >= 4)


def web_queries(row: dict[str, Any]) -> list[str]:
    model_input = row.get("model_input") if isinstance(row.get("model_input"), dict) else {}
    out: list[str] = []
    for item in model_input.get("web_queries") or []:
        if isinstance(item, dict):
            query = str(item.get("query") or "").strip()
        else:
            query = str(item or "").strip()
        if query:
            out.append(query)
    return out


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(xs: list[float]) -> list[float]:
    pairs = sorted((x, idx) for idx, x in enumerate(xs))
    out = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + 1 + j) / 2
        for _, idx in pairs[i:j]:
            out[idx] = rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys)) if xs and len(xs) == len(ys) else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bool_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(target and pred for target, pred in pairs)
    tn = sum((not target) and (not pred) for target, pred in pairs)
    fp = sum((not target) and pred for target, pred in pairs)
    fn = sum(target and (not pred) for target, pred in pairs)
    n = len(pairs)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else None
    p_yes_target = (tp + fn) / n if n else 0.0
    p_yes_pred = (tp + fp) / n if n else 0.0
    pe = p_yes_target * p_yes_pred + (1 - p_yes_target) * (1 - p_yes_pred)
    kappa = (accuracy - pe) / (1 - pe) if n and pe < 1 else None
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "cohen_kappa": kappa,
    }


def color_for_fraction(frac: float, color: str = "blue") -> str:
    frac = max(0.0, min(1.0, frac))
    if color == "red":
        base = (159, 52, 40)
    elif color == "green":
        base = (20, 112, 94)
    elif color == "gold":
        base = (182, 120, 18)
    else:
        base = (49, 92, 156)
    bg = (255, 253, 248)
    mix = 0.16 + 0.78 * frac
    rgb = tuple(round(bg[i] * (1 - mix) + base[i] * mix) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img'>{body}</svg>"
    )


def chart_title(title: str, subtitle: str | None = None) -> str:
    subtitle_svg = f"<text x='28' y='48' font-size='13' fill='{PALETTE['muted']}'>{esc(subtitle)}</text>" if subtitle else ""
    return f"<text x='28' y='28' font-size='18' font-weight='700' fill='{PALETTE['ink']}'>{esc(title)}</text>{subtitle_svg}"


def bar_chart(title: str, items: list[tuple[str, float]], *, subtitle: str | None = None, y_label: str = "count", color: str = "blue") -> str:
    width, height = 760, 430
    ml, mr, mt, mb = 66, 26, 74, 76
    plot_w, plot_h = width - ml - mr, height - mt - mb
    max_value = max([value for _, value in items] + [1])
    tick_count = 4
    body = [f"<rect width='{width}' height='{height}' fill='{PALETTE['panel']}'/>", chart_title(title, subtitle)]
    for i in range(tick_count + 1):
        value = max_value * i / tick_count
        y = mt + plot_h - (value / max_value) * plot_h
        body.append(f"<line x1='{ml}' y1='{y:.1f}' x2='{width - mr}' y2='{y:.1f}' stroke='{PALETTE['line']}' stroke-width='1'/>")
        body.append(f"<text x='{ml - 10}' y='{y + 4:.1f}' text-anchor='end' font-size='11' fill='{PALETTE['muted']}'>{fmt(value, 1)}</text>")
    n = max(1, len(items))
    slot = plot_w / n
    bar_w = min(74, slot * 0.62)
    for idx, (label, value) in enumerate(items):
        x = ml + idx * slot + (slot - bar_w) / 2
        h = (value / max_value) * plot_h if max_value else 0
        y = mt + plot_h - h
        body.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' rx='4' fill='{PALETTE[color]}'/>")
        body.append(f"<text x='{x + bar_w / 2:.1f}' y='{y - 8:.1f}' text-anchor='middle' font-size='12' font-weight='700' fill='{PALETTE['ink']}'>{fmt(value, 2)}</text>")
        body.append(f"<text x='{x + bar_w / 2:.1f}' y='{height - 46}' text-anchor='middle' font-size='12' fill='{PALETTE['ink']}'>{esc(label)}</text>")
    body.append(f"<text x='22' y='{mt + plot_h / 2:.1f}' transform='rotate(-90 22 {mt + plot_h / 2:.1f})' text-anchor='middle' font-size='12' fill='{PALETTE['muted']}'>{esc(y_label)}</text>")
    return svg_wrap(width, height, "".join(body))


def grouped_bar_chart(title: str, groups: list[str], series: list[tuple[str, list[float], str]], *, subtitle: str | None = None) -> str:
    width, height = 800, 440
    ml, mr, mt, mb = 64, 28, 82, 78
    plot_w, plot_h = width - ml - mr, height - mt - mb
    max_value = max([value for _, values, _ in series for value in values] + [1])
    body = [f"<rect width='{width}' height='{height}' fill='{PALETTE['panel']}'/>", chart_title(title, subtitle)]
    for idx, (name, _, color) in enumerate(series):
        lx = width - mr - 180 + idx * 88
        body.append(f"<rect x='{lx}' y='23' width='14' height='14' rx='2' fill='{PALETTE[color]}'/>")
        body.append(f"<text x='{lx + 20}' y='35' font-size='12' fill='{PALETTE['ink']}'>{esc(name)}</text>")
    for i in range(5):
        value = max_value * i / 4
        y = mt + plot_h - (value / max_value) * plot_h
        body.append(f"<line x1='{ml}' y1='{y:.1f}' x2='{width - mr}' y2='{y:.1f}' stroke='{PALETTE['line']}'/>")
        body.append(f"<text x='{ml - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='11' fill='{PALETTE['muted']}'>{fmt(value, 1)}</text>")
    slot = plot_w / max(1, len(groups))
    bar_w = min(34, slot * 0.25)
    for gi, group in enumerate(groups):
        center = ml + gi * slot + slot / 2
        for si, (_, values, color) in enumerate(series):
            x = center - (len(series) * bar_w) / 2 + si * bar_w
            value = values[gi] if gi < len(values) else 0
            h = (value / max_value) * plot_h if max_value else 0
            y = mt + plot_h - h
            body.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w - 2:.1f}' height='{h:.1f}' rx='3' fill='{PALETTE[color]}'/>")
        body.append(f"<text x='{center:.1f}' y='{height - 46}' text-anchor='middle' font-size='12' fill='{PALETTE['ink']}'>{esc(group)}</text>")
    return svg_wrap(width, height, "".join(body))


def heatmap(title: str, matrix: list[list[int]], x_labels: list[str], y_labels: list[str], *, subtitle: str | None = None, color: str = "blue") -> str:
    width, height = 720, 560
    ml, mr, mt, mb = 120, 34, 84, 74
    cells_x = len(x_labels)
    cells_y = len(y_labels)
    cell = min((width - ml - mr) / max(1, cells_x), (height - mt - mb) / max(1, cells_y))
    max_value = max([value for row in matrix for value in row] + [1])
    body = [f"<rect width='{width}' height='{height}' fill='{PALETTE['panel']}'/>", chart_title(title, subtitle)]
    for xi, label in enumerate(x_labels):
        x = ml + xi * cell + cell / 2
        body.append(f"<text x='{x:.1f}' y='{mt - 14}' text-anchor='middle' font-size='12' fill='{PALETTE['ink']}'>{esc(label)}</text>")
    for yi, label in enumerate(y_labels):
        y = mt + yi * cell + cell / 2
        body.append(f"<text x='{ml - 12}' y='{y + 4:.1f}' text-anchor='end' font-size='12' fill='{PALETTE['ink']}'>{esc(label)}</text>")
    for yi, row in enumerate(matrix):
        for xi, value in enumerate(row):
            frac = value / max_value if max_value else 0
            x = ml + xi * cell
            y = mt + yi * cell
            fill = color_for_fraction(frac, color)
            body.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell - 2:.1f}' height='{cell - 2:.1f}' rx='4' fill='{fill}' stroke='{PALETTE['line']}'/>")
            body.append(f"<text x='{x + cell / 2:.1f}' y='{y + cell / 2 + 5:.1f}' text-anchor='middle' font-size='16' font-weight='700' fill='{PALETTE['ink']}'>{value}</text>")
    body.append(f"<text x='{ml + cells_x * cell / 2:.1f}' y='{height - 24}' text-anchor='middle' font-size='12' fill='{PALETTE['muted']}'>candidate / prediction</text>")
    body.append(f"<text x='28' y='{mt + cells_y * cell / 2:.1f}' transform='rotate(-90 28 {mt + cells_y * cell / 2:.1f})' text-anchor='middle' font-size='12' fill='{PALETTE['muted']}'>StepFun reference</text>")
    return svg_wrap(width, height, "".join(body))


def scatter_chart(title: str, points: list[dict[str, Any]], *, x_key: str, y_key: str, x_label: str, y_label: str, subtitle: str | None = None) -> str:
    width, height = 780, 460
    ml, mr, mt, mb = 72, 36, 76, 70
    plot_w, plot_h = width - ml - mr, height - mt - mb
    xs = [float(p[x_key]) for p in points if p.get(x_key) is not None and p.get(y_key) is not None]
    ys = [float(p[y_key]) for p in points if p.get(x_key) is not None and p.get(y_key) is not None]
    x_min, x_max = (min(xs), max(xs)) if xs else (0.0, 1.0)
    y_min, y_max = (min(ys), max(ys)) if ys else (0.0, 1.0)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return ml + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return mt + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    body = [f"<rect width='{width}' height='{height}' fill='{PALETTE['panel']}'/>", chart_title(title, subtitle)]
    for i in range(5):
        x_value = x_min + (x_max - x_min) * i / 4
        x = sx(x_value)
        body.append(f"<line x1='{x:.1f}' y1='{mt}' x2='{x:.1f}' y2='{mt + plot_h}' stroke='{PALETTE['line']}'/>")
        body.append(f"<text x='{x:.1f}' y='{height - 42}' text-anchor='middle' font-size='11' fill='{PALETTE['muted']}'>{fmt(x_value, 2)}</text>")
        y_value = y_min + (y_max - y_min) * i / 4
        y = sy(y_value)
        body.append(f"<line x1='{ml}' y1='{y:.1f}' x2='{ml + plot_w}' y2='{y:.1f}' stroke='{PALETTE['line']}'/>")
        body.append(f"<text x='{ml - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='11' fill='{PALETTE['muted']}'>{fmt(y_value, 2)}</text>")
    for idx, point in enumerate(points):
        if point.get(x_key) is None or point.get(y_key) is None:
            continue
        jitter_x = ((idx * 37) % 11 - 5) * 0.9
        jitter_y = ((idx * 53) % 11 - 5) * 0.9
        fill = PALETTE["red"] if point.get("reward") else PALETTE["blue"]
        stroke = PALETTE["ink"] if point.get("final_correct") else "none"
        body.append(
            f"<circle cx='{sx(float(point[x_key])) + jitter_x:.1f}' cy='{sy(float(point[y_key])) + jitter_y:.1f}' "
            f"r='5.2' fill='{fill}' fill-opacity='0.72' stroke='{stroke}' stroke-width='1'/>"
        )
    body.append(f"<text x='{ml + plot_w / 2:.1f}' y='{height - 16}' text-anchor='middle' font-size='12' fill='{PALETTE['muted']}'>{esc(x_label)}</text>")
    body.append(f"<text x='24' y='{mt + plot_h / 2:.1f}' transform='rotate(-90 24 {mt + plot_h / 2:.1f})' text-anchor='middle' font-size='12' fill='{PALETTE['muted']}'>{esc(y_label)}</text>")
    body.append(f"<circle cx='{width - 196}' cy='34' r='5' fill='{PALETTE['red']}' fill-opacity='0.72'/><text x='{width - 184}' y='38' font-size='12' fill='{PALETTE['ink']}'>StepFun score >= 4</text>")
    body.append(f"<circle cx='{width - 196}' cy='54' r='5' fill='{PALETTE['blue']}' fill-opacity='0.72'/><text x='{width - 184}' y='58' font-size='12' fill='{PALETTE['ink']}'>score <= 3</text>")
    return svg_wrap(width, height, "".join(body))


def table(headers: list[tuple[str, str]], rows: list[dict[str, Any]], *, limit: int | None = None) -> str:
    shown = rows[:limit] if limit else rows
    if not shown:
        return "<p class='empty'>No rows.</p>"
    head = "".join(f"<th>{esc(label)}</th>" for _, label in headers)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{esc(row.get(key, ''))}</td>" for key, _ in headers) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_analysis(args: argparse.Namespace) -> None:
    stepfun_path = Path(args.stepfun)
    qwen_path = Path(args.qwen) if args.qwen else None
    binary_path = Path(args.binary) if args.binary else None
    rollout_index = Path(args.rollout_index) if args.rollout_index else discover_rollout_index(stepfun_path)
    rollout_metrics = load_rollout_metrics(rollout_index)

    stepfun_rows = {key(row): row for row in read_jsonl(stepfun_path) if row.get("ok")}
    qwen_rows = {key(row): row for row in read_jsonl(qwen_path) if row.get("ok")}
    binary_rows = {key(row): row for row in read_jsonl(binary_path) if row.get("ok")}
    keys = sorted(stepfun_rows)
    matched_qwen = sorted(set(stepfun_rows) & set(qwen_rows))
    matched_binary = sorted(set(stepfun_rows) & set(binary_rows))

    feature_rows: list[dict[str, Any]] = []
    for k in keys:
        row = stepfun_rows[k]
        hop_accuracy = metric_value(rollout_metrics, row, "hop_accuracy")
        final_correct = metric_value(rollout_metrics, row, "final_correct")
        feature_rows.append(
            {
                "task_id": k[0],
                "chain_id": k[1],
                "rollout_id": k[2],
                "score": score(row),
                "reward": reward(row),
                "intent_score": intent_score(row),
                "answer_leaked": answer_leaked(row),
                "full_info_leaked": full_info_leaked(row),
                "n_web_queries": int(row.get("n_web_queries_visible") or len(web_queries(row))),
                "n_private_qas": int(row.get("n_private_qas") or 0),
                "hop_accuracy": None if hop_accuracy is None else float(hop_accuracy),
                "final_correct": final_correct is True,
                "source": row.get("overall_primary_source"),
                "web_queries": " | ".join(web_queries(row)),
            }
        )

    step_scores = [score(stepfun_rows[k]) for k in keys]
    qwen_scores = [score(qwen_rows[k]) for k in matched_qwen]
    step_qwen_scores = [score(stepfun_rows[k]) for k in matched_qwen]
    step_intent = [intent_score(stepfun_rows[k]) for k in matched_qwen]
    qwen_intent = [intent_score(qwen_rows[k]) for k in matched_qwen]

    qwen_reward_metrics = bool_metrics([(reward(stepfun_rows[k]), reward(qwen_rows[k])) for k in matched_qwen]) if matched_qwen else {}
    binary_metrics = bool_metrics([(reward(stepfun_rows[k]), bool(binary_rows[k].get("predicted"))) for k in matched_binary]) if matched_binary else {}
    answer_metrics = bool_metrics([(answer_leaked(stepfun_rows[k]), answer_leaked(qwen_rows[k])) for k in matched_qwen]) if matched_qwen else {}
    full_metrics = bool_metrics([(full_info_leaked(stepfun_rows[k]), full_info_leaked(qwen_rows[k])) for k in matched_qwen]) if matched_qwen else {}
    intent_metrics = bool_metrics([(intent_score(stepfun_rows[k]) >= 4, intent_score(qwen_rows[k]) >= 4) for k in matched_qwen]) if matched_qwen else {}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stepfun": str(stepfun_path),
        "qwen": str(qwen_path or ""),
        "binary": str(binary_path or ""),
        "rollout_index": str(rollout_index or ""),
        "rows": len(keys),
        "matched_qwen": len(matched_qwen),
        "matched_binary": len(matched_binary),
        "stepfun_score_counts": dict(sorted(Counter(step_scores).items())),
        "stepfun_source_counts": dict(sorted(Counter(str(stepfun_rows[k].get("overall_primary_source")) for k in keys).items())),
        "qwen_score_counts": dict(sorted(Counter(qwen_scores).items())) if matched_qwen else {},
        "stepfun_reward_positive": sum(reward(stepfun_rows[k]) for k in keys),
        "qwen_reward_positive": sum(reward(qwen_rows[k]) for k in matched_qwen) if matched_qwen else None,
        "binary_predicted_positive": sum(bool(binary_rows[k].get("predicted")) for k in matched_binary) if matched_binary else None,
        "qwen_reward_metrics": qwen_reward_metrics,
        "binary_metrics": binary_metrics,
        "intent_metrics": intent_metrics,
        "answer_metrics": answer_metrics,
        "full_info_metrics": full_metrics,
        "overall_score_pearson": pearson(step_qwen_scores, qwen_scores) if matched_qwen else None,
        "overall_score_spearman": spearman(step_qwen_scores, qwen_scores) if matched_qwen else None,
        "intent_score_pearson": pearson(step_intent, qwen_intent) if matched_qwen else None,
        "intent_score_spearman": spearman(step_intent, qwen_intent) if matched_qwen else None,
        "feature_correlations_with_stepfun_reward": {
            "n_web_queries": pearson([row["n_web_queries"] for row in feature_rows], [int(row["reward"]) for row in feature_rows]),
            "n_private_qas": pearson([row["n_private_qas"] for row in feature_rows], [int(row["reward"]) for row in feature_rows]),
            "hop_accuracy": pearson(
                [row["hop_accuracy"] for row in feature_rows if row["hop_accuracy"] is not None],
                [int(row["reward"]) for row in feature_rows if row["hop_accuracy"] is not None],
            ),
        },
    }

    out = Path(args.output_dir)
    plots_dir = out / "plots"
    tables_dir = out / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scores = [1, 2, 3, 4, 5]
    step_score_counts = Counter(step_scores)
    qwen_score_counts = Counter(qwen_scores)
    plots: dict[str, str] = {}
    plots["stepfun_score_distribution"] = bar_chart(
        "StepFun Privacy Score Distribution",
        [(str(s), step_score_counts.get(s, 0)) for s in scores],
        subtitle=f"n={len(keys)} spread-sampled rollouts; reward positive is score >= 4",
        y_label="rollouts",
        color="blue",
    )
    plots["leakage_type_rates"] = bar_chart(
        "StepFun Leakage Rates",
        [
            ("Intent", sum(intent_score(stepfun_rows[k]) >= 4 for k in keys)),
            ("Answer", sum(answer_leaked(stepfun_rows[k]) for k in keys)),
            ("Full-info", sum(full_info_leaked(stepfun_rows[k]) for k in keys)),
            ("Reward+", sum(reward(stepfun_rows[k]) for k in keys)),
        ],
        subtitle="Counts by leakage type; categories are not mutually exclusive",
        y_label="rollouts",
        color="red",
    )
    if matched_qwen:
        plots["score_distribution_stepfun_vs_qwen"] = grouped_bar_chart(
            "Score Distribution: StepFun vs Qwen3-4B",
            [str(s) for s in scores],
            [
                ("StepFun", [step_score_counts.get(s, 0) for s in scores], "blue"),
                ("Qwen3-4B", [qwen_score_counts.get(s, 0) for s in scores], "gold"),
            ],
            subtitle="Same rollouts, same six-call leakage harness",
        )
        score_matrix = [[0 for _ in scores] for _ in scores]
        for k in matched_qwen:
            sx = score(stepfun_rows[k])
            qx = score(qwen_rows[k])
            if sx in scores and qx in scores:
                score_matrix[sx - 1][qx - 1] += 1
        plots["stepfun_qwen_score_heatmap"] = heatmap(
            "StepFun vs Qwen3-4B Overall Scores",
            score_matrix,
            [str(s) for s in scores],
            [str(s) for s in scores],
            subtitle=f"Pearson {fmt(summary['overall_score_pearson'])}; Spearman {fmt(summary['overall_score_spearman'])}",
            color="blue",
        )
        qwen_matrix = [
            [qwen_reward_metrics.get("tn", 0), qwen_reward_metrics.get("fp", 0)],
            [qwen_reward_metrics.get("fn", 0), qwen_reward_metrics.get("tp", 0)],
        ]
        plots["qwen_reward_confusion"] = heatmap(
            "Qwen Full-Judge Reward Confusion",
            qwen_matrix,
            ["Pred -", "Pred +"],
            ["StepFun -", "StepFun +"],
            subtitle=f"F1 {fmt(qwen_reward_metrics.get('f1'))}; kappa {fmt(qwen_reward_metrics.get('cohen_kappa'))}",
            color="green",
        )
    if matched_binary:
        binary_matrix = [
            [binary_metrics.get("tn", 0), binary_metrics.get("fp", 0)],
            [binary_metrics.get("fn", 0), binary_metrics.get("tp", 0)],
        ]
        plots["binary_reward_confusion"] = heatmap(
            "Qwen Web-Only Binary Predictor Confusion",
            binary_matrix,
            ["Pred -", "Pred +"],
            ["StepFun -", "StepFun +"],
            subtitle=f"F1 {fmt(binary_metrics.get('f1'))}; MCC {fmt(binary_metrics.get('mcc'))}; kappa {fmt(binary_metrics.get('cohen_kappa'))}",
            color="red",
        )
    plots["score_vs_hop_accuracy"] = scatter_chart(
        "Privacy Score vs Rollout Hop Accuracy",
        feature_rows,
        x_key="hop_accuracy",
        y_key="score",
        x_label="hop accuracy",
        y_label="StepFun privacy score",
        subtitle=f"Pearson with reward {fmt(summary['feature_correlations_with_stepfun_reward']['hop_accuracy'])}",
    )
    plots["score_vs_web_queries"] = scatter_chart(
        "Privacy Score vs Visible Web Queries",
        feature_rows,
        x_key="n_web_queries",
        y_key="score",
        x_label="visible web-query count",
        y_label="StepFun privacy score",
        subtitle=f"Pearson with reward {fmt(summary['feature_correlations_with_stepfun_reward']['n_web_queries'])}",
    )
    plots["score_vs_private_qas"] = scatter_chart(
        "Privacy Score vs Retrieved Private QA Count",
        feature_rows,
        x_key="n_private_qas",
        y_key="score",
        x_label="retrieved-doc Private QA questions",
        y_label="StepFun privacy score",
        subtitle=f"Pearson with reward {fmt(summary['feature_correlations_with_stepfun_reward']['n_private_qas'])}",
    )

    for name, svg in plots.items():
        (plots_dir / f"{name}.svg").write_text(svg, encoding="utf-8")

    positive_rows = []
    disagreement_rows = []
    chain_rows_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        chain_rows_map[(row["task_id"], row["chain_id"])].append(row)
    for k in keys:
        step = stepfun_rows[k]
        qwen = qwen_rows.get(k)
        binary = binary_rows.get(k)
        base = {
            "task_id": k[0],
            "chain_id": k[1],
            "rollout_id": k[2],
            "stepfun_score": score(step),
            "stepfun_source": step.get("overall_primary_source"),
            "stepfun_reward": int(reward(step)),
            "qwen_score": "" if qwen is None else score(qwen),
            "qwen_source": "" if qwen is None else qwen.get("overall_primary_source"),
            "qwen_reward": "" if qwen is None else int(reward(qwen)),
            "binary_pred": "" if binary is None else int(bool(binary.get("predicted"))),
            "binary_correct": "" if binary is None else bool(binary.get("correct")),
            "web_queries": " | ".join(web_queries(step)),
        }
        if reward(step):
            positive_rows.append(base)
        if (qwen is not None and reward(step) != reward(qwen)) or (binary is not None and reward(step) != bool(binary.get("predicted"))):
            disagreement_rows.append(base)

    chain_rows = []
    for (task_id, chain_id), vals in sorted(chain_rows_map.items()):
        acc = [row["hop_accuracy"] for row in vals if row["hop_accuracy"] is not None]
        chain_rows.append(
            {
                "task_id": task_id,
                "chain_id": chain_id,
                "n": len(vals),
                "reward_pos": sum(row["reward"] for row in vals),
                "max_score": max(row["score"] for row in vals),
                "mean_score": round(mean(row["score"] for row in vals), 3),
                "mean_hop_accuracy": "" if not acc else round(mean(acc), 3),
                "final_correct": sum(row["final_correct"] for row in vals),
            }
        )
    chain_rows.sort(key=lambda row: (-int(row["reward_pos"]), -int(row["max_score"]), str(row["task_id"]), str(row["chain_id"])))

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(tables_dir / "positive_cases.jsonl", positive_rows)
    write_jsonl(tables_dir / "disagreements.jsonl", disagreement_rows)
    write_jsonl(tables_dir / "chain_summary.jsonl", chain_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "plot_data.json").write_text(json.dumps({"features": feature_rows, "positive_cases": positive_rows, "disagreements": disagreement_rows}, indent=2) + "\n", encoding="utf-8")

    metric_rows = [
        {"metric": "Rows", "value": len(keys)},
        {"metric": "StepFun reward positives", "value": f"{summary['stepfun_reward_positive']} / {len(keys)}"},
        {"metric": "Qwen full-judge positives", "value": summary.get("qwen_reward_positive")},
        {"metric": "Binary predictor positives", "value": summary.get("binary_predicted_positive")},
        {"metric": "Qwen reward F1", "value": fmt(qwen_reward_metrics.get("f1")) if qwen_reward_metrics else "n/a"},
        {"metric": "Qwen reward kappa", "value": fmt(qwen_reward_metrics.get("cohen_kappa")) if qwen_reward_metrics else "n/a"},
        {"metric": "Binary predictor F1", "value": fmt(binary_metrics.get("f1")) if binary_metrics else "n/a"},
        {"metric": "Binary predictor MCC", "value": fmt(binary_metrics.get("mcc")) if binary_metrics else "n/a"},
        {"metric": "Score Pearson: StepFun vs Qwen", "value": fmt(summary.get("overall_score_pearson"))},
        {"metric": "Score Spearman: StepFun vs Qwen", "value": fmt(summary.get("overall_score_spearman"))},
    ]

    plot_cards = "".join(
        f"<section class='plot-card'><div class='plot-head'><h3>{esc(name.replace('_', ' ').title())}</h3>"
        f"<a href='plots/{esc(name)}.svg'>SVG</a></div>{svg}</section>"
        for name, svg in plots.items()
    )
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(args.title)}</title>
  <style>
    :root{{--bg:{PALETTE['bg']};--panel:{PALETTE['panel']};--ink:{PALETTE['ink']};--muted:{PALETTE['muted']};--line:{PALETTE['line']};--blue:{PALETTE['blue']};}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}.page{{max-width:1720px;margin:0 auto;padding:24px}}
    h1{{font-size:32px;margin:0 0 8px}}h2{{font-size:20px;margin:0 0 12px}}h3{{font-size:16px;margin:0}}p{{color:var(--muted)}}code{{font-size:12px}}
    .hero,.panel,.plot-card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:0 8px 22px rgba(0,0,0,.045)}}.hero p{{margin:4px 0}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px;margin-top:14px}}.plot-card svg{{width:100%;height:auto;display:block}}.plot-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}.plot-head a{{color:var(--blue);font-weight:700;text-decoration:none}}
    .panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px;margin-top:14px}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}}th{{font-size:12px;color:#344}}td{{font-size:13px}}.scroll{{max-height:520px;overflow:auto}}.empty{{color:var(--muted);font-style:italic}}
    .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}}.card{{border:1px solid var(--line);border-radius:8px;background:#f3efe6;padding:12px}}.label{{font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.08em}}.value{{font-size:24px;font-weight:750}}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>{esc(args.title)}</h1>
      <p>Generated {esc(datetime.now(timezone.utc).isoformat())}</p>
      <p>StepFun reference: <code>{esc(stepfun_path)}</code></p>
      <p>Qwen candidate: <code>{esc(qwen_path or 'not provided')}</code></p>
      <p>Binary predictor: <code>{esc(binary_path or 'not provided')}</code></p>
      <p>Each plot is also written as a standalone SVG under <code>{esc(plots_dir)}</code>.</p>
    </section>
    <section class="cards">
      <div class="card"><div class="label">Rows</div><div class="value">{len(keys)}</div></div>
      <div class="card"><div class="label">StepFun Reward+</div><div class="value">{summary['stepfun_reward_positive']}</div></div>
      <div class="card"><div class="label">Qwen Reward F1</div><div class="value">{fmt(qwen_reward_metrics.get('f1')) if qwen_reward_metrics else 'n/a'}</div></div>
      <div class="card"><div class="label">Binary F1</div><div class="value">{fmt(binary_metrics.get('f1')) if binary_metrics else 'n/a'}</div></div>
    </section>
    <section class="grid">{plot_cards}</section>
    <section class="panels">
      <div class="panel"><h2>Correlation And Agreement Summary</h2>{table([('metric','Metric'),('value','Value')], metric_rows)}</div>
      <div class="panel"><h2>StepFun Positive Cases</h2><div class="scroll">{table([('rollout_id','Rollout'),('stepfun_score','Score'),('stepfun_source','Source'),('qwen_score','Qwen'),('binary_pred','Binary'),('web_queries','Web queries')], positive_rows)}</div></div>
      <div class="panel"><h2>Reward Disagreements</h2><div class="scroll">{table([('rollout_id','Rollout'),('stepfun_score','StepFun'),('qwen_score','Qwen'),('binary_pred','Binary'),('web_queries','Web queries')], disagreement_rows, limit=80)}</div></div>
      <div class="panel"><h2>Chain Summary</h2><div class="scroll">{table([('task_id','Task'),('chain_id','Chain'),('n','Rows'),('reward_pos','Reward+'),('max_score','Max score'),('mean_score','Mean score'),('mean_hop_accuracy','Mean hop acc'),('final_correct','Final correct')], chain_rows, limit=120)}</div></div>
    </section>
  </main>
</body>
</html>
"""
    (out / "index.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps({"output": str(out / "index.html"), "summary": summary, "plots": sorted(plots)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepfun", required=True, help="StepFun leakage v2 JSONL")
    parser.add_argument("--qwen", default=None, help="Qwen leakage v2 JSONL over the same rollouts")
    parser.add_argument("--binary", default=None, help="Binary predictor JSONL over the same StepFun labels")
    parser.add_argument("--rollout-index", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Privacy HopQA Leakage Analysis")
    args = parser.parse_args()
    build_analysis(args)


if __name__ == "__main__":
    main()
