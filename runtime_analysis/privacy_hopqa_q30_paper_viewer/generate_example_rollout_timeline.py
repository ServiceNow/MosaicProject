#!/usr/bin/env python3
"""Generate a paper-ready example rollout timeline for Privacy HopQA."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"
TRACE_DIR = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_4b_situational_privreward_tau05_lr1e6_rep8_200s_20260522191644/"
    "privacy_hopqa_runs/DR0001_035c3d89_001b9f03"
)

STAGES = [
    ("hop_plan", "Plan"),
    ("search", "Retrieve"),
    ("doc_choose", "Select docs"),
    ("doc_read", "Read docs"),
    ("hop_resolve", "Resolve"),
]
STAGE_LABEL = dict(STAGES)
STAGE_COLORS = {
    "hop_plan": "#0072B2",
    "search": "#E69F00",
    "doc_choose": "#CC79A7",
    "doc_read": "#009E73",
    "hop_resolve": "#D55E00",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def short(text: str, n: int = 52) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def stage_interval(events: list[dict]) -> tuple[float, float] | None:
    starts = [float(event.get("start_s", 0.0)) for event in events]
    ends = [float(event.get("end_s", 0.0)) for event in events]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def collect() -> dict:
    trace = load_json(TRACE_DIR / "trace.json")
    events = load_json(TRACE_DIR / "timeline_events.json")
    summary = trace["summary"]
    hop_states = trace["hop_states"]

    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for event in events:
        stage = event.get("stage")
        meta = event.get("meta") or {}
        hop_number = meta.get("hop_number")
        if stage not in STAGE_LABEL or hop_number is None:
            continue
        grouped[(int(hop_number), str(stage))].append(event)

    hop_rows = []
    for hop in hop_states:
        hop_number = int(hop["hop_number"])
        stage_rows = []
        for stage, label in STAGES:
            interval = stage_interval(grouped.get((hop_number, stage), []))
            if interval is None:
                continue
            start, end = interval
            event_count = len(grouped[(hop_number, stage)])
            stage_rows.append(
                {
                    "stage": stage,
                    "label": label,
                    "start_s": start,
                    "end_s": end,
                    "duration_s": max(0.0, end - start),
                    "event_count": event_count,
                }
            )
        web_queries = []
        local_queries = []
        for search in hop.get("search_history") or []:
            if search.get("type") == "web_search":
                web_queries.append(search.get("query", ""))
            elif search.get("type") == "local_document_search":
                local_queries.append(search.get("query", ""))
        hop_rows.append(
            {
                "hop_number": hop_number,
                "hop_type": hop.get("hop_type"),
                "status": hop.get("status"),
                "question": hop.get("question"),
                "answer": hop.get("answer"),
                "stage_rows": stage_rows,
                "web_queries": web_queries,
                "local_queries": local_queries,
                "gold_doc_ref": hop.get("gold_doc_ref"),
            }
        )
    return {"summary": summary, "hops": hop_rows}


def make_plot(data: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    hops = data["hops"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y_gap = 0.9
    bar_h = 0.45
    y_positions = []
    y_labels = []
    for idx, hop in enumerate(hops):
        y = (len(hops) - 1 - idx) * y_gap
        y_positions.append(y)
        y_labels.append(
            f"H{hop['hop_number']} {hop['hop_type']} → {hop.get('answer') or 'unanswered'}\n{short(hop.get('question') or '', 44)}"
        )
        for stage in hop["stage_rows"]:
            start = stage["start_s"]
            width = stage["duration_s"]
            ax.broken_barh(
                [(start, width)],
                (y - bar_h / 2, bar_h),
                facecolors=STAGE_COLORS[stage["stage"]],
                edgecolors="white",
                linewidth=0.7,
            )
            if width >= 8:
                label = stage["label"]
                if stage["event_count"] > 1 and stage["stage"] in {"search", "doc_read"}:
                    label += f" ×{stage['event_count']}"
                ax.text(start + width / 2, y, label, ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    duration = float(data["summary"].get("duration_s") or 0.0)
    ax.set_xlim(0, max(duration, max(stage["end_s"] for hop in hops for stage in hop["stage_rows"])) * 1.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Seconds since rollout start")
    ax.set_title("Example Privacy HopQA rollout timeline")
    ax.grid(axis="x", color="#d8dfdc", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=STAGE_COLORS[stage], label=label) for stage, label in STAGES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=5, frameon=False)
    caption = (
        f"Trace {data['summary']['task_id']}_{data['summary']['chain_id']}; "
        f"{data['summary']['resolved_hops']}/{data['summary']['total_hops']} hops resolved; "
        f"{data['summary']['duration_s']:.0f}s total. Parallel calls are shown as one critical-path stage span."
    )
    fig.text(0.01, 0.01, caption, fontsize=7.5, color="#5d6964")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for suffix in ("svg", "pdf"):
        fig.savefig(PLOTS_DIR / f"example_rollout_timeline.{suffix}", bbox_inches="tight")
    plt.close(fig)


def write_tables(data: dict) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for hop in data["hops"]:
        rows.append(
            {
                "hop": hop["hop_number"],
                "type": hop["hop_type"],
                "answer": hop.get("answer") or "",
                "question": hop.get("question") or "",
                "local_queries": " | ".join(hop["local_queries"]),
                "web_queries": " | ".join(hop["web_queries"]),
            }
        )
    (OUT_DIR / "example_rollout_timeline_summary.json").write_text(json.dumps(data, indent=2))
    html_rows = []
    for row in rows:
        html_rows.append(
            "<tr>"
            f"<td>H{row['hop']} {html.escape(row['type'])}</td>"
            f"<td>{html.escape(row['answer'])}</td>"
            f"<td>{html.escape(row['question'])}</td>"
            f"<td>{html.escape(row['local_queries']) or '<span class=\"progress-empty\">none</span>'}</td>"
            f"<td>{html.escape(row['web_queries']) or '<span class=\"progress-empty\">none</span>'}</td>"
            "</tr>"
        )
    (TABLES_DIR / "example_rollout_query_table.html").write_text("\n".join(html_rows))
    with (TABLES_DIR / "example_rollout_query_table.tex").open("w") as f:
        f.write("% Auto-generated by generate_example_rollout_timeline.py\n")
        f.write("\\begin{tabular}{llll}\n\\toprule\nHop & Answer & Local queries & Web queries \\\\\n\\midrule\n")
        for row in rows:
            local = row["local_queries"].replace("&", "\\&")[:120]
            web = row["web_queries"].replace("&", "\\&")[:120]
            answer = row["answer"].replace("&", "\\&")
            f.write(f"H{row['hop']} {row['type']} & {answer} & {local} & {web} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    data = collect()
    make_plot(data)
    write_tables(data)
    print(json.dumps({"trace": str(TRACE_DIR), "summary": data["summary"], "hops": len(data["hops"])}, indent=2))


if __name__ == "__main__":
    main()
