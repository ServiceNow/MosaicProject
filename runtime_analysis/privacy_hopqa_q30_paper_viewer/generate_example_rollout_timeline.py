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
    "privacy_hopqa_runs/DR0001_25cbf112_298ad369"
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
                "events": [
                    {
                        "stage": event.get("stage"),
                        "label": STAGE_LABEL[event.get("stage")],
                        "start_s": float(event.get("start_s", 0.0)),
                        "end_s": float(event.get("end_s", 0.0)),
                        "duration_s": max(0.0, float(event.get("end_s", 0.0)) - float(event.get("start_s", 0.0))),
                        "meta": event.get("meta") or {},
                    }
                    for event in events
                    if event.get("stage") in STAGE_LABEL
                    and (event.get("meta") or {}).get("hop_number") == hop_number
                ],
                "web_queries": web_queries,
                "local_queries": local_queries,
                "gold_doc_ref": hop.get("gold_doc_ref"),
            }
        )
    return {"summary": summary, "hops": hop_rows}


def add_event_lanes(hops: list[dict]) -> list[dict]:
    """Assign overlapping events to compact lanes within each hop row."""

    plot_events = []
    y_cursor = 0.0
    lane_height = 0.085
    group_gap = 0.24
    for hop in hops:
        lane_ends: list[float] = []
        hop_events = sorted(hop["events"], key=lambda event: (event["start_s"], event["end_s"]))
        for event in hop_events:
            lane = 0
            while lane < len(lane_ends) and event["start_s"] < lane_ends[lane] - 1e-6:
                lane += 1
            if lane == len(lane_ends):
                lane_ends.append(event["end_s"])
            else:
                lane_ends[lane] = event["end_s"]
            event = dict(event)
            event["lane"] = lane
            event["y"] = y_cursor + lane * lane_height
            plot_events.append(event)
        lane_count = max(1, len(lane_ends))
        hop["label_y"] = y_cursor + (lane_count - 1) * lane_height / 2
        hop["lane_count"] = lane_count
        hop["row_start_y"] = y_cursor - lane_height * 0.65
        hop["row_end_y"] = y_cursor + (lane_count - 1) * lane_height + lane_height * 0.65
        y_cursor += lane_count * lane_height + group_gap
    return plot_events


def make_plot(data: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    hops = data["hops"]
    plot_events = add_event_lanes(hops)
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
    fig, ax = plt.subplots(figsize=(13.8, 3.1))
    bar_h = 0.064
    for idx, hop in enumerate(hops):
        if idx % 2 == 0:
            ax.axhspan(hop["row_start_y"], hop["row_end_y"], color="#f4f7f5", zorder=0)
    for event in plot_events:
        start = event["start_s"]
        width = max(event["duration_s"], 0.08)
        stage = event["stage"]
        edgecolor = "#1e2723" if stage == "doc_read" and event["meta"].get("can_answer") else "white"
        linewidth = 1.1 if edgecolor != "white" else 0.35
        ax.broken_barh(
            [(start, width)],
            (event["y"] - bar_h / 2, bar_h),
            facecolors=STAGE_COLORS[stage],
            edgecolors=edgecolor,
            linewidth=linewidth,
            alpha=0.95,
            zorder=3,
        )
        if stage == "doc_read" and event["meta"].get("can_answer"):
            ax.plot(start + width / 2, event["y"], marker="o", markersize=3.4, color="#1e2723", zorder=4)

    y_positions = [hop["label_y"] for hop in hops]
    y_labels = [
        f"H{hop['hop_number']} {hop['hop_type']} -> {hop.get('answer') or 'unanswered'}"
        for hop in hops
    ]

    duration = float(data["summary"].get("duration_s") or 0.0)
    ax.set_xlim(0, max(duration, max(event["end_s"] for event in plot_events)) * 1.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds since rollout start")
    ax.set_title("Example Privacy HopQA rollout: parallel harness stages over wall-clock time")
    ax.grid(axis="x", color="#d8dfdc", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=STAGE_COLORS[stage], label=label) for stage, label in STAGES]
    handles.append(Patch(facecolor="#009E73", edgecolor="#1e2723", label="Read found answer"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.33), ncol=6, frameon=False)
    caption = (
        f"Trace {data['summary']['task_id']}_{data['summary']['chain_id']}; "
        f"{data['summary']['resolved_hops']}/{data['summary']['total_hops']} hops resolved; "
        f"{data['summary']['duration_s']:.0f}s total. Multiple bars on the same hop row are parallel calls, not sequential work."
    )
    fig.text(0.01, 0.01, caption, fontsize=7.5, color="#5d6964")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
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
