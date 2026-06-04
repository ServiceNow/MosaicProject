#!/usr/bin/env python3
"""Interactive HTML viewer for mosaicleaks chain JSONL (build / filter output).

Reads chains.jsonl, chains_filtered.jsonl, or chains_verified.jsonl - any line with
chain_id, hops, verification, posthoc, metadata.trace.

Usage:
  python3 -m mosaicleaks.scripts.view_generation \
    ~/mosaicleaks_runs/example_run/chains_filtered.jsonl
  python3 -m mosaicleaks.scripts.view_generation chains.jsonl -o chains.html --open
"""

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path


def _esc(s: object) -> str:
    return html.escape(str(s) if s is not None else "", quote=True)


def _chain_card(c: dict, idx: int) -> str:
    cid = c.get("chain_id", "?")
    pattern = c.get("pattern", "")
    valid = c.get("valid")
    v = c.get("verification") or {}
    is_valid = v.get("is_valid")
    post = c.get("posthoc") or {}
    score = post.get("score")
    meta = c.get("metadata") or {}
    trace = meta.get("trace") or []

    hops_html = []
    for h in c.get("hops") or []:
        q = h.get("question", "")
        d = h.get("doc_id", "")
        hops_html.append(
            f'<div class="hop"><span class="tag">{_esc(h.get("hop_type",""))}</span> '
            f'<span class="hop-num">#{h.get("hop_number")}</span>'
            f'<div class="q">{_esc(q)}</div>'
            f'<div class="a"><strong>A:</strong> {_esc(h.get("answer"))}</div>'
            f'<div class="doc">{_esc(d[:120])}{"..." if len(d or "") > 120 else ""}</div>'
            f'<details class="quote"><summary>Quote</summary><pre>{_esc(h.get("quote") or "")}</pre></details>'
            f'</div>'
        )

    trace_block = ""
    if trace:
        trace_json = json.dumps(trace, indent=2, ensure_ascii=False)
        trace_block = (
            f'<details class="trace"><summary>Build trace ({len(trace)} steps)</summary>'
            f'<pre>{_esc(trace_json)}</pre></details>'
        )

    badge = "valid" if valid else "invalid"
    if is_valid is True:
        badge += " verify-ok"
    elif is_valid is False:
        badge += " verify-fail"

    nq = c.get("numbered_questions") or ""
    return f"""
<div class="chain" id="c-{idx}">
  <div class="chain-head">
    <span class="badge {badge}">{_esc(badge)}</span>
    <span class="cid">{_esc(cid)}</span>
    <span class="pat">{_esc(pattern)}</span>
    <span class="task">{_esc(meta.get("task_id") or meta.get("seed_task_id") or "")}</span>
    {f'<span class="score">judge {score}/10</span>' if score is not None else ""}
  </div>
  <div class="numbered"><pre>{_esc(nq)}</pre></div>
  <div class="global"><strong>Final answer:</strong> {_esc(c.get("global_answer"))}</div>
  <div class="hops">{"".join(hops_html)}</div>
  {trace_block}
  <details><summary>Verification</summary><pre>{_esc(json.dumps(v, indent=2)) if v else "-"}</pre></details>
  <details><summary>Posthoc / metadata</summary><pre>{_esc(json.dumps({"posthoc": post, "metadata": {k: v for k, v in meta.items() if k != "trace"}}, indent=2))}</pre></details>
</div>
"""


def _page(chains: list[dict], title: str) -> str:
    nav = "".join(
        f'<a href="#c-{i}">{_esc(c.get("chain_id") or str(i))}</a>'
        for i, c in enumerate(chains)
    )
    cards = "".join(_chain_card(c, i) for i, c in enumerate(chains))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_esc(title)}</title>
<style>
:root {{
  --bg: #f8fafc;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --accent: #2563eb;
  --border: #e2e8f0;
  --accent-soft: #eff6ff;
}}
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; line-height: 1.5; }}
header {{ padding: 16px 24px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--card); z-index: 2; box-shadow: 0 1px 0 rgba(15,23,42,.06); }}
h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 8px 0; color: var(--text); }}
nav {{ display: flex; flex-wrap: wrap; gap: 8px; font-size: 13px; }}
nav a {{ color: var(--accent); text-decoration: none; }}
nav a:hover {{ text-decoration: underline; }}
main {{ max-width: 920px; margin: 0 auto; padding: 24px; }}
.chain {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 24px; margin-bottom: 24px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.chain-head {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }}
.badge {{ font-size: 11px; text-transform: uppercase; padding: 2px 8px; border-radius: 4px; }}
.badge.valid {{ background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
.badge.invalid {{ background: #f8fafc; color: #64748b; border: 1px solid var(--border); }}
.badge.verify-ok {{ border: 2px solid #22c55e; }}
.badge.verify-fail {{ border: 2px solid #ef4444; }}
.cid {{ font-family: ui-monospace, monospace; font-weight: 600; }}
.pat, .task {{ color: var(--muted); font-size: 13px; }}
.score {{ font-size: 13px; color: #b45309; font-weight: 600; }}
.numbered pre {{ white-space: pre-wrap; font-size: 14px; margin: 8px 0; color: var(--text); }}
.global {{ margin: 12px 0; font-size: 14px; }}
.hop {{ border-left: 3px solid var(--accent); padding: 8px 12px; margin: 12px 0; background: var(--accent-soft); border-radius: 0 8px 8px 0; }}
.tag {{ font-size: 11px; background: var(--border); color: var(--muted); padding: 1px 6px; border-radius: 3px; }}
.hop-num {{ color: var(--muted); font-size: 12px; margin-left: 6px; }}
.q {{ margin: 8px 0; }}
.a {{ color: #1e40af; }}
.doc {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
details {{ margin-top: 10px; }}
details summary {{ cursor: pointer; color: var(--muted); font-size: 13px; }}
pre {{ font-size: 12px; overflow-x: auto; max-height: 420px; margin: 8px 0; background: #f1f5f9; padding: 10px; border-radius: 6px; border: 1px solid var(--border); }}
.trace pre {{ max-height: 60vh; }}
.quote pre {{ max-height: 200px; }}
</style>
</head>
<body>
<header>
  <h1>{_esc(title)}</h1>
  <nav>{nav}</nav>
</header>
<main>
{cards}
</main>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description="HTML viewer for mosaicleaks chain JSONL")
    p.add_argument("jsonl", type=Path, help="Path to chains*.jsonl")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output HTML (default: <stem>.html)")
    p.add_argument("--open", action="store_true", help="Open in default browser")
    args = p.parse_args()

    if not args.jsonl.exists():
        print(f"Not found: {args.jsonl}", file=sys.stderr)
        return 1

    chains: list[dict] = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chains.append(json.loads(line))

    out = args.output
    if out is None:
        out = args.jsonl.with_suffix(".html")

    title = f"{args.jsonl.name} ({len(chains)} chains)"
    out.write_text(_page(chains, title), encoding="utf-8")
    print(f"Wrote {out} ({len(chains)} chains)")

    if args.open:
        webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
