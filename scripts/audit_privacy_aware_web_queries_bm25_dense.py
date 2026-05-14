#!/usr/bin/env python3
"""Compare current web retrieval against BM25 for privacy-aware paraphrase queries.

This diagnostic looks for web_search calls in Privacy HopQA traces where the
model appears to generalize or paraphrase the query, e.g. dropping the company
name or exact private bridge value and searching for public/industry analogues.
It then compares the live helper's current web ranking against a local BM25
ranking over BrowseComp full docs plus curated helper web chunks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import argparse
import html
import json
import math
import re

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122")
DEFAULT_DATASET = REPO / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl"
DEFAULT_HELPER_URL = "http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30ps20260513173046:8012"
DEFAULT_OUT_DIR = REPO / "runtime_analysis/privacy_hopqa_privacy_aware_web_query_bm25_dense_2026-05-14"
FULL_DOCS_DIR = Path("/home/toolkit/full_docs")
HELPER_INPUTS = Path("/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/helper_inputs")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
GENERAL_RE = re.compile(
    r"\b(general|industry|benchmark|public|publicly|standard|typical|average|retail|grocery chain|provider|program|regulatory|framework|similar|analog|cap|budget|funding)\b",
    re.I,
)
STOPWORDS = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "over", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "with", "under", "per", "each",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def compact(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower().replace("\ufb01", "fi").replace("\ufb02", "fl"))


def content_tokens(text: str) -> set[str]:
    return {tok for tok in tokens(text) if tok not in STOPWORDS and (len(tok) > 1 or tok.isdigit())}


def normalize_doc_id(doc_id: Any) -> str:
    doc = str(doc_id or "").split("#", 1)[0]
    if not doc:
        return ""
    if doc.startswith("web/"):
        match = re.search(r"_(\d+)$", doc)
        if match:
            return f"web/{match.group(1)}"
        return doc
    if doc.isdigit():
        return f"web/{doc}"
    return doc


def rank_of(hits: list[dict[str, Any]], gold_doc_id: str) -> int | None:
    gold = normalize_doc_id(gold_doc_id)
    if not gold:
        return None
    for idx, hit in enumerate(hits, start=1):
        if normalize_doc_id(hit.get("doc_id")) == gold:
            return idx
    return None


def hit_doc_id(hit: dict[str, Any]) -> str:
    return str(hit.get("docid") or hit.get("doc_id") or hit.get("parent_doc_id") or "").split("#", 1)[0]


def load_dataset(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row.get("task_id") or "")
        chain_id = str(row.get("chain_id") or "")
        for hop in row.get("hops") or []:
            hop_number = int(hop.get("hop_number") or 0)
            out[(task_id, chain_id, hop_number)] = {"row": row, "hop": hop}
    return out


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    source: str
    canonical_id: str


class BM25Index:
    def __init__(self, docs: list[CorpusDoc]) -> None:
        self.docs = docs
        self.doc_tokens: list[list[str]] = []
        self.freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        df: Counter[str] = Counter()
        for doc in docs:
            toks = [tok for tok in tokens(doc.title + "\n" + doc.text) if tok not in STOPWORDS]
            self.doc_tokens.append(toks)
            counts = Counter(toks)
            self.freqs.append(counts)
            self.doc_lens.append(len(toks))
            df.update(counts.keys())
        self.n = len(docs)
        self.avgdl = sum(self.doc_lens) / max(1, self.n)
        self.idf = {term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        q_terms = [tok for tok in tokens(query) if tok not in STOPWORDS]
        if not q_terms:
            return []
        q_counts = Counter(q_terms)
        scores: list[tuple[float, int]] = []
        k1 = 1.5
        b = 0.75
        for idx, counts in enumerate(self.freqs):
            score = 0.0
            dl = self.doc_lens[idx] or 1
            for term, qtf in q_counts.items():
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                denom = tf + k1 * (1 - b + b * dl / max(1e-9, self.avgdl))
                score += self.idf.get(term, 0.0) * ((tf * (k1 + 1)) / denom) * min(1.0, qtf)
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda item: (-item[0], self.docs[item[1]].doc_id))
        hits = []
        for rank, (score, idx) in enumerate(scores[:k], start=1):
            doc = self.docs[idx]
            hits.append(
                {
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "canonical_doc_id": doc.canonical_id,
                    "score": score,
                    "title": doc.title,
                    "source": doc.source,
                    "snippet": best_snippet(doc.text, q_terms),
                }
            )
        return hits


def best_snippet(text: str, q_terms: list[str], radius: int = 260) -> str:
    lower = text.lower()
    positions = [lower.find(term) for term in q_terms if len(term) > 2 and lower.find(term) >= 0]
    if not positions:
        return compact(text, 520)
    pos = min(positions)
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return compact(text[start:end], 520)


def load_bm25_corpus(full_docs_dir: Path, helper_inputs: Path) -> list[CorpusDoc]:
    docs: dict[str, CorpusDoc] = {}
    for path in sorted(full_docs_dir.glob("*.txt")):
        doc_id = f"web/{path.stem}"
        docs[doc_id] = CorpusDoc(
            doc_id=doc_id,
            title=path.name,
            text=path.read_text(encoding="utf-8", errors="replace"),
            source="browsecomp_full_docs",
            canonical_id=doc_id,
        )
    for path in sorted(helper_inputs.glob("chunks_curated_web*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("doc_id") or "")
            if not doc_id:
                continue
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
            canonical = normalize_doc_id(meta.get("source_original_doc_id") or doc_id)
            title = str(meta.get("title") or doc_id)
            text = str(obj.get("text") or "")
            if doc_id in docs:
                docs[doc_id].text += "\n\n" + text
            else:
                docs[doc_id] = CorpusDoc(doc_id=doc_id, title=title, text=text, source="curated_web_chunks", canonical_id=canonical)
    return list(docs.values())


def helper_web_search(helper_url: str, task_id: str, query: str, k: int, timeout_s: float) -> list[dict[str, Any]]:
    payload = {"query": query, "task_id": task_id, "k": k}
    request = Request(
        helper_url.rstrip("/") + "/browsecomp/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    hits = []
    for rank, hit in enumerate(data.get("hits") or [], start=1):
        doc_id = hit_doc_id(hit)
        hits.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "canonical_doc_id": normalize_doc_id(doc_id),
                "score": hit.get("score"),
                "title": str(hit.get("title") or hit.get("url") or doc_id),
                "source": str(hit.get("source") or "current_helper"),
                "snippet": compact(hit.get("text") or hit.get("snippet") or "", 520),
            }
        )
    return hits


def query_contains_any(query: str, values: list[str]) -> bool:
    q = " ".join(tokens(query))
    for value in values:
        toks = tokens(value)
        if toks and " ".join(toks) in q:
            return True
    return False


def collect_candidate_queries(run_root: Path, dataset_map: dict[tuple[str, str, int], dict[str, Any]], max_traces: int) -> list[dict[str, Any]]:
    out: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    trace_paths = sorted((run_root / "privacy_hopqa_runs").glob("*/trace.json"))[:max_traces]
    for trace_path in trace_paths:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        summary = data.get("summary") or {}
        task_id = str(summary.get("task_id") or "")
        chain_id = str(summary.get("chain_id") or "")
        actions = {str(action.get("id") or ""): action for action in data.get("actions") or []}
        for hop_state in data.get("hop_states") or []:
            hop_number = int(hop_state.get("hop_number") or 0)
            ds = dataset_map.get((task_id, chain_id, hop_number))
            if ds is None:
                continue
            row = ds["row"]
            hop = ds["hop"]
            company = str(row.get("company") or "")
            answer_values = [str(hop.get("answer") or "")]
            answer_values.extend(str(v) for v in hop.get("accepted_answer_variants") or [])
            for event in hop_state.get("search_history") or []:
                if event.get("type") != "web_search":
                    continue
                query = str(event.get("query") or "").strip()
                if not query:
                    continue
                action = actions.get(str(event.get("action_id") or ""), {})
                description = str(action.get("description") or "")
                expected = str(action.get("expected_output") or "")
                contains_company = query_contains_any(query, [company]) if company else False
                contains_answer = query_contains_any(query, [v for v in answer_values if v])
                general = bool(GENERAL_RE.search(query + " " + description + " " + expected))
                hop_type = str(hop.get("hop_type") or hop_state.get("hop_type") or "")
                private_omission = not contains_company and not contains_answer
                if not (general and private_omission):
                    continue
                category = "local_hop_privacy_paraphrase" if hop_type == "L" else "web_hop_privacy_paraphrase"
                key = (task_id, chain_id, hop_number, query.lower())
                item = out.setdefault(
                    key,
                    {
                        "task_id": task_id,
                        "chain_id": chain_id,
                        "hop_number": hop_number,
                        "hop_type": hop_type,
                        "category": category,
                        "company": company,
                        "question": str(hop_state.get("question") or hop.get("question") or ""),
                        "answer": str(hop_state.get("canonical_answer") or hop.get("answer") or ""),
                        "gold_doc_id": str(hop.get("doc_id") or ""),
                        "gold_doc_ref": str(hop.get("doc_ref") or ""),
                        "query": query,
                        "description": description,
                        "expected_output": expected,
                        "contains_company": contains_company,
                        "contains_answer": contains_answer,
                        "observed_count": 0,
                        "trace_example": str(trace_path),
                    },
                )
                item["observed_count"] += 1
    rows = list(out.values())
    rows.sort(key=lambda item: (-int(item["observed_count"]), item["category"], item["task_id"], item["chain_id"], item["hop_number"], item["query"]))
    return rows


def hit_overlap(a: list[dict[str, Any]], b: list[dict[str, Any]], k: int) -> float:
    aset = {normalize_doc_id(hit.get("doc_id")) for hit in a[:k]}
    bset = {normalize_doc_id(hit.get("doc_id")) for hit in b[:k]}
    if not aset and not bset:
        return 1.0
    return len(aset & bset) / max(1, len(aset | bset))


def audit_queries(rows: list[dict[str, Any]], bm25: BM25Index, helper_url: str, k: int, timeout_s: float) -> list[dict[str, Any]]:
    audited = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['category']} {row['task_id']} {row['chain_id']} h{row['hop_number']} :: {row['query'][:90]}", flush=True)
        try:
            current_hits = helper_web_search(helper_url, row["task_id"], row["query"], k, timeout_s)
            error = None
        except Exception as exc:
            current_hits = []
            error = f"{type(exc).__name__}: {exc}"
        bm25_hits = bm25.search(row["query"], k)
        gold_rank_current = None
        gold_rank_bm25 = None
        if row["hop_type"] == "W":
            gold_rank_current = rank_of(current_hits, row["gold_doc_id"])
            gold_rank_bm25 = rank_of(bm25_hits, row["gold_doc_id"])
        audited.append(
            {
                **row,
                "current_error": error,
                "current_hits": current_hits,
                "bm25_hits": bm25_hits,
                "current_top5_ids": [hit["canonical_doc_id"] for hit in current_hits[:5]],
                "bm25_top5_ids": [hit["canonical_doc_id"] for hit in bm25_hits[:5]],
                "top5_overlap": hit_overlap(current_hits, bm25_hits, 5),
                "top10_overlap": hit_overlap(current_hits, bm25_hits, 10),
                "gold_rank_current": gold_rank_current,
                "gold_rank_bm25": gold_rank_bm25,
            }
        )
    return audited


def select_balanced_queries(candidates: list[dict[str, Any]], max_queries: int, max_per_hop: int) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_category[row["category"]].append(row)
    for rows in by_category.values():
        rows.sort(key=lambda item: (-int(item["observed_count"]), item["task_id"], item["chain_id"], item["hop_number"], item["query"]))

    selected: list[dict[str, Any]] = []
    per_hop_counts: Counter[tuple[str, str, int]] = Counter()
    target_each = max_queries // 2
    for category in ["local_hop_privacy_paraphrase", "web_hop_privacy_paraphrase"]:
        added = 0
        for row in by_category.get(category, []):
            key = (row["task_id"], row["chain_id"], int(row["hop_number"]))
            if per_hop_counts[key] >= max_per_hop:
                continue
            selected.append(row)
            per_hop_counts[key] += 1
            added += 1
            if added >= target_each:
                break
    if len(selected) < max_queries:
        selected_keys = {(row["task_id"], row["chain_id"], row["hop_number"], row["query"]) for row in selected}
        for row in candidates:
            row_key = (row["task_id"], row["chain_id"], row["hop_number"], row["query"])
            hop_key = (row["task_id"], row["chain_id"], int(row["hop_number"]))
            if row_key in selected_keys or per_hop_counts[hop_key] >= max_per_hop:
                continue
            selected.append(row)
            selected_keys.add(row_key)
            per_hop_counts[hop_key] += 1
            if len(selected) >= max_queries:
                break
    return selected[:max_queries]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows), "generated_at": datetime.now(UTC).isoformat(), "groups": {}}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[row["category"]].append(row)
        groups[f"hop_type={row['hop_type']}"] .append(row)
    for key, items in sorted(groups.items()):
        web_items = [row for row in items if row["hop_type"] == "W"]
        summary["groups"][key] = {
            "n": len(items),
            "mean_top5_overlap": sum(float(row["top5_overlap"]) for row in items) / max(1, len(items)),
            "mean_top10_overlap": sum(float(row["top10_overlap"]) for row in items) / max(1, len(items)),
            "different_top1": sum((row["current_top5_ids"][:1] != row["bm25_top5_ids"][:1]) for row in items),
            "web_gold_n": len(web_items),
            "web_current_gold_top5": sum((row["gold_rank_current"] is not None and row["gold_rank_current"] <= 5) for row in web_items),
            "web_bm25_gold_top5": sum((row["gold_rank_bm25"] is not None and row["gold_rank_bm25"] <= 5) for row in web_items),
            "web_current_gold_top10": sum((row["gold_rank_current"] is not None and row["gold_rank_current"] <= 10) for row in web_items),
            "web_bm25_gold_top10": sum((row["gold_rank_bm25"] is not None and row["gold_rank_bm25"] <= 10) for row in web_items),
            "web_current_gold_top20": sum((row["gold_rank_current"] is not None and row["gold_rank_current"] <= 20) for row in web_items),
            "web_bm25_gold_top20": sum((row["gold_rank_bm25"] is not None and row["gold_rank_bm25"] <= 20) for row in web_items),
        }
    return summary


def fmt_count(count: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{count}/{total} ({100*count/total:.1f}%)"


def build_viewer(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    summary_rows = []
    for key, group in summary["groups"].items():
        n = int(group["n"])
        w = int(group["web_gold_n"])
        summary_rows.append(
            "<tr>"
            f"<td>{esc(key)}</td><td>{n}</td>"
            f"<td>{float(group['mean_top5_overlap']):.2f}</td><td>{group['different_top1']}</td>"
            f"<td>{fmt_count(int(group['web_current_gold_top5']), w)}</td><td>{fmt_count(int(group['web_bm25_gold_top5']), w)}</td>"
            f"<td>{fmt_count(int(group['web_current_gold_top20']), w)}</td><td>{fmt_count(int(group['web_bm25_gold_top20']), w)}</td>"
            "</tr>"
        )
    detail_rows = []
    for row in rows:
        detail_rows.append(
            "<tr>"
            f"<td>{esc(row['category'])}<br>{esc(row['task_id'])} {esc(row['chain_id'])} h{esc(row['hop_number'])}<br>observed {esc(row['observed_count'])}x</td>"
            f"<td><b>Q:</b> {esc(row['question'])}<br><b>Gold:</b> {esc(row['answer'])}<br><b>Query:</b> {esc(row['query'])}<br><b>Description:</b> {esc(row['description'])}<br><b>Expected:</b> {esc(row['expected_output'])}<br><b>Trace:</b> <code>{esc(row['trace_example'])}</code></td>"
            f"<td>current rank={esc(row['gold_rank_current'])}<br><pre>{esc(json.dumps(row['current_hits'][:8], indent=2))}</pre></td>"
            f"<td>bm25 rank={esc(row['gold_rank_bm25'])}<br><pre>{esc(json.dumps(row['bm25_hits'][:8], indent=2))}</pre></td>"
            f"<td>top5 overlap={float(row['top5_overlap']):.2f}<br>current={esc(row['current_top5_ids'])}<br>bm25={esc(row['bm25_top5_ids'])}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Privacy-aware web query BM25 vs current helper audit</title>
<style>body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#f8f4ec;color:#17202a}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:16px 0}}td,th{{border:1px solid #d8cdbb;padding:8px;vertical-align:top}}th{{background:#dceee5}}pre{{white-space:pre-wrap;max-height:420px;overflow:auto;background:#f6f1e8;padding:8px}}code{{font-size:12px}}</style></head>
<body><h1>Privacy-aware web query BM25 vs current helper audit</h1>
<p>Queries are actual web_search calls from Privacy HopQA traces where the model used generalized/public/industry phrasing and omitted the company name and exact accepted answer. Current results come from the live helper. BM25 results are local lexical retrieval over BrowseComp full docs plus curated helper web chunks.</p>
<h2>Summary</h2><table><thead><tr><th>group</th><th>n</th><th>mean top5 overlap</th><th>different top1</th><th>current web gold top5</th><th>BM25 web gold top5</th><th>current web gold top20</th><th>BM25 web gold top20</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<h2>Rows</h2><table><thead><tr><th>case</th><th>query context</th><th>current helper hits</th><th>BM25 hits</th><th>overlap</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table>
</body></html>"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--helper-url", default=DEFAULT_HELPER_URL)
    parser.add_argument("--full-docs-dir", type=Path, default=FULL_DOCS_DIR)
    parser.add_argument("--helper-inputs", type=Path, default=HELPER_INPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-traces", type=int, default=2500)
    parser.add_argument("--max-queries", type=int, default=80)
    parser.add_argument("--max-per-hop", type=int, default=3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    candidates = collect_candidate_queries(args.run_root, dataset, args.max_traces)
    selected = select_balanced_queries(candidates, args.max_queries, args.max_per_hop)
    print(f"selected {len(selected)} / {len(candidates)} privacy-aware candidate web queries", flush=True)
    corpus = load_bm25_corpus(args.full_docs_dir, args.helper_inputs)
    print(f"loaded BM25 corpus docs={len(corpus)}", flush=True)
    bm25 = BM25Index(corpus)
    audited = audit_queries(selected, bm25, args.helper_url, args.k, args.timeout_s)
    summary = summarize(audited)
    (args.out_dir / "privacy_aware_web_query_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audited), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_viewer(args.out_dir, audited, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
