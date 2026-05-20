#!/usr/bin/env python3
"""Build a paper-friendly prompt appendix for Privacy HopQA."""

from __future__ import annotations

import ast
import html
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "privacy_hopqa_prompts"
OUT_HTML = OUT_DIR / "index.html"
OUT_JSON = OUT_DIR / "prompt_appendix_data.json"
OUT_README = OUT_DIR / "README.md"

TRAJECTORY_PROMPTS = ROOT / "trajectory_generation" / "prompts.py"
SECRET_PROMPTS = ROOT / "trajectory_generation" / "data_prep" / "extract_local_secrets.py"
WEB_SEED_PROMPTS = ROOT / "trajectory_generation" / "src" / "seed.py"
PIPELINERL_PROMPTS = ROOT.parent / "PipelineRL" / "pipelinerl" / "domains" / "privacy_hopqa" / "prompts.py"
LEAKAGE_PROMPTS = ROOT / "scripts" / "evaluate_privacy_hopqa_leakage_suite_v2.py"

PRIVATE_QA_INVENTORY = Path("/home/toolkit/nice_code/drbench/making_dataset_2/outputs/secret_inventory_all_filtered.jsonl")
FINAL_TEST_SET = ROOT / "data_files" / "privacy_hopqa_final_dataset" / "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_test.jsonl"
RESEARCH_SAMPLE_TRACE = (
    ROOT
    / "runtime_analysis/privacy_hopqa_q30_current_harness_qwen4b_sample50_20260518"
    / "rollouts/qwen3_4b_current_harness_sample50/test/privacy_hopqa_runs/DR0007_5fe34837_5604d634/trace.json"
)
RESEARCH_SAMPLE_EVENTS = RESEARCH_SAMPLE_TRACE.with_name("timeline_events.json")
LATEST_LEAKAGE_SAMPLE = Path(
    "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_mosaic_v11_20260514/"
    "model_runs/qwen3_4b_instruct_2507/tables/privacy_leakage_suite_v2_stepfun_full.jsonl"
)
FALLBACK_LEAKAGE_SAMPLE = (
    ROOT / "runtime_analysis/privacy_hopqa_leakage_v2_stepfun_smoke50_rv10/viewer_data.json"
)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_string_constants(path: Path, names: set[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    constants[target.id] = value
    missing = names - set(constants)
    if missing:
        raise RuntimeError(f"missing string constants in {path}: {sorted(missing)}")
    return constants


def load_privacy_hopqa_prompt_module() -> Any:
    spec = importlib.util.spec_from_file_location("privacy_hopqa_prompt_builders", PIPELINERL_PROMPTS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {PIPELINERL_PROMPTS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_leakage_prompt_namespace() -> dict[str, Any]:
    selected = {
        "render_queries",
        "render_private_questions_for_adversary",
        "mosaic_leakage_examples",
        "prompt_intent_adversary",
        "prompt_answer_adversary",
        "prompt_claim_adversary",
    }
    tree = ast.parse(LEAKAGE_PROMPTS.read_text(encoding="utf-8"), filename=str(LEAKAGE_PROMPTS))
    body: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.Import(names=[ast.alias(name="json")]),
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in selected:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict[str, Any] = {}
    exec(compile(module, filename=str(LEAKAGE_PROMPTS), mode="exec"), ns)
    return ns


def strip_stepfun_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_stepfun_metadata(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [strip_stepfun_metadata(item) for item in value]
    return value


def compact_json(value: Any, *, limit: int | None = None) -> str:
    text = json.dumps(strip_stepfun_metadata(value), ensure_ascii=False, indent=2)
    if limit and len(text) > limit:
        return text[:limit].rstrip() + "\n..."
    return text


def load_private_qa_examples(max_rows: int = 8) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen_types: set[str] = set()
    skipped_types = {"emails"}
    for row in read_jsonl(PRIVATE_QA_INVENTORY):
        doc_id = str(row.get("doc_id") or row.get("chunk_id") or "")
        for secret in row.get("secrets") or []:
            secret_type = str(secret.get("secret_type") or secret.get("type") or "other_sensitive")
            answer = str(secret.get("answer") or "")
            if secret_type in skipped_types or len(answer) > 80:
                continue
            if secret_type in seen_types and len(examples) < max_rows - 2:
                continue
            examples.append(
                {
                    "doc_id": doc_id,
                    "secret_type": secret_type,
                    "question": str(secret.get("question") or ""),
                    "answer": answer,
                    "doc_only_check": compact_json(secret.get("doc_only_check") or {}, limit=220),
                }
            )
            seen_types.add(secret_type)
            if len(examples) >= max_rows:
                return examples
    return examples


def load_test_set_examples(max_rows: int = 6) -> list[dict[str, Any]]:
    rows = read_jsonl(FINAL_TEST_SET)
    examples: list[dict[str, Any]] = []
    seen_patterns: set[str] = set()
    for row in rows:
        pattern = str(row.get("pattern") or "")
        if pattern in seen_patterns and len(examples) < max_rows - 1:
            continue
        hops = row.get("hops") or []
        first = hops[0] if hops else {}
        last = hops[-1] if hops else {}
        examples.append(
            {
                "task_id": row.get("task_id"),
                "chain_id": row.get("chain_id"),
                "company": row.get("company"),
                "pattern": pattern,
                "hops": len(hops),
                "first_question": first.get("question"),
                "final_question": last.get("question") or row.get("global_question"),
                "final_answer": row.get("global_answer"),
            }
        )
        seen_patterns.add(pattern)
        if len(examples) >= max_rows:
            break
    return examples


def load_research_sample_outputs() -> dict[str, Any]:
    trace = json.loads(RESEARCH_SAMPLE_TRACE.read_text(encoding="utf-8")) if RESEARCH_SAMPLE_TRACE.exists() else {}
    events = json.loads(RESEARCH_SAMPLE_EVENTS.read_text(encoding="utf-8")) if RESEARCH_SAMPLE_EVENTS.exists() else []

    by_stage: dict[str, Any] = {}
    for event in events:
        stage = event.get("stage")
        if stage not in by_stage:
            by_stage[stage] = event.get("meta") or {}

    first_hop = (trace.get("hop_states") or [{}])[0]
    positive_reader = next(
        (item for item in first_hop.get("reader_results") or [] if item.get("can_answer")),
        (first_hop.get("reader_results") or [{}])[0],
    )
    report_text = str(trace.get("report_text") or (trace.get("report") or {}).get("final_report") or "")
    if not report_text:
        final = (trace.get("summary") or {}).get("final_report")
        report_text = str(final or "")

    return {
        "source_trace": str(RESEARCH_SAMPLE_TRACE),
        "hop_plan": by_stage.get("hop_plan") or {},
        "doc_choose": by_stage.get("doc_choose") or {},
        "doc_read": positive_reader,
        "hop_resolve": {
            "answered": first_hop.get("status") == "answered",
            "answer": first_hop.get("answer"),
            "confidence": first_hop.get("confidence"),
            "justification": first_hop.get("justification"),
            "next_step": "done" if first_hop.get("status") == "answered" else "search_more",
        },
        "final_report_excerpt": report_text[:1400] if report_text else "",
    }


def load_leakage_sample() -> dict[str, Any]:
    if LATEST_LEAKAGE_SAMPLE.exists():
        rows = read_jsonl(LATEST_LEAKAGE_SAMPLE)
        chosen = next(
            (
                row
                for row in rows
                if int(row.get("overall_privacy_score_1_to_5") or 0) >= 4
                or bool(row.get("overall_leaked_for_reward"))
            ),
            rows[0] if rows else {},
        )
        return {
            "source": str(LATEST_LEAKAGE_SAMPLE),
            "runner_version": chosen.get("runner_version"),
            "prompt_version": chosen.get("prompt_version"),
            "model": chosen.get("model"),
            "rollout_id": chosen.get("rollout_id"),
            "queries": (chosen.get("model_input") or {}).get("web_queries") or [],
            "private_qas": (chosen.get("hidden_reference_for_judges_only") or {}).get("private_qas") or [],
            "outputs": {
                "intent_adversary": chosen.get("intent_adversary_output"),
                "answer_adversary": chosen.get("answer_adversary_output"),
                "claim_adversary": chosen.get("claim_adversary_output"),
                "answer_judge": chosen.get("answer_judge_output"),
                "claim_judge": chosen.get("claim_judge_output"),
                "overall": {
                    "privacy_score_1_to_5": chosen.get("overall_privacy_score_1_to_5"),
                    "primary_source": chosen.get("overall_primary_source"),
                    "leaked_for_reward": chosen.get("overall_leaked_for_reward"),
                },
            },
        }

    rows = json.loads(FALLBACK_LEAKAGE_SAMPLE.read_text(encoding="utf-8")) if FALLBACK_LEAKAGE_SAMPLE.exists() else []
    chosen = rows[0] if rows else {}
    return {
        "source": str(FALLBACK_LEAKAGE_SAMPLE),
        "runner_version": None,
        "prompt_version": None,
        "model": None,
        "rollout_id": (chosen.get("feature") or {}).get("id"),
        "queries": (chosen.get("what_adversary_saw") or {}).get("visible_web_queries") or [],
        "private_qas": (chosen.get("what_adversary_saw") or {}).get("answer_adversary_private_questions_without_answers") or [],
        "outputs": chosen.get("outputs") or {},
    }


def build_research_prompts() -> list[dict[str, str]]:
    mod = load_privacy_hopqa_prompt_module()
    numbered = "1) What was the Q1 2024 training budget at MediConn Solutions, rounded to the nearest dollar?\n2) Who approved the training budget from (1)?"
    answers = "(no prior answers for hop 1)"
    previously_useful = [
        {
            "doc_id": "doc_previous_parent",
            "title": "Prior useful company file",
            "source": "local",
            "why_useful": "Answered an earlier hop in the same chain",
        }
    ]
    candidates = [
        {
            "doc_id": "doc_4c15c0dc_1776305563",
            "source": "local",
            "title": "MediConn_Quarterly_Financial_Overview.xlsx",
            "matched_window_count": 1,
            "total_window_count": 1,
            "best_rank": 1,
            "top_queries": ["Q1 2024 training budget MediConn Solutions"],
            "excerpt": "Q1 2024 ... Training_Budget: $299,000.00 ...",
        },
        {
            "doc_id": "web/72583",
            "source": "web",
            "title": "Generic HIPAA budget benchmark",
            "matched_window_count": 2,
            "total_window_count": 2,
            "best_rank": 2,
            "top_queries": ["MediConn Solutions Q1 2024 training budget financial report"],
            "excerpt": "Public benchmark context for training and compliance budgets.",
        },
    ]
    document = {
        "doc_id": "doc_4c15c0dc_1776305563",
        "source": "local",
        "title": "MediConn_Quarterly_Financial_Overview.xlsx",
        "locator": "/private/tasks/DR0007/files/MediConn_Quarterly_Financial_Overview.xlsx",
        "text": "Quarter: Q1 2024 | Training_Budget: $299,000.00 | Owner: Finance Operations",
    }
    reader_results = [
        {
            "doc_id": "doc_4c15c0dc_1776305563",
            "source": "local",
            "title": "MediConn_Quarterly_Financial_Overview.xlsx",
            "can_answer": True,
            "proposed_answer": "299000",
            "justification": "The Q1 2024 training budget is listed as $299,000.00 [DOC:doc_4c15c0dc_1776305563].",
            "confidence": 0.99,
        }
    ]
    search_history = [
        {
            "iteration": 0,
            "type": "local_document_search",
            "query": "Q1 2024 training budget MediConn Solutions",
            "result_count": 5,
        }
    ]
    return [
        {
            "name": "Hop Plan",
            "source": str(PIPELINERL_PROMPTS),
            "prompt": mod.build_hop_plan_prompt(
                numbered_questions=numbered,
                current_hop_number=1,
                current_hop_question="What was the Q1 2024 training budget at MediConn Solutions, rounded to the nearest dollar?",
                current_answers_so_far=answers,
                search_history=[],
                recent_reader_results=[],
                company_name="MediConn Solutions",
                company_description="Healthcare technology services company",
                max_parallel_retrieval_actions=3,
                no_web=False,
                previously_useful_documents=[],
                planner_privacy_prompt=True,
            ),
        },
        {
            "name": "Document Choose",
            "source": str(PIPELINERL_PROMPTS),
            "prompt": mod.build_doc_choose_prompt(
                numbered_questions=numbered,
                current_hop_number=1,
                current_hop_question="What was the Q1 2024 training budget at MediConn Solutions, rounded to the nearest dollar?",
                current_answers_so_far=answers,
                candidate_cards=candidates,
                choose_top_k=3,
                read_all_windows_max_count=5,
                large_parent_top_windows=4,
                large_parent_neighbor_windows=1,
                previously_useful_documents=previously_useful,
            ),
        },
        {
            "name": "Document Read",
            "source": str(PIPELINERL_PROMPTS),
            "prompt": mod.build_doc_reader_prompt(
                numbered_questions=numbered,
                current_hop_number=1,
                current_hop_question="What was the Q1 2024 training budget at MediConn Solutions, rounded to the nearest dollar?",
                current_answers_so_far=answers,
                document=document,
            ),
        },
        {
            "name": "Hop Resolve",
            "source": str(PIPELINERL_PROMPTS),
            "prompt": mod.build_hop_resolve_prompt(
                numbered_questions=numbered,
                current_hop_number=1,
                current_hop_question="What was the Q1 2024 training budget at MediConn Solutions, rounded to the nearest dollar?",
                current_answers_so_far=answers,
                search_history=search_history,
                reader_results=reader_results,
                unread_candidate_cards=candidates[1:],
                choose_top_k=3,
                constrained=False,
            ),
        },
    ]


def build_adversary_prompts(sample: dict[str, Any]) -> list[dict[str, str]]:
    ns = load_leakage_prompt_namespace()
    queries = sample.get("queries") or []
    private_qas = sample.get("private_qas") or []
    max_items = max(2, min(10, 2 * max(1, len(private_qas))))
    system = load_string_constants(LEAKAGE_PROMPTS, {"SYSTEM"})["SYSTEM"]
    return [
        {
            "name": "Shared System Prompt",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": system,
        },
        {
            "name": "Intent Adversary",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": ns["prompt_intent_adversary"](queries, max_items),
        },
        {
            "name": "Answer Adversary",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": ns["prompt_answer_adversary"](queries, private_qas),
        },
        {
            "name": "Full-Information Claim Adversary",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": ns["prompt_claim_adversary"](queries, max_items),
        },
    ]


def build_dataset_prompts() -> list[dict[str, str]]:
    secret = load_string_constants(
        SECRET_PROMPTS,
        {"PROMPT_TEMPLATE", "ANSWER_WITH_DOC_TEMPLATE", "ANSWER_WITHOUT_DOC_TEMPLATE"},
    )
    trajectory = load_string_constants(
        TRAJECTORY_PROMPTS,
        {
            "PROMPT_CONSTRAINED",
            "PROMPT_PICK",
            "PROMPT_RANK",
            "PROMPT_TRANSITION_JUDGE",
            "REWORD_PROMPT",
            "CHAIN_VERIFICATION_TEMPLATE",
        },
    )
    web_seed = load_string_constants(WEB_SEED_PROMPTS, {"PROMPT_WEB_SEED"})
    prompts: list[dict[str, str]] = [
        {
            "name": "Private QA Extraction",
            "source": str(SECRET_PROMPTS),
            "prompt": secret["PROMPT_TEMPLATE"],
        },
        {
            "name": "Document-Only Answerability Check",
            "source": str(SECRET_PROMPTS),
            "prompt": secret["ANSWER_WITH_DOC_TEMPLATE"],
        },
        {
            "name": "No-Document Memorization Check",
            "source": str(SECRET_PROMPTS),
            "prompt": secret["ANSWER_WITHOUT_DOC_TEMPLATE"],
        },
        {
            "name": "Web Seed Question",
            "source": str(WEB_SEED_PROMPTS),
            "prompt": web_seed["PROMPT_WEB_SEED"],
        },
    ]
    for key, label in [
        ("PROMPT_CONSTRAINED", "Next-Hop Question, Fixed Target"),
        ("PROMPT_PICK", "Next-Hop Question, Pick Target"),
        ("PROMPT_RANK", "Candidate Hop Ranker"),
        ("PROMPT_TRANSITION_JUDGE", "Transition Judge"),
        ("REWORD_PROMPT", "Question Rewording"),
        ("CHAIN_VERIFICATION_TEMPLATE", "Chain Verification"),
    ]:
        prompts.append({"name": label, "source": str(TRAJECTORY_PROMPTS), "prompt": trajectory[key]})
    return prompts


def esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def prompt_card(item: dict[str, str], *, open_details: bool = True) -> str:
    detail_open = " open" if open_details else ""
    return f"""
    <details class="prompt-card"{detail_open} id="{slug(item['name'])}">
      <summary><span>{esc(item['name'])}</span><code>{esc(Path(item['source']).name)}</code></summary>
      <pre>{esc(item['prompt'])}</pre>
    </details>
    """


def json_panel(title: str, payload: Any, *, limit: int | None = 6000) -> str:
    return f"""
    <figure class="json-panel">
      <figcaption>{esc(title)}</figcaption>
      <pre>{esc(compact_json(payload, limit=limit))}</pre>
    </figure>
    """


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def render_html(data: dict[str, Any]) -> str:
    private_rows = [
        [row["secret_type"], row["question"], row["answer"], row["doc_id"]]
        for row in data["private_qa_examples"]
    ]
    test_rows = [
        [
            row["task_id"],
            row["chain_id"],
            row["company"],
            row["pattern"],
            row["hops"],
            row["first_question"],
            row["final_question"],
            row["final_answer"],
        ]
        for row in data["test_set_examples"]
    ]
    research_outputs = data["research_sample_outputs"]
    adversary_outputs = data["leakage_sample"]["outputs"]
    source_rows = [[name, path] for name, path in data["sources"].items()]

    dataset_prompt_cards = "\n".join(prompt_card(item, open_details=(idx < 4)) for idx, item in enumerate(data["dataset_prompts"]))
    research_prompt_cards = "\n".join(prompt_card(item) for item in data["research_prompts"])
    adversary_prompt_cards = "\n".join(prompt_card(item) for item in data["adversary_prompts"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy HopQA Prompt Appendix</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5f6d7c;
      --rule: #d7dde7;
      --paper: #fbfbf8;
      --band: #eef3f6;
      --panel: #ffffff;
      --blue: #2457a6;
      --green: #166b54;
      --gold: #9a6500;
      --red: #a33a2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 42px 48px 28px;
      border-bottom: 1px solid var(--rule);
      background: linear-gradient(180deg, #ffffff 0%, var(--band) 100%);
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 34px; font-weight: 760; }}
    h2 {{ font-size: 24px; margin-bottom: 14px; }}
    h3 {{ font-size: 17px; margin: 22px 0 10px; }}
    p {{ max-width: 980px; color: var(--muted); margin: 10px 0 0; }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 22px;
    }}
    nav a {{
      color: #fff;
      background: var(--blue);
      padding: 6px 10px;
      border-radius: 6px;
      text-decoration: none;
      font-size: 13px;
      font-weight: 650;
    }}
    main {{ padding: 0; }}
    section {{
      padding: 34px 48px;
      border-bottom: 1px solid var(--rule);
    }}
    section:nth-child(even) {{ background: #fff; }}
    .kicker {{
      color: var(--green);
      font-weight: 760;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: .08em;
      margin-bottom: 8px;
    }}
    .note {{
      border-left: 4px solid var(--gold);
      padding: 10px 14px;
      background: #fff8e7;
      color: #4b3a18;
      margin: 16px 0;
      max-width: 1060px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    details.prompt-card {{
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 8px;
      overflow: hidden;
      break-inside: avoid;
    }}
    details.prompt-card summary {{
      cursor: pointer;
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      font-weight: 760;
      background: #f4f7fa;
      border-bottom: 1px solid var(--rule);
    }}
    details.prompt-card summary code {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.42;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      color: #0e1726;
      background: #ffffff;
    }}
    .json-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
      align-items: start;
      margin-top: 16px;
    }}
    .json-panel {{
      margin: 0;
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 8px;
      overflow: hidden;
      break-inside: avoid;
    }}
    figcaption {{
      padding: 10px 14px;
      background: #f5f1e8;
      color: #362713;
      border-bottom: 1px solid var(--rule);
      font-weight: 760;
      font-size: 13px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--rule);
      border-radius: 8px;
      background: #fff;
      margin-top: 14px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 900px;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      padding: 9px 10px;
      border-bottom: 1px solid var(--rule);
    }}
    th {{
      background: #eef3f6;
      color: #27364a;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    td:nth-child(1), td:nth-child(2), td:nth-child(4), td:nth-child(5) {{
      white-space: nowrap;
    }}
    .source-table table {{ min-width: 720px; }}
    @media print {{
      body {{ background: #fff; }}
      header, section {{ padding: 22px 28px; }}
      nav {{ display: none; }}
      details.prompt-card {{ break-inside: avoid; }}
      details.prompt-card:not([open]) > *:not(summary) {{ display: block; }}
    }}
  </style>
</head>
<body>
<header>
  <div class="kicker">Paper Appendix Artifact</div>
  <h1>Privacy HopQA Prompts and Sample Outputs</h1>
  <p>Self-contained view of the prompt templates, rendered research-agent step prompts, adversary prompts, and representative outputs used for the Privacy HopQA work. Generated from source files and saved artifacts, not manually pasted.</p>
  <nav>
    <a href="#dataset">Private QA Generation</a>
    <a href="#research-agent">Research Agent Steps</a>
    <a href="#adversary">Adversary Prompts</a>
    <a href="#sources">Sources</a>
  </nav>
</header>
<main>
  <section id="dataset">
    <div class="kicker">Dataset Construction</div>
    <h2>Private QA Generation and Test Examples</h2>
    <p>The private QA inventory is extracted from local/internal documents, then checked with and without document context. The final Privacy HopQA test split uses those private/local facts as chain hops mixed with public web hops.</p>
    <div class="note">For screenshots, the first four prompt cards are open by default. The remaining QA-chain construction prompts are included below as expandable cards.</div>
    <div class="grid">{dataset_prompt_cards}</div>
    <h3>Private QA Inventory Examples</h3>
    {table(["Type", "Question", "Answer", "Document"], private_rows)}
    <h3>Final Test Split Examples</h3>
    {table(["Task", "Chain", "Company", "Pattern", "Hops", "First Question", "Final Question", "Final Answer"], test_rows)}
    <div class="json-grid">
      {json_panel("Sample Private QA Extraction Output", data["private_qa_examples"][0] if data["private_qa_examples"] else {}, limit=2600)}
    </div>
  </section>

  <section id="research-agent">
    <div class="kicker">Rollout Harness</div>
    <h2>Research Agent Step Prompts</h2>
    <p>These are rendered examples of the PipelineRL Privacy HopQA prompt builders for the four LLM-controlled research-agent stages. The JSON panels show parsed outputs from a saved test rollout.</p>
    <div class="grid">{research_prompt_cards}</div>
    <h3>Sample Outputs by Step</h3>
    <div class="json-grid">
      {json_panel("Hop Plan Output", research_outputs.get("hop_plan"), limit=3200)}
      {json_panel("Document Choose Output", research_outputs.get("doc_choose"), limit=3200)}
      {json_panel("Document Read Output", research_outputs.get("doc_read"), limit=3600)}
      {json_panel("Hop Resolve Output", research_outputs.get("hop_resolve"), limit=3000)}
      {json_panel("Final Report Excerpt", research_outputs.get("final_report_excerpt"), limit=2600)}
    </div>
  </section>

  <section id="adversary">
    <div class="kicker">Privacy Evaluation</div>
    <h2>Adversary Prompts and Outputs</h2>
    <p>The leakage suite uses separate adversary calls for inferred intent, targeted Private QA answering, and free-form private claim generation. The answer adversary sees Private QA questions but not their answers; the other adversaries see only visible web queries.</p>
    <div class="note">The rendered prompts use the current source file. The sample outputs come from a saved StepFun leakage-evaluation row (runner {esc(data["leakage_sample"].get("runner_version"))}, prompt {esc(data["leakage_sample"].get("prompt_version"))}): {esc(data["leakage_sample"].get("source"))}.</div>
    <div class="grid">{adversary_prompt_cards}</div>
    <h3>Visible Inputs and Sample Outputs</h3>
    <div class="json-grid">
      {json_panel("Visible Web Queries", data["leakage_sample"].get("queries"), limit=4200)}
      {json_panel("Private Questions Shown to Answer Adversary", data["leakage_sample"].get("private_qas"), limit=5000)}
      {json_panel("Intent Adversary Output", adversary_outputs.get("intent_adversary"), limit=4600)}
      {json_panel("Answer Adversary Output", adversary_outputs.get("answer_adversary"), limit=4600)}
      {json_panel("Claim Adversary Output", adversary_outputs.get("claim_adversary"), limit=4600)}
      {json_panel("Overall/Judge Output", {"answer_judge": adversary_outputs.get("answer_judge"), "claim_judge": adversary_outputs.get("claim_judge"), "overall": adversary_outputs.get("overall")}, limit=5200)}
    </div>
  </section>

  <section id="sources" class="source-table">
    <div class="kicker">Traceability</div>
    <h2>Source Files and Artifacts</h2>
    <p>These paths are included so paper screenshots can be traced back to source code or saved experiment output.</p>
    {table(["Source", "Path"], source_rows)}
  </section>
</main>
</body>
</html>
"""


def build_data() -> dict[str, Any]:
    leakage_sample = load_leakage_sample()
    data = {
        "sources": {
            "dataset_private_qa_prompts": str(SECRET_PROMPTS),
            "qa_chain_prompts": str(TRAJECTORY_PROMPTS),
            "web_seed_prompt": str(WEB_SEED_PROMPTS),
            "research_agent_prompts": str(PIPELINERL_PROMPTS),
            "adversary_prompts": str(LEAKAGE_PROMPTS),
            "private_qa_inventory": str(PRIVATE_QA_INVENTORY),
            "final_test_split": str(FINAL_TEST_SET),
            "research_sample_trace": str(RESEARCH_SAMPLE_TRACE),
            "leakage_sample": str(leakage_sample.get("source")),
        },
        "dataset_prompts": build_dataset_prompts(),
        "research_prompts": build_research_prompts(),
        "adversary_prompts": build_adversary_prompts(leakage_sample),
        "private_qa_examples": load_private_qa_examples(),
        "test_set_examples": load_test_set_examples(),
        "research_sample_outputs": load_research_sample_outputs(),
        "leakage_sample": leakage_sample,
    }
    return data


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_HTML.write_text(render_html(data), encoding="utf-8")
    OUT_README.write_text(
        "\n".join(
            [
                "# Privacy HopQA Prompt Appendix",
                "",
                "Generated paper-facing prompt appendix for Privacy HopQA.",
                "",
                "- `index.html`: screenshot-friendly single-page view.",
                "- `prompt_appendix_data.json`: extracted prompts, examples, and sample outputs.",
                "",
                "Regenerate from the MosaicProject root with:",
                "",
                "```bash",
                "python3 scripts/build_privacy_hopqa_prompt_appendix.py",
                "```",
                "",
                "The generator reads source prompt files in this repository plus the PipelineRL Privacy HopQA prompt builders and saved runtime/leakage artifacts listed at the bottom of `index.html`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUT_HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
