#!/usr/bin/env python3
"""Build a paper-friendly prompt appendix for Privacy HopQA."""

from __future__ import annotations

import ast
import html
import importlib.util
import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "privacy_hopqa_prompts"
OUT_HTML = OUT_DIR / "index.html"
OUT_JSON = OUT_DIR / "prompt_appendix_data.json"
OUT_README = OUT_DIR / "README.md"
PROMPT_TEX_DIR = OUT_DIR / "latex_prompt_snippets"
PROMPT_OUTPUT_TEX_DIR = OUT_DIR / "latex_prompt_output_snippets"
PROMPT_SHORT_TEX_DIR = OUT_DIR / "latex_prompt_short_snippets"
TABLE_TEX_DIR = OUT_DIR / "latex_table_snippets"
PROMPT_PNG_DIR = OUT_DIR / "overleaf_prompt_pngs"
PROMPT_OUTPUT_PNG_DIR = OUT_DIR / "overleaf_prompt_output_pngs"
PROMPT_SHORT_PNG_DIR = OUT_DIR / "overleaf_prompt_short_pngs"
DATASET_TABLE_DIR = ROOT / "paper_assets" / "privacy_hopqa_dataset_tables"
DATASET_STATS_JSON = DATASET_TABLE_DIR / "dataset_stats.json"

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
PLACEHOLDER_RE = re.compile(r"(<<[A-Z0-9_ -]+>>|\{[A-Za-z_][A-Za-z0-9_]*\})")
DOC_ID_RE = re.compile(r"\bdoc_[0-9a-f]{8,}_[0-9]+\b")
LOCAL_PATH_RE = re.compile(r"/(?:home|mnt)/[^\s\"']+")


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


def sanitize_research_output(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "locator":
                sanitized[key] = "<<EVIDENCE_LOCATOR_OR_URL>>"
            elif key in {"doc_id", "parent_doc_id", "evidence_window_id"}:
                sanitized[key] = "<<EVIDENCE_DOC_ID>>"
            elif key == "title" and isinstance(item, str) and "." in item:
                sanitized[key] = "<<EVIDENCE_DOCUMENT_TITLE>>"
            else:
                sanitized[key] = sanitize_research_output(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_research_output(item) for item in value]
    if isinstance(value, str):
        text = LOCAL_PATH_RE.sub("<<EVIDENCE_LOCATOR_OR_URL>>", value)
        text = DOC_ID_RE.sub("<<EVIDENCE_DOC_ID>>", text)
        text = text.replace("MediConn_Quarterly_Financial_Overview.xlsx", "<<EVIDENCE_DOCUMENT_TITLE>>")
        return text
    return value


def truncate_mosaic_icl_exact(prompt: str, *, positive_count: int = 2, negative_count: int = 1) -> str:
    match = re.search(r"\n(MOSAIC_LEAKAGE_EXAMPLES\n.*?)(\n\nVISIBLE_WEB_QUERIES)", prompt, flags=re.DOTALL)
    if not match:
        return prompt
    block = match.group(1)
    lines = block.splitlines()
    positive_lines: list[str] = []
    negative_lines: list[str] = []
    target: list[str] | None = None
    for line in lines:
        if line == "Positive examples:":
            target = positive_lines
            continue
        if line == "Negative examples:":
            target = negative_lines
            continue
        if target is not None and line.startswith("-"):
            target.append(line)
    compact_lines = [
        "MOSAIC_LEAKAGE_EXAMPLES",
        "Positive examples:",
        *positive_lines[:positive_count],
    ]
    if len(positive_lines) > positive_count:
        compact_lines.append("<<ADDITIONAL_POSITIVE_ICL_EXAMPLES_OMITTED_FOR_SPACE>>")
    compact_lines.extend(["Negative examples:", *negative_lines[:negative_count]])
    if len(negative_lines) > negative_count:
        compact_lines.append("<<ADDITIONAL_NEGATIVE_ICL_EXAMPLES_OMITTED_FOR_SPACE>>")
    return re.sub(
        r"\nMOSAIC_LEAKAGE_EXAMPLES\n.*?\n\nVISIBLE_WEB_QUERIES",
        "\n" + "\n".join(compact_lines) + "\n\nVISIBLE_WEB_QUERIES",
        prompt,
        flags=re.DOTALL,
    )


def truncate_intent_icl(prompt: str) -> str:
    return truncate_mosaic_icl_exact(prompt)


def truncate_full_information_icl(prompt: str) -> str:
    return truncate_mosaic_icl_exact(prompt)


def truncate_intent_output(output: Any, max_items: int = 3) -> Any:
    if not isinstance(output, dict):
        return output
    predictions = output.get("predicted_subquestions")
    if not isinstance(predictions, list):
        return strip_stepfun_metadata(output)
    compact = [str(item) for item in predictions[:max_items]]
    if len(predictions) > max_items:
        compact.append("<<ADDITIONAL_PREDICTED_SUBQUESTIONS_OMITTED_FOR_SPACE>>")
    return {"predicted_subquestions": compact}


def answer_adversary_answers(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    answers = output.get("answers")
    if isinstance(answers, list) and answers:
        return [item for item in answers if isinstance(item, dict)]
    if output.get("qa_id") and output.get("answer"):
        return [
            {
                "qa_id": output.get("qa_id"),
                "answer": output.get("answer"),
                "confidence": output.get("confidence") or "medium",
            }
        ]
    return []


def normalize_answer_adversary_output(output: Any) -> dict[str, Any]:
    answers = []
    for item in answer_adversary_answers(output):
        answers.append(
            {
                "qa_id": item.get("qa_id"),
                "answer": item.get("answer"),
                "confidence": item.get("confidence") or "medium",
            }
        )
    return {"answers": answers}


PRIVATE_QA_TYPE_ORDER = ["kpi_numeric", "dates", "money", "names", "other_sensitive", "ids", "emails"]
AWKWARD_EXAMPLE_TERMS = [
    "at the start of",
    "at the beginning of",
    "beginning of the article",
    "start of the article",
    "opening sentence",
    "article",
    "passage",
    "excerpt",
    "according to",
    "document",
    "sentence",
    "paragraph",
    "quote",
    "introduced before",
]


def short_private_qa_example(secret_type: str, question: str, answer: str) -> bool:
    if secret_type == "emails":
        return len(answer) <= 45 and ";" not in answer and "," not in answer and len(question) <= 150
    if secret_type == "ids":
        return len(answer) <= 60 and answer.count(";") <= 2 and len(question) <= 150
    if len(answer) > 45 or len(question) > 150:
        return False
    if any(separator in answer for separator in [";", "\n", " / "]):
        return False
    listy_question_terms = ["names of", "usernames", "users in", "user directory", "email directory", "mattermost export"]
    if any(term in question.casefold() for term in listy_question_terms):
        return False
    return True


def load_private_qa_examples(max_rows: int = len(PRIVATE_QA_TYPE_ORDER)) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in read_jsonl(PRIVATE_QA_INVENTORY):
        doc_id = str(row.get("doc_id") or row.get("chunk_id") or "")
        for secret in row.get("secrets") or []:
            secret_type = str(secret.get("secret_type") or secret.get("type") or "other_sensitive")
            answer = str(secret.get("answer") or "")
            question = str(secret.get("question") or "")
            if not short_private_qa_example(secret_type, question, answer):
                continue
            candidates.append(
                {
                    "doc_id": doc_id,
                    "secret_type": secret_type,
                    "question": question,
                    "answer": answer,
                    "doc_only_check": compact_json(secret.get("doc_only_check") or {}, limit=220),
                }
            )

    examples: list[dict[str, str]] = []
    used_questions: set[str] = set()
    for secret_type in PRIVATE_QA_TYPE_ORDER:
        match = next((item for item in candidates if item["secret_type"] == secret_type and item["question"] not in used_questions), None)
        if match:
            examples.append(match)
            used_questions.add(match["question"])
        if len(examples) >= max_rows:
            return examples

    for item in candidates:
        if item["question"] in used_questions:
            continue
        examples.append(item)
        used_questions.add(item["question"])
        if len(examples) >= max_rows:
            return examples
    return examples


def paper_friendly_example_question(question: Any) -> bool:
    text = str(question or "").casefold()
    return not any(term in text for term in AWKWARD_EXAMPLE_TERMS)


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
        final_question = last.get("question") or row.get("global_question")
        if not (
            paper_friendly_example_question(first.get("question"))
            and paper_friendly_example_question(final_question)
        ):
            continue
        examples.append(
            {
                "task_id": row.get("task_id"),
                "chain_id": row.get("chain_id"),
                "company": row.get("company"),
                "pattern": pattern,
                "hops": len(hops),
                "first_question": first.get("question"),
                "final_question": final_question,
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

    doc_choose_meta = by_stage.get("doc_choose") or {}
    selected_doc_ids = doc_choose_meta.get("selected_doc_ids") or doc_choose_meta.get("selected_parent_doc_ids") or []
    doc_choose_output = {
        "selected_doc_ids": [
            f"<<SELECTED_PARENT_DOC_ID_{idx}>>"
            for idx, _ in enumerate(selected_doc_ids[:2] or ["placeholder"], start=1)
        ]
    }

    return {
        "source_trace": str(RESEARCH_SAMPLE_TRACE),
        "hop_plan": sanitize_research_output(by_stage.get("hop_plan") or {}),
        "doc_choose": doc_choose_output,
        "doc_read": sanitize_research_output(positive_reader),
        "hop_resolve": sanitize_research_output({
            "answered": first_hop.get("status") == "answered",
            "answer": first_hop.get("answer"),
            "confidence": first_hop.get("confidence"),
            "justification": first_hop.get("justification"),
            "next_step": "done" if first_hop.get("status") == "answered" else "search_more",
        }),
        "final_report_excerpt": report_text[:1400] if report_text else "",
    }


def load_leakage_sample() -> dict[str, Any]:
    if LATEST_LEAKAGE_SAMPLE.exists():
        rows = read_jsonl(LATEST_LEAKAGE_SAMPLE)
        def leaked(row: dict[str, Any]) -> bool:
            return int(row.get("overall_privacy_score_1_to_5") or 0) >= 4 or bool(row.get("overall_leaked_for_reward"))

        chosen = next(
            (
                row
                for row in rows
                if leaked(row) and answer_adversary_answers(row.get("answer_adversary_output"))
            ),
            next((row for row in rows if leaked(row)), rows[0] if rows else {}),
        )
        hidden = chosen.get("hidden_reference_for_judges_only") or {}
        return {
            "source": str(LATEST_LEAKAGE_SAMPLE),
            "runner_version": chosen.get("runner_version"),
            "prompt_version": chosen.get("prompt_version"),
            "model": chosen.get("model"),
            "rollout_id": chosen.get("rollout_id"),
            "queries": (chosen.get("model_input") or {}).get("web_queries") or [],
            "private_qas": hidden.get("private_qas") or [],
            "local_facts": hidden.get("local_hop_facts") or [],
            "outputs": {
                "intent_adversary": truncate_intent_output(chosen.get("intent_adversary_output")),
                "answer_adversary": normalize_answer_adversary_output(chosen.get("answer_adversary_output")),
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
        "local_facts": [],
        "outputs": {
            **(chosen.get("outputs") or {}),
            "intent_adversary": truncate_intent_output((chosen.get("outputs") or {}).get("intent_adversary")),
            "answer_adversary": normalize_answer_adversary_output((chosen.get("outputs") or {}).get("answer_adversary")),
        },
    }


def build_research_prompts() -> list[dict[str, str]]:
    mod = load_privacy_hopqa_prompt_module()
    numbered = "<<FULL_NUMBERED_QUESTION_CHAIN>>"
    answers = "<<CURRENT_ANSWERS_SO_FAR>>"
    previously_useful = [
        {
            "doc_id": "<<PREVIOUSLY_USEFUL_DOC_ID>>",
            "title": "<<PREVIOUSLY_USEFUL_DOCUMENT_TITLE>>",
            "source": "<<SOURCE_KIND>>",
            "why_useful": "<<WHY_THIS_DOCUMENT_HELPED_EARLIER>>",
        }
    ]
    candidates = [
        {
            "doc_id": "<<CANDIDATE_PARENT_DOC_ID>>",
            "source": "<<LOCAL_OR_WEB>>",
            "title": "<<CANDIDATE_DOCUMENT_TITLE>>",
            "matched_window_count": "<<N_MATCHED_WINDOWS>>",
            "total_window_count": "<<N_TOTAL_WINDOWS>>",
            "best_rank": "<<BEST_RETRIEVAL_RANK>>",
            "top_queries": ["<<RETRIEVAL_QUERY_THAT_FOUND_THIS_DOC>>"],
            "excerpt": "<<SHORT_RETRIEVED_WINDOW_EXCERPT>>",
        },
        {
            "doc_id": "<<SECOND_CANDIDATE_PARENT_DOC_ID>>",
            "source": "<<LOCAL_OR_WEB>>",
            "title": "<<SECOND_CANDIDATE_DOCUMENT_TITLE>>",
            "matched_window_count": "<<N_MATCHED_WINDOWS>>",
            "total_window_count": "<<N_TOTAL_WINDOWS>>",
            "best_rank": "<<BEST_RETRIEVAL_RANK>>",
            "top_queries": ["<<ANOTHER_RETRIEVAL_QUERY>>"],
            "excerpt": "<<ANOTHER_SHORT_RETRIEVED_WINDOW_EXCERPT>>",
        },
    ]
    document = {
        "doc_id": "<<EVIDENCE_DOC_ID>>",
        "source": "<<LOCAL_OR_WEB>>",
        "title": "<<EVIDENCE_DOCUMENT_TITLE>>",
        "locator": "<<EVIDENCE_LOCATOR_OR_URL>>",
        "text": "<<FULL_EVIDENCE_WINDOW_TEXT>>",
    }
    reader_results = [
        {
            "doc_id": "<<EVIDENCE_DOC_ID>>",
            "source": "<<LOCAL_OR_WEB>>",
            "title": "<<EVIDENCE_DOCUMENT_TITLE>>",
            "can_answer": True,
            "proposed_answer": "<<PROPOSED_SHORT_ANSWER>>",
            "justification": "<<DOCUMENT_READER_JUSTIFICATION_WITH_DOC_CITATION>>",
            "confidence": "<<CONFIDENCE_SCORE>>",
        }
    ]
    search_history = [
        {
            "iteration": "<<ITERATION_INDEX>>",
            "type": "<<RETRIEVAL_ACTION_TYPE>>",
            "query": "<<VISIBLE_OR_LOCAL_RETRIEVAL_QUERY>>",
            "result_count": "<<N_RESULTS>>",
        }
    ]
    return [
        {
            "name": "Hop Plan",
            "source": str(PIPELINERL_PROMPTS),
            "prompt": mod.build_hop_plan_prompt(
                numbered_questions=numbered,
                current_hop_number=1,
                current_hop_question="<<CURRENT_HOP_QUESTION>>",
                current_answers_so_far=answers,
                search_history=[],
                recent_reader_results=[],
                company_name="<<COMPANY_NAME>>",
                company_description="<<COMPANY_DESCRIPTION>>",
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
                current_hop_question="<<CURRENT_HOP_QUESTION>>",
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
                current_hop_question="<<CURRENT_HOP_QUESTION>>",
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
                current_hop_question="<<CURRENT_HOP_QUESTION>>",
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
    queries = [
        {"index": 1, "query": "<<VISIBLE_WEB_QUERY_1>>"},
        {"index": 2, "query": "<<VISIBLE_WEB_QUERY_2>>"},
        {"index": "<<N>>", "query": "<<VISIBLE_WEB_QUERY_N>>"},
    ]
    private_qas = [
        {
            "qa_id": "<<PRIVATE_QA_ID>>",
            "question": "<<PRIVATE_QA_QUESTION_TEXT>>",
            "secret_type": "<<SECRET_TYPE>>",
        }
    ]
    max_items = 10
    system = load_string_constants(LEAKAGE_PROMPTS, {"SYSTEM"})["SYSTEM"]
    intent_prompt = truncate_intent_icl(
        ns["prompt_intent_adversary"](queries, max_items).replace(
            "up to 10 subquestions",
            "up to <<MAX_OUTPUT_ITEMS>> subquestions",
        )
    )
    claim_prompt = truncate_full_information_icl(
        ns["prompt_claim_adversary"](queries, max_items).replace(
            "up to 10 claim groups",
            "up to <<MAX_OUTPUT_ITEMS>> claim groups",
        )
    )
    return [
        {
            "name": "Shared System Prompt",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": system,
        },
        {
            "name": "Intent Adversary",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": intent_prompt,
        },
        {
            "name": "Answer Adversary",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": ns["prompt_answer_adversary"](queries, private_qas),
        },
        {
            "name": "Full-Information Claim Adversary",
            "source": str(LEAKAGE_PROMPTS),
            "prompt": claim_prompt,
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


def latex_escape(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text if text is not None else ""))


def latex_escape_with_placeholders(text: Any) -> str:
    raw = str(text if text is not None else "")
    parts: list[str] = []
    cursor = 0
    for match in PLACEHOLDER_RE.finditer(raw):
        parts.append(latex_escape(raw[cursor : match.start()]))
        parts.append(r"\textcolor{my-green}{\textbf{" + latex_escape(match.group(0)) + r"}}")
        cursor = match.end()
    parts.append(latex_escape(raw[cursor:]))
    return "".join(parts)


def html_with_placeholders(text: Any) -> str:
    raw = str(text if text is not None else "")
    parts: list[str] = []
    cursor = 0
    for match in PLACEHOLDER_RE.finditer(raw):
        parts.append(esc(raw[cursor : match.start()]))
        parts.append(f'<span class="placeholder">{esc(match.group(0))}</span>')
        cursor = match.end()
    parts.append(esc(raw[cursor:]))
    return "".join(parts)


def latex_prompt_body(text: Any) -> str:
    lines = str(text if text is not None else "").splitlines()
    if not lines:
        return r"\emph{empty}\par"
    return "\n".join((latex_escape_with_placeholders(line) if line else r"\mbox{}") + r"\par" for line in lines)


def tcolorbox_latex(title: str, prompt: str, sample_output: Any | None = None, *, short_note: bool = False) -> str:
    parts = [
        r"\begin{tcolorbox}[colback=white, colframe=my-green, coltitle=black, parskip=5mm, title=\textbf{"
        + latex_escape(title)
        + r"}, halign=flush left, enhanced, breakable, skin first=enhanced, skin middle=enhanced, skin last=enhanced]",
        r"\begin{scriptsize}",
        r"\ttfamily",
        r"\setlength{\parindent}{0pt}",
    ]
    if short_note:
        parts.extend(
            [
                r"\normalfont\textit{Short paper variant; in-context examples omitted for space.}\par",
                r"\medskip",
                r"\ttfamily",
            ]
        )
    parts.append(latex_prompt_body(prompt))
    if sample_output is not None:
        parts.extend(
            [
                r"\medskip",
                r"\normalfont\textbf{Sample output}\par",
                r"\ttfamily",
                latex_prompt_body(sample_output if isinstance(sample_output, str) else compact_json(sample_output)),
            ]
        )
    parts.extend([r"\end{scriptsize}", r"\end{tcolorbox}", ""])
    return "\n".join(parts)


def write_text_clean(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    if text.endswith("\n"):
        cleaned += "\n"
    path.write_text(cleaned, encoding="utf-8")


def short_prompt_variant(prompt: str) -> str | None:
    short = prompt
    short = re.sub(
        r"\nMOSAIC_LEAKAGE_EXAMPLES\n.*?\n\nVISIBLE_WEB_QUERIES",
        "\nMOSAIC_LEAKAGE_EXAMPLES\n<<ICL_EXAMPLES_OMITTED_FOR_SPACE>>\n\nVISIBLE_WEB_QUERIES",
        short,
        flags=re.DOTALL,
    )
    short = re.sub(
        r"\nGood example:\n.*?(?=\nEnd with exactly one final block)",
        "\n<<ICL_EXAMPLES_OMITTED_FOR_SPACE>>\n\n",
        short,
        flags=re.DOTALL,
    )
    short = re.sub(
        r"\n- Example: for .+?(?=\n- |\n\n)",
        "\n",
        short,
        flags=re.DOTALL,
    )
    short = re.sub(
        r"\nOutput only a comma-separated ranking of candidate numbers, for example:\n[^\n]+\n",
        "\nOutput only a comma-separated ranking of candidate numbers.\n",
        short,
    )
    short = re.sub(r"\n{3,}", "\n\n", short).strip() + "\n"
    if len(prompt) - len(short) < 280:
        return None
    return short


def extract_claim_texts(output: Any) -> list[str]:
    if not isinstance(output, dict):
        return []
    claims = output.get("claims")
    if isinstance(claims, list):
        return [str(item) for item in claims if str(item).strip()]
    groups = output.get("claim_groups")
    out: list[str] = []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            if group.get("base_claim"):
                out.append(str(group["base_claim"]))
            for claim in group.get("more_specific_claims") or []:
                out.append(str(claim))
    return out


def matched_private_facts(sample: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = sample.get("outputs") or {}
    refs = []
    refs.extend(sample.get("local_facts") or [])
    refs.extend(sample.get("private_qas") or [])
    by_id = {
        str(item.get("fact_id") or item.get("qa_id")): item
        for item in refs
        if item.get("fact_id") or item.get("qa_id")
    }
    matched_ids: list[str] = []
    for key in ["matched_fact_ids", "partial_private_fact_ids", "value_context_private_fact_ids"]:
        values = (outputs.get("claim_judge") or {}).get(key) or []
        matched_ids.extend(str(value) for value in values)
    for key in ["correct_qa_ids", "partial_qa_ids", "value_context_qa_ids"]:
        values = (outputs.get("answer_judge") or {}).get(key) or []
        matched_ids.extend(str(value) for value in values)
    matched = [by_id[item_id] for item_id in matched_ids if item_id in by_id]
    if matched:
        return matched
    return refs[:1]


def itemize_latex(items: list[Any]) -> str:
    if not items:
        return r"\emph{None recorded.}"
    lines = [r"\begin{itemize}[leftmargin=*,nosep]"]
    for item in items:
        lines.append("    " + r"\item " + latex_escape(item))
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def table_star_latex(
    *,
    title: str,
    label: str,
    headers: list[str],
    rows: list[list[Any]],
    widths: list[float],
    caption: str,
) -> str:
    colspec = "".join(f"p{{{width:.2f}\\linewidth}}" for width in widths)
    body = []
    body.append(r"\begin{table*}[t]")
    body.append(r"\centering")
    body.append(r"\small")
    body.append(r"\begin{tabular}{" + colspec + "}")
    body.append(r"\toprule")
    body.append(" & ".join(r"\textbf{" + latex_escape(header) + "}" for header in headers) + r" \\")
    body.append(r"\midrule")
    for row in rows:
        body.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\caption{" + latex_escape(caption) + "}")
    body.append(r"\label{" + label + "}")
    body.append(r"\end{table*}")
    body.append("")
    return "\n".join(body)


def leakage_example_table_latex(sample: dict[str, Any]) -> str:
    queries = [row.get("query") for row in sample.get("queries") or [] if row.get("query")]
    claims = extract_claim_texts((sample.get("outputs") or {}).get("claim_adversary"))
    facts = matched_private_facts(sample)
    fact_lines = []
    for fact in facts:
        answer = fact.get("canonical_answer") or fact.get("answer") or ""
        question = fact.get("question") or fact.get("fact_id") or fact.get("qa_id") or ""
        fact_lines.append(latex_escape(question) + r" \textbf{" + latex_escape(answer) + "}")
    if not fact_lines:
        fact_lines = [r"\emph{No matched fact recorded in sample row.}"]

    body = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\begin{tabular}{p{0.92\linewidth}}",
        r"\toprule",
        r"\textbf{Visible web queries} \\",
        r"\midrule",
        itemize_latex(queries),
        r"\\",
        r"\midrule",
        r"\textbf{Adversary claims from above web queries} \\",
        r"\midrule",
        itemize_latex(claims),
        r"\\",
        r"\midrule",
        r"\textbf{Local fact, judged as leaked} \\",
        r"\midrule",
        r"\\ ".join(fact_lines) + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Example of mosaic-style privacy leakage. The adversary saw the provided web queries and generated claims; the judge compared against hidden local/private facts from retrieved documents and marked the fact as leaked.}",
        r"\label{tab:privacy_hopqa_mosaic_leak_example}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(body)


def load_dataset_table_snippets() -> list[dict[str, str]]:
    table_files = [
        ("dataset_summary_compact", "Dataset Summary Table"),
        ("dataset_split_summary", "Dataset Split Summary Table"),
        ("private_qa_secret_type_counts", "Private QA Secret-Type Counts Table"),
    ]
    snippets: list[dict[str, str]] = []
    for snippet_id, name in table_files:
        path = DATASET_TABLE_DIR / f"{snippet_id}.tex"
        if not path.exists():
            continue
        snippets.append(
            {
                "id": snippet_id,
                "name": name,
                "latex": path.read_text(encoding="utf-8"),
            }
        )
    return snippets


def load_dataset_stats() -> dict[str, Any]:
    if not DATASET_STATS_JSON.exists():
        return {}
    return json.loads(DATASET_STATS_JSON.read_text(encoding="utf-8"))


def build_table_snippets(data: dict[str, Any]) -> list[dict[str, str]]:
    private_rows = [
        [row["secret_type"], row["question"], row["answer"]]
        for row in data["private_qa_examples"]
    ]
    test_rows = [
        [row["chain_id"], row["pattern"], row["first_question"], row["final_question"], row["final_answer"]]
        for row in data["test_set_examples"][:5]
    ]
    snippets = [
        {
            "id": "private_qa_examples",
            "name": "Private QA Examples Table",
            "latex": table_star_latex(
                title="Private QA examples",
                label="tab:privacy_hopqa_private_qa_examples",
                headers=["Type", "Private question", "Answer"],
                rows=private_rows,
                widths=[0.13, 0.52, 0.25],
                caption="Short examples from the retrieved-document Private QA Set, with one row per secret type.",
            ),
        },
        {
            "id": "final_test_examples",
            "name": "Final Test Split Examples Table",
            "latex": table_star_latex(
                title="Final test examples",
                label="tab:privacy_hopqa_test_examples",
                headers=["Chain", "Pattern", "First question", "Final question", "Final answer"],
                rows=test_rows,
                widths=[0.10, 0.09, 0.28, 0.35, 0.10],
                caption="Example chains from the finalized Privacy HopQA test split.",
            ),
        },
        {
            "id": "mosaic_leakage_example",
            "name": "Mosaic Leakage Example Table",
            "latex": leakage_example_table_latex(data["leakage_sample"]),
        },
    ]
    snippets.extend(data.get("dataset_table_snippets") or [])
    return snippets


def prompt_sample_outputs(data: dict[str, Any]) -> dict[str, Any]:
    private = data["private_qa_examples"][0] if data["private_qa_examples"] else {}
    private_block = "\n".join(
        [
            f"Question: {private.get('question', '')}",
            f"Answer: {private.get('answer', '')}",
            f"Type: {private.get('secret_type', '')}",
            "Justification: Internal document-grounded fact.",
        ]
    )
    outputs = data["research_sample_outputs"]
    adversary = data["leakage_sample"].get("outputs") or {}
    return {
        "private-qa-extraction": private_block,
        "document-only-answerability-check": (private.get("doc_only_check") or ""),
        "no-document-memorization-check": "NOT_ANSWERABLE",
        "hop-plan": outputs.get("hop_plan"),
        "document-choose": outputs.get("doc_choose"),
        "document-read": outputs.get("doc_read"),
        "hop-resolve": outputs.get("hop_resolve"),
        "intent-adversary": adversary.get("intent_adversary"),
        "answer-adversary": adversary.get("answer_adversary"),
        "full-information-claim-adversary": adversary.get("claim_adversary"),
    }


def enrich_prompts(data: dict[str, Any]) -> None:
    all_prompts: list[dict[str, Any]] = []
    prompt_index = 1
    for category_key, category_label in [
        ("dataset_prompts", "Dataset"),
        ("research_prompts", "Research Agent"),
        ("adversary_prompts", "Adversary"),
    ]:
        for item in data[category_key]:
            item["category"] = category_label
            item["id"] = f"{prompt_index:02d}-{slug(category_label)}-{slug(item['name'])}"
            item["latex"] = tcolorbox_latex(item["name"], item["prompt"])
            item["tex_path"] = f"latex_prompt_snippets/{item['id']}.tex"
            item["png_path"] = f"overleaf_prompt_pngs/{item['id']}.png"
            short = short_prompt_variant(item["prompt"])
            if short:
                item["short_prompt"] = short
                item["latex_short"] = tcolorbox_latex(
                    f"{item['name']} (Short)",
                    short,
                    short_note=True,
                )
                item["tex_short_path"] = f"latex_prompt_short_snippets/{item['id']}-short.tex"
                item["png_short_path"] = f"overleaf_prompt_short_pngs/{item['id']}-short.png"
            all_prompts.append(item)
            prompt_index += 1

    samples = prompt_sample_outputs(data)
    for item in all_prompts:
        output = samples.get(slug(item["name"]))
        if output is None:
            continue
        item["sample_output"] = output
        item["latex_with_output"] = tcolorbox_latex(
            f"{item['name']} with Sample Output",
            item["prompt"],
            output,
        )
        item["tex_output_path"] = f"latex_prompt_output_snippets/{item['id']}-with-output.tex"
        item["png_output_path"] = f"overleaf_prompt_output_pngs/{item['id']}-with-output.png"
    data["all_prompts"] = all_prompts
    data["table_snippets"] = build_table_snippets(data)


def try_import_pillow() -> Any | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    return Image, ImageDraw, ImageFont


def load_pil_font(ImageFont: Any, *, mono: bool, bold: bool, size: int) -> Any:
    names = []
    if mono:
        names.append("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    else:
        names.append("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_for_png(text: str, width: int = 112) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        if not line:
            out.append("")
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        wrapped = textwrap.wrap(
            line,
            width=width,
            subsequent_indent=indent + "  ",
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        out.extend(wrapped or [""])
    return out


def draw_png_text_with_placeholders(draw: Any, xy: tuple[int, int], line: str, *, font: Any, ink: tuple[int, int, int], placeholder: tuple[int, int, int]) -> None:
    x, y = xy
    cursor = 0
    for match in PLACEHOLDER_RE.finditer(line):
        before = line[cursor : match.start()]
        if before:
            draw.text((x, y), before, fill=ink, font=font)
            x += int(draw.textlength(before, font=font))
        token = match.group(0)
        draw.text((x, y), token, fill=placeholder, font=font)
        x += int(draw.textlength(token, font=font))
        cursor = match.end()
    tail = line[cursor:]
    if tail:
        draw.text((x, y), tail, fill=ink, font=font)


def render_prompt_png(
    title: str,
    prompt: str,
    path: Path,
    sample_output: Any | None = None,
    *,
    short_note: bool = False,
) -> bool:
    pil = try_import_pillow()
    if pil is None:
        return False
    Image, ImageDraw, ImageFont = pil
    width = 1800
    pad = 46
    title_h = 82
    border = 6
    font = load_pil_font(ImageFont, mono=True, bold=False, size=25)
    title_font = load_pil_font(ImageFont, mono=False, bold=True, size=34)
    label_font = load_pil_font(ImageFont, mono=False, bold=True, size=27)
    line_h = 36
    lines = wrap_for_png(prompt)
    if sample_output is not None:
        lines.extend(["", "Sample output", ""])
        rendered_output = sample_output if isinstance(sample_output, str) else compact_json(sample_output)
        lines.extend(wrap_for_png(str(rendered_output)))
    note_h = line_h + 10 if short_note else 0
    height = max(360, pad * 2 + title_h + note_h + len(lines) * line_h + 28)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    green = (22, 107, 84)
    rule = (215, 221, 231)
    ink = (23, 32, 51)
    muted = (76, 88, 104)
    placeholder = green
    draw.rectangle((border // 2, border // 2, width - border // 2, height - border // 2), outline=green, width=border)
    draw.rectangle((border, border, width - border, title_h), fill=(244, 247, 250), outline=rule, width=1)
    draw.text((pad, 26), title, fill=ink, font=title_font)
    y = title_h + 28
    if short_note:
        draw.text((pad, y), "Short paper variant; in-context examples omitted for space.", fill=green, font=label_font)
        y += line_h + 10
    for line in lines:
        if line == "Sample output":
            draw.text((pad, y + 4), line, fill=green, font=label_font)
        else:
            draw_png_text_with_placeholders(
                draw,
                (pad, y),
                line,
                font=font,
                ink=muted if not line else ink,
                placeholder=placeholder,
            )
        y += line_h
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True


def write_generated_assets(data: dict[str, Any]) -> None:
    for directory in [
        PROMPT_TEX_DIR,
        PROMPT_OUTPUT_TEX_DIR,
        PROMPT_SHORT_TEX_DIR,
        TABLE_TEX_DIR,
        PROMPT_PNG_DIR,
        PROMPT_OUTPUT_PNG_DIR,
        PROMPT_SHORT_PNG_DIR,
    ]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    all_prompt_tex: list[str] = []
    all_prompt_output_tex: list[str] = []
    all_prompt_short_tex: list[str] = []
    png_count = 0
    png_output_count = 0
    png_short_count = 0
    for item in data["all_prompts"]:
        write_text_clean(OUT_DIR / item["tex_path"], item["latex"])
        all_prompt_tex.append(item["latex"])
        if render_prompt_png(item["name"], item["prompt"], OUT_DIR / item["png_path"]):
            png_count += 1
        if item.get("latex_short"):
            write_text_clean(OUT_DIR / item["tex_short_path"], item["latex_short"])
            all_prompt_short_tex.append(item["latex_short"])
            if render_prompt_png(
                f"{item['name']} (Short)",
                item["short_prompt"],
                OUT_DIR / item["png_short_path"],
                short_note=True,
            ):
                png_short_count += 1
        if item.get("latex_with_output"):
            write_text_clean(OUT_DIR / item["tex_output_path"], item["latex_with_output"])
            all_prompt_output_tex.append(item["latex_with_output"])
            if render_prompt_png(
                f"{item['name']} with Sample Output",
                item["prompt"],
                OUT_DIR / item["png_output_path"],
                item.get("sample_output"),
            ):
                png_output_count += 1

    write_text_clean(OUT_DIR / "all_prompt_tcolorboxes.tex", "\n".join(all_prompt_tex))
    write_text_clean(OUT_DIR / "all_prompt_output_tcolorboxes.tex", "\n".join(all_prompt_output_tex))
    write_text_clean(OUT_DIR / "all_prompt_short_tcolorboxes.tex", "\n".join(all_prompt_short_tex))

    table_tex: list[str] = []
    for item in data["table_snippets"]:
        path = TABLE_TEX_DIR / f"{item['id']}.tex"
        item["tex_path"] = str(path.relative_to(OUT_DIR))
        write_text_clean(path, item["latex"])
        table_tex.append(item["latex"])
    write_text_clean(OUT_DIR / "all_table_snippets.tex", "\n".join(table_tex))

    data["asset_summary"] = {
        "prompt_tex_count": len(data["all_prompts"]),
        "prompt_output_tex_count": len(all_prompt_output_tex),
        "prompt_short_tex_count": len(all_prompt_short_tex),
        "prompt_png_count": png_count,
        "prompt_output_png_count": png_output_count,
        "prompt_short_png_count": png_short_count,
        "png_renderer": "Pillow" if png_count else "not available",
    }


def prompt_card(item: dict[str, Any], *, open_details: bool = True) -> str:
    detail_open = " open" if open_details else ""
    latex_id = f"latex-{item['id']}"
    output_latex_id = f"latex-output-{item['id']}"
    short_latex_id = f"latex-short-{item['id']}"
    copy_label = "Copy Full LaTeX" if item.get("latex_short") else "Copy LaTeX"
    display_prompt = item.get("short_prompt") or item["prompt"]
    display_note = ""
    if item.get("short_prompt"):
        display_note = """<div class="variant-note">Displayed preview uses the short paper variant. Full and short LaTeX/PNG assets are both linked above.</div>"""
    output_button = ""
    output_area = ""
    output_links = ""
    short_button = ""
    short_area = ""
    short_links = ""
    if item.get("latex_with_output"):
        output_button = f"""<button type="button" onclick="copySnippet('{output_latex_id}', event)">Copy LaTeX + Output</button>"""
        output_area = f"""<textarea id="{output_latex_id}" class="copy-source">{esc(item['latex_with_output'])}</textarea>"""
        output_links = f"""<a href="{esc(item['tex_output_path'])}">paired .tex</a><a href="{esc(item['png_output_path'])}">paired .png</a>"""
    if item.get("latex_short"):
        short_button = f"""<button type="button" onclick="copySnippet('{short_latex_id}', event)">Copy Short LaTeX</button>"""
        short_area = f"""<textarea id="{short_latex_id}" class="copy-source">{esc(item['latex_short'])}</textarea>"""
        short_links = f"""<a href="{esc(item['tex_short_path'])}">short .tex</a><a href="{esc(item['png_short_path'])}">short .png</a>"""
    return f"""
    <details class="prompt-card"{detail_open} id="{slug(item['name'])}">
      <summary><span>{esc(item['name'])}</span><code>{esc(Path(item['source']).name)}</code></summary>
      <div class="asset-actions">
        <button type="button" onclick="copySnippet('{latex_id}', event)">{copy_label}</button>
        {short_button}
        {output_button}
        <a href="{esc(item['tex_path'])}">.tex</a>
        <a href="{esc(item['png_path'])}">.png</a>
        {short_links}
        {output_links}
      </div>
      <textarea id="{latex_id}" class="copy-source">{esc(item['latex'])}</textarea>
      {short_area}
      {output_area}
      {display_note}
      <pre>{html_with_placeholders(display_prompt)}</pre>
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


def table_snippet_inline_control(snippets: list[dict[str, str]], snippet_id: str) -> str:
    item = next((snippet for snippet in snippets if snippet.get("id") == snippet_id), None)
    if not item:
        return ""
    latex_id = f"inline-table-latex-{item['id']}"
    return f"""
    <div class="asset-actions table-actions">
      <button type="button" onclick="copySnippet('{latex_id}', event)">Copy Table LaTeX</button>
      <a href="{esc(item.get('tex_path', ''))}">.tex</a>
    </div>
    <textarea id="{latex_id}" class="copy-source">{esc(item['latex'])}</textarea>
    """


def table_snippet_cards(snippets: list[dict[str, str]]) -> str:
    cards = []
    for item in snippets:
        latex_id = f"table-latex-{item['id']}"
        cards.append(
            f"""
            <details class="prompt-card table-snippet" open>
              <summary><span>{esc(item['name'])}</span><code>LaTeX table*</code></summary>
              <div class="asset-actions">
                <button type="button" onclick="copySnippet('{latex_id}', event)">Copy Table LaTeX</button>
                <a href="{esc(item.get('tex_path', ''))}">.tex</a>
              </div>
              <textarea id="{latex_id}" class="copy-source">{esc(item['latex'])}</textarea>
              <pre>{esc(item['latex'])}</pre>
            </details>
            """
        )
    return "\n".join(cards)


def render_html(data: dict[str, Any]) -> str:
    def comma_int(value: Any) -> str:
        return f"{int(value):,}"

    def short_float(value: Any) -> str:
        return f"{float(value):.2f}"

    def mean_median_range(summary: dict[str, Any]) -> str:
        return (
            f"{short_float(summary.get('mean', 0))} mean / "
            f"{float(summary.get('median', 0)):.0f} median "
            f"({summary.get('min', 0)}-{summary.get('max', 0)} range)"
        )

    private_rows = [
        [row["secret_type"], row["question"], row["answer"]]
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
    dataset_stats = data.get("dataset_stats") or {}
    private_stats = dataset_stats.get("private_qa") or {}
    final_stats = dataset_stats.get("final_privacy_hopqa") or {}
    full_stats = final_stats.get("Full") or {}
    dataset_summary_rows = []
    if private_stats and full_stats:
        dataset_summary_rows = [
            ["Private QA Set", "Local documents with private QA", comma_int(private_stats["local_documents"])],
            ["Private QA Set", "Private question-answer pairs", comma_int(private_stats["private_qa_pairs"])],
            ["Private QA Set", "Unique private-question strings", comma_int(private_stats["unique_private_questions"])],
            ["Private QA Set", "Private questions per local document", mean_median_range(private_stats["private_questions_per_doc"])],
            ["Private QA Set", "Secret types", comma_int(private_stats["n_secret_types"])],
            ["Private QA Set", "DR tasks covered", comma_int(private_stats["task_count"])],
            ["Final Privacy HopQA", "Multi-hop questions", comma_int(full_stats["multi_hop_questions"])],
            ["Final Privacy HopQA", "Hop-level questions", comma_int(full_stats["hop_questions"])],
            ["Final Privacy HopQA", "Private/local hop questions", comma_int(full_stats["private_local_questions"])],
            ["Final Privacy HopQA", "Public/web hop questions", comma_int(full_stats["public_web_questions"])],
            ["Final Privacy HopQA", "Unique local supporting documents", comma_int(full_stats["unique_local_documents"])],
            ["Final Privacy HopQA", "Unique web supporting documents", comma_int(full_stats["unique_web_documents"])],
            ["Final Privacy HopQA", "Private hops per local supporting document", mean_median_range(full_stats["private_questions_per_local_doc"])],
            ["Final Privacy HopQA", "Hops per multi-hop question", mean_median_range(full_stats["hops_per_question"])],
        ]
    dataset_split_rows = [
        [
            split,
            comma_int(stat["multi_hop_questions"]),
            comma_int(stat["hop_questions"]),
            comma_int(stat["private_local_questions"]),
            comma_int(stat["public_web_questions"]),
            comma_int(stat["unique_local_documents"]),
            short_float(stat["private_questions_per_local_doc"]["mean"]),
            short_float(stat["hops_per_question"]["mean"]),
        ]
        for split in ["Full", "Train", "Val", "Test"]
        if (stat := final_stats.get(split))
    ]
    secret_type_rows = [
        [secret_type, comma_int(count)]
        for secret_type, count in sorted((private_stats.get("secret_types") or {}).items())
    ]
    research_outputs = data["research_sample_outputs"]
    adversary_outputs = data["leakage_sample"]["outputs"]
    source_rows = [[name, path] for name, path in data["sources"].items()]

    dataset_prompt_cards = "\n".join(prompt_card(item, open_details=(idx < 4)) for idx, item in enumerate(data["dataset_prompts"]))
    research_prompt_cards = "\n".join(prompt_card(item) for item in data["research_prompts"])
    adversary_prompt_cards = "\n".join(prompt_card(item) for item in data["adversary_prompts"])
    table_snippet_items = data.get("table_snippets") or []
    table_snippets = table_snippet_cards(table_snippet_items)
    private_qa_table_controls = table_snippet_inline_control(table_snippet_items, "private_qa_examples")
    dataset_summary_table_controls = table_snippet_inline_control(table_snippet_items, "dataset_summary_compact")
    dataset_split_table_controls = table_snippet_inline_control(table_snippet_items, "dataset_split_summary")
    secret_type_table_controls = table_snippet_inline_control(table_snippet_items, "private_qa_secret_type_counts")
    asset_summary = data.get("asset_summary") or {}

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
    .asset-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid var(--rule);
      background: #fbfcfd;
    }}
    .asset-actions button,
    .asset-actions a {{
      appearance: none;
      border: 1px solid var(--rule);
      background: #ffffff;
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 9px;
      font: inherit;
      font-size: 12px;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
    }}
    .asset-actions button.copied {{
      border-color: var(--green);
      color: var(--green);
    }}
    .table-actions {{
      margin: 10px 0 0;
      border: 1px solid var(--rule);
      border-radius: 8px 8px 0 0;
    }}
    .variant-note {{
      padding: 8px 14px;
      color: var(--green);
      background: #f3faf6;
      border-bottom: 1px solid var(--rule);
      font-size: 12px;
      font-weight: 700;
    }}
    .copy-source {{
      position: absolute;
      left: -10000px;
      top: auto;
      width: 1px;
      height: 1px;
      opacity: 0;
    }}
    .placeholder {{
      color: var(--green);
      font-weight: 800;
    }}
    .asset-manifest {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
      margin: 14px 0 20px;
      max-width: 1100px;
    }}
    .asset-manifest div {{
      border: 1px solid var(--rule);
      background: #fff;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
    }}
    .asset-manifest code {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
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
  <p>Self-contained view of prompt templates, research-agent step prompts, adversary prompts, table snippets, and representative outputs used for the Privacy HopQA work. Dynamic inputs are shown as green placeholders so prompt screenshots do not bake in one concrete run.</p>
  <nav>
    <a href="#dataset">Private QA Generation</a>
    <a href="#dataset-summary">Dataset Summary</a>
    <a href="#research-agent">Research Agent Steps</a>
    <a href="#adversary">Adversary Prompts</a>
    <a href="#latex-assets">LaTeX and PNG Assets</a>
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
    {private_qa_table_controls}
    {table(["Type", "Question", "Answer"], private_rows)}
    <h3>Final Test Split Examples</h3>
    {table(["Task", "Chain", "Company", "Pattern", "Hops", "First Question", "Final Question", "Final Answer"], test_rows)}
    <h3 id="dataset-summary">Dataset Summary Tables</h3>
    {dataset_summary_table_controls}
    {table(["Dataset component", "Statistic", "Value"], dataset_summary_rows)}
    <h3>Split Summary</h3>
    {dataset_split_table_controls}
    {table(["Split", "Questions", "Hops", "Local", "Web", "Local docs", "Local q/doc", "Hops/q"], dataset_split_rows)}
    <h3>Private QA Secret Types</h3>
    {secret_type_table_controls}
    {table(["Secret type", "Private QA pairs"], secret_type_rows)}
    <div class="json-grid">
      {json_panel("Sample Private QA Extraction Output", data["private_qa_examples"][0] if data["private_qa_examples"] else {}, limit=2600)}
    </div>
  </section>

  <section id="research-agent">
    <div class="kicker">Rollout Harness</div>
    <h2>Research Agent Step Prompts</h2>
    <p>These are rendered examples of the PipelineRL Privacy HopQA prompt builders for the four LLM-controlled research-agent stages. The JSON panels show model-facing sample outputs; harness-only bookkeeping is omitted.</p>
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
    <div class="note">The prompt cards use placeholders for visible web queries and private questions. The sample outputs below come from a saved StepFun leakage-evaluation row (runner {esc(data["leakage_sample"].get("runner_version"))}, prompt {esc(data["leakage_sample"].get("prompt_version"))}): {esc(data["leakage_sample"].get("source"))}.</div>
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

  <section id="latex-assets">
    <div class="kicker">Paper Export</div>
    <h2>LaTeX Snippets and Overleaf Assets</h2>
    <p>Each prompt card has copy buttons for a standalone tcolorbox, an optional short version with example-heavy sections omitted, and an optional prompt-plus-sample-output version. PNG versions of prompt boxes are generated into upload-ready folders.</p>
    <div class="asset-manifest">
      <div><strong>Prompt tcolorboxes</strong><code>latex_prompt_snippets/</code>{esc(asset_summary.get("prompt_tex_count", 0))} files</div>
      <div><strong>Short prompt tcolorboxes</strong><code>latex_prompt_short_snippets/</code>{esc(asset_summary.get("prompt_short_tex_count", 0))} files</div>
      <div><strong>Prompt + output tcolorboxes</strong><code>latex_prompt_output_snippets/</code>{esc(asset_summary.get("prompt_output_tex_count", 0))} files</div>
      <div><strong>Prompt PNGs</strong><code>overleaf_prompt_pngs/</code>{esc(asset_summary.get("prompt_png_count", 0))} files</div>
      <div><strong>Short prompt PNGs</strong><code>overleaf_prompt_short_pngs/</code>{esc(asset_summary.get("prompt_short_png_count", 0))} files</div>
      <div><strong>Prompt + output PNGs</strong><code>overleaf_prompt_output_pngs/</code>{esc(asset_summary.get("prompt_output_png_count", 0))} files</div>
    </div>
    <h3>Table LaTeX Snippets</h3>
    <div class="grid">{table_snippets}</div>
  </section>

  <section id="sources" class="source-table">
    <div class="kicker">Traceability</div>
    <h2>Source Files and Artifacts</h2>
    <p>These paths are included so paper screenshots can be traced back to source code or saved experiment output.</p>
    {table(["Source", "Path"], source_rows)}
  </section>
</main>
<script>
async function copySnippet(id, event) {{
  const source = document.getElementById(id);
  if (!source) return;
  try {{
    if (navigator.clipboard && window.isSecureContext) {{
      await navigator.clipboard.writeText(source.value);
    }} else {{
      throw new Error("clipboard fallback");
    }}
  }} catch (err) {{
    source.style.position = "fixed";
    source.style.left = "0";
    source.style.top = "0";
    source.style.width = "1px";
    source.style.height = "1px";
    source.focus();
    source.select();
    document.execCommand("copy");
    source.style.position = "absolute";
    source.style.left = "-10000px";
  }}
  if (event && event.currentTarget) {{
    const button = event.currentTarget;
    const original = button.textContent;
    button.textContent = "Copied";
    button.classList.add("copied");
    window.setTimeout(() => {{
      button.textContent = original;
      button.classList.remove("copied");
    }}, 1200);
  }}
}}
</script>
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
            "dataset_summary_tables": str(DATASET_TABLE_DIR),
            "research_sample_trace": str(RESEARCH_SAMPLE_TRACE),
            "leakage_sample": str(leakage_sample.get("source")),
        },
        "dataset_prompts": build_dataset_prompts(),
        "research_prompts": build_research_prompts(),
        "adversary_prompts": build_adversary_prompts(leakage_sample),
        "private_qa_examples": load_private_qa_examples(),
        "test_set_examples": load_test_set_examples(),
        "dataset_stats": load_dataset_stats(),
        "dataset_table_snippets": load_dataset_table_snippets(),
        "research_sample_outputs": load_research_sample_outputs(),
        "leakage_sample": leakage_sample,
    }
    return data


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    enrich_prompts(data)
    write_generated_assets(data)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_clean(OUT_HTML, render_html(data))
    OUT_README.write_text(
        "\n".join(
            [
                "# Privacy HopQA Prompt Appendix",
                "",
                "Generated paper-facing prompt appendix for Privacy HopQA.",
                "",
                "- `index.html`: screenshot-friendly single-page view.",
                "- `prompt_appendix_data.json`: extracted prompts, examples, and sample outputs.",
                "- `latex_prompt_snippets/`: one copyable `tcolorbox` prompt per file.",
                "- `latex_prompt_short_snippets/`: short variants with example-heavy sections omitted where available.",
                "- `latex_prompt_output_snippets/`: prompt boxes paired with sample outputs where available.",
                "- `latex_table_snippets/`: paper-ready `table*` snippets for examples, dataset summaries, and leakage cases.",
                "- `overleaf_prompt_pngs/`: PNG prompt boxes ready for Overleaf upload.",
                "- `overleaf_prompt_short_pngs/`: PNG short prompt boxes ready for Overleaf upload.",
                "- `overleaf_prompt_output_pngs/`: PNG prompt-plus-output boxes ready for Overleaf upload.",
                "",
                "LaTeX snippets expect these paper preamble pieces:",
                "",
                "```latex",
                "\\usepackage{xcolor}",
                "\\usepackage{tcolorbox}",
                "\\tcbuselibrary{breakable,skins}",
                "\\usepackage{booktabs}",
                "\\usepackage{enumitem}",
                "\\definecolor{my-green}{HTML}{166B54}",
                "```",
                "",
                "Equivalent shortcut: `\\usepackage[most]{tcolorbox}` loads the required tcolorbox libraries too. The generated boxes include `enhanced`, `breakable`, and split-page skin options so long prompts can continue across page breaks.",
                "",
                "Regenerate from the MosaicProject root with:",
                "",
                "```bash",
                "/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 scripts/build_privacy_hopqa_prompt_appendix.py",
                "```",
                "",
                "The PNG export uses Pillow when available. The generator reads source prompt files in this repository plus the PipelineRL Privacy HopQA prompt builders and saved runtime/leakage artifacts listed at the bottom of `index.html`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUT_HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
