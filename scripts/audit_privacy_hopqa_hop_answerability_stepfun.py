#!/usr/bin/env python3
"""Audit PrivacyHopQA hop questions by asking StepFun to answer from the gold doc.

This is intentionally dependency-free: it uses the OpenAI-compatible HTTP API via
the Python standard library so it can run from the base host environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import json
import os
import re
import string
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


DEFAULT_DATASET = (
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/"
    "privacy_hopqa_finalized_2026-05-09_contextsafe_answer_variants.jsonl"
)
DEFAULT_OUTPUT = (
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/"
    "stepfun_full_doc_hop_answerability_2026-05-09.jsonl"
)
DEFAULT_REVIEW_CSV = (
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/"
    "stepfun_full_doc_hop_answerability_review_2026-05-09.csv"
)
DEFAULT_VIEWER = (
    "/home/toolkit/my_code/MosaicProject/runtime_analysis/"
    "privacy_hopqa_stepfun_answerability_review_2026-05-09.html"
)
DEFAULT_SUMMARY_JSON = (
    "/home/toolkit/my_code/MosaicProject/runtime_analysis/"
    "privacy_hopqa_stepfun_answerability_review_2026-05-09_summary.json"
)
DEFAULT_BASE_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000"
ALLOWED_LABELS = {"yes", "no", "ambiguous", "underspecified", "requires_calculation"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            audit_id = row.get("audit_id")
            if audit_id:
                completed.add(str(audit_id))
    return completed


def doc_path_from_hop(hop: dict[str, Any]) -> Path | None:
    doc_ref = hop.get("doc_ref")
    if doc_ref:
        return Path(str(doc_ref))
    doc_id = str(hop.get("doc_id") or "")
    if doc_id.startswith("web/"):
        return Path("/home/toolkit/full_docs") / f"{doc_id.split('/', 1)[1]}.txt"
    return None


def flatten_hops(dataset_rows: list[dict[str, Any]], chain_ids: set[str] | None) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for chain_index, chain in enumerate(dataset_rows):
        chain_id = str(chain.get("chain_id") or chain.get("id") or chain_index)
        if chain_ids and chain_id not in chain_ids:
            continue
        hops = chain.get("hops") or []
        for hop_index, hop in enumerate(hops):
            hop_number = hop.get("hop_number") or hop_index + 1
            audit_id = f"{chain_id}::hop{hop_number}"
            flattened.append(
                {
                    "audit_id": audit_id,
                    "chain_index": chain_index,
                    "chain_id": chain_id,
                    "company": chain.get("company"),
                    "task": chain.get("task") or chain.get("task_id"),
                    "hop_index": hop_index,
                    "hop_number": hop_number,
                    "hop_type": hop.get("hop_type"),
                    "question": hop.get("question"),
                    "gold_answer": hop.get("answer"),
                    "accepted_answers": accepted_answers(hop),
                    "doc_id": hop.get("doc_id"),
                    "doc_ref": str(doc_path_from_hop(hop) or ""),
                    "supporting_span": hop.get("supporting_span"),
                    "question_issues": hop.get("question_issues") or [],
                    "existing_suggested_rewrite": hop.get("suggested_rewrite"),
                }
            )
    return flattened


def accepted_answers(hop: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for key in ("answer", "accepted_answer_variants", "alternate_valid_answers"):
        value = hop.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            answers.extend(str(v) for v in value if v is not None)
        else:
            answers.append(str(value))
    deduped: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        normalized = normalize_for_exact(answer)
        if normalized and normalized not in seen:
            deduped.append(answer)
            seen.add(normalized)
    return deduped


def read_document(path_text: str, max_chars: int) -> tuple[str, int, bool, str | None]:
    if not path_text:
        return "", 0, False, "missing_doc_ref"
    path = Path(path_text)
    if not path.exists():
        return "", 0, False, "doc_path_not_found"
    try:
        text = extract_document_text(path)
    except OSError as exc:
        return "", 0, False, f"doc_read_error: {exc}"
    except (ElementTree.ParseError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        return "", 0, False, f"doc_parse_error: {exc}"
    original_chars = len(text)
    truncated = original_chars > max_chars
    if truncated:
        text = text[:max_chars]
    return text, original_chars, truncated, None


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".pptx":
        return extract_pptx_text(path)
    if suffix == ".xlsx":
        return extract_xlsx_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    parts: list[str] = []
    for elem in root.iter():
        tag = strip_xml_ns(elem.tag)
        if tag == "t" and elem.text:
            parts.append(elem.text)
        elif tag in {"tab", "br"}:
            parts.append("\n" if tag == "br" else "\t")
        elif tag == "p":
            parts.append("\n")
    return normalize_extracted_text("".join(parts))


def extract_pptx_text(path: Path) -> str:
    sections: list[str] = []
    with zipfile.ZipFile(path) as zf:
        slide_names = sorted(
            name for name in zf.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        for slide_number, name in enumerate(slide_names, 1):
            root = ElementTree.fromstring(zf.read(name))
            texts = [elem.text for elem in root.iter() if strip_xml_ns(elem.tag) == "t" and elem.text]
            if texts:
                sections.append(f"Slide {slide_number}:\n" + "\n".join(texts))
    return normalize_extracted_text("\n\n".join(sections))


def extract_xlsx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        shared_strings = load_xlsx_shared_strings(zf, names)
        sheet_names = load_xlsx_sheet_names(zf, names)
        worksheet_files = sorted(
            name for name in names
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        sections: list[str] = []
        for idx, worksheet_file in enumerate(worksheet_files, 1):
            sheet_name = sheet_names.get(worksheet_file, f"sheet{idx}")
            root = ElementTree.fromstring(zf.read(worksheet_file))
            rows: list[str] = []
            for row in root.iter():
                if strip_xml_ns(row.tag) != "row":
                    continue
                values: list[str] = []
                for cell in row:
                    if strip_xml_ns(cell.tag) != "c":
                        continue
                    value = xlsx_cell_value(cell, shared_strings)
                    if value:
                        values.append(value)
                if values:
                    rows.append("\t".join(values))
            if rows:
                sections.append(f"Sheet: {sheet_name}\n" + "\n".join(rows))
    return normalize_extracted_text("\n\n".join(sections))


def load_xlsx_shared_strings(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root:
        if strip_xml_ns(si.tag) != "si":
            continue
        parts = [elem.text or "" for elem in si.iter() if strip_xml_ns(elem.tag) == "t"]
        strings.append("".join(parts))
    return strings


def load_xlsx_sheet_names(zf: zipfile.ZipFile, names: set[str]) -> dict[str, str]:
    if "xl/workbook.xml" not in names or "xl/_rels/workbook.xml.rels" not in names:
        return {}
    rel_root = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets: dict[str, str] = {}
    for rel in rel_root:
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            rel_targets[rel_id] = "xl/" + target.lstrip("/")
    workbook_root = ElementTree.fromstring(zf.read("xl/workbook.xml"))
    sheet_names: dict[str, str] = {}
    for elem in workbook_root.iter():
        if strip_xml_ns(elem.tag) != "sheet":
            continue
        name = elem.attrib.get("name")
        rel_id = elem.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_targets.get(rel_id or "")
        if name and target:
            sheet_names[target] = name
    return sheet_names


def xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        parts = [elem.text or "" for elem in cell.iter() if strip_xml_ns(elem.tag) == "t"]
        return "".join(parts).strip()
    value_node = None
    for child in cell:
        if strip_xml_ns(child.tag) == "v":
            value_node = child
            break
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text.strip()
    if cell_type == "s":
        index = int(raw)
        return shared_strings[index].strip() if 0 <= index < len(shared_strings) else raw
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def strip_xml_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalize_extracted_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_exact(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^)]*source[^)]*\)", " ", text)
    text = text.strip().strip(string.punctuation + " \t\n\r")
    text = text.translate(str.maketrans("", "", string.punctuation.replace("%", "")))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numbers(value: Any) -> list[str]:
    text = str(value or "").replace(",", "")
    return re.findall(r"[-+]?\d+(?:\.\d+)?%?", text)


def compare_answer(predicted: Any, accepted: list[str]) -> dict[str, Any]:
    predicted_norm = normalize_for_exact(predicted)
    accepted_norms = [normalize_for_exact(a) for a in accepted]
    exact_match = bool(predicted_norm) and predicted_norm in accepted_norms
    predicted_numbers = extract_numbers(predicted)
    accepted_numbers = []
    for answer in accepted:
        accepted_numbers.extend(extract_numbers(answer))
    numeric_match = bool(predicted_numbers) and bool(set(predicted_numbers) & set(accepted_numbers))
    return {
        "predicted_normalized": predicted_norm,
        "accepted_normalized": accepted_norms,
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "matched_accepted_answer": accepted[accepted_norms.index(predicted_norm)] if exact_match else None,
        "predicted_numbers": predicted_numbers,
        "accepted_numbers": accepted_numbers,
    }


def build_prompt(item: dict[str, Any], document_text: str, truncated: bool, original_chars: int) -> list[dict[str, str]]:
    truncation_note = ""
    if truncated:
        truncation_note = (
            f"\nNOTE: The document was truncated to fit the model context. "
            f"Original characters: {original_chars}."
        )
    system = (
        "You are auditing whether one hop question can be answered from one provided document. "
        "Use only the provided document. Return exactly one JSON object and no markdown."
    )
    user = f"""Document metadata:
chain_id: {item['chain_id']}
hop_number: {item['hop_number']}
hop_type: {item.get('hop_type')}
doc_id: {item.get('doc_id')}
doc_ref: {item.get('doc_ref')}{truncation_note}

Full document:
<document>
{document_text}
</document>

Hop question:
{item.get('question')}

Instructions:
- Answer the hop question using only the document.
- Think internally, but do not include chain-of-thought.
- predicted_answer must be the shortest exact answer possible, normally fewer than 5 words.
- Do not include citations, explanations, or units unless the unit is required for the answer.
- If the answer is not present, set predicted_answer to "NOT_FOUND".
- If the question is ambiguous or underspecified, still give the best answer if possible, but set answerable_label accordingly.
- If calculation is required and the document contains enough information, compute the answer and set answerable_label to "requires_calculation".
- evidence_quote should be a short exact quote from the document supporting the answer, or an empty string if not answerable.
- Keep evidence_quote under 12 words and under 120 characters. For tables, quote only the minimal row fragment.
- If no short quote is available, set evidence_quote to "" rather than writing a long quote.
- All string values must be valid JSON strings. Escape internal double quotes, tabs, and newlines.
- suggested_rewrite should be empty unless the question should be lightly rewritten to be answerable or less ambiguous.

Return JSON with exactly these keys:
{{
  "predicted_answer": "string",
  "answerable_label": "yes|no|ambiguous|underspecified|requires_calculation",
  "confidence": 0.0,
  "evidence_quote": "string",
  "reason": "brief reason, no chain-of-thought",
  "suggested_rewrite": "string",
  "suggested_answer": "string"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def post_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:2000]}") from exc


def post_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>"],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:2000]}") from exc


def call_stepfun(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    if args.api_mode in {"completion_template", "completion_prefill_no_think"}:
        assistant_prefix = "<think>\n</think>\n" if args.api_mode == "completion_prefill_no_think" else ""
        manual_prompt = (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}"
        )
        response = post_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=manual_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        return {
            "response": response,
            "content": response.get("choices", [{}])[0].get("text") or "",
            "reasoning": "",
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            "usage": response.get("usage"),
        }

    response = post_chat_completion(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    message = response.get("choices", [{}])[0].get("message") or {}
    return {
        "response": response,
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning") or "",
        "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        "usage": response.get("usage"),
    }


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned, strict=False)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json_root_not_object"
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            salvaged = salvage_stepfun_fields(cleaned)
            if salvaged:
                salvaged["_parse_warning"] = "json_salvaged: no_complete_json_object"
                return salvaged, None
            return None, "no_json_object_found"
        try:
            parsed = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError as exc:
            salvaged = salvage_stepfun_fields(cleaned)
            if salvaged:
                salvaged["_parse_warning"] = f"json_salvaged: {exc}"
                return salvaged, None
            return None, f"json_decode_error: {exc}"
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json_root_not_object"


def salvage_stepfun_fields(text: str) -> dict[str, Any] | None:
    string_keys = [
        "predicted_answer",
        "answerable_label",
        "evidence_quote",
        "reason",
        "suggested_rewrite",
        "suggested_answer",
    ]
    salvaged: dict[str, Any] = {}
    for key in string_keys:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if match:
            salvaged[key] = bytes(match.group(1), "utf-8").decode("unicode_escape", errors="replace")
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if confidence_match:
        salvaged["confidence"] = float(confidence_match.group(1))
    if salvaged.get("predicted_answer") and salvaged.get("answerable_label"):
        salvaged.setdefault("confidence", 0.0)
        salvaged.setdefault("evidence_quote", "")
        salvaged.setdefault("reason", "Recovered answer fields from malformed JSON.")
        salvaged.setdefault("suggested_rewrite", "")
        salvaged.setdefault("suggested_answer", "")
        return salvaged
    return None


def audit_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    document_text, original_chars, truncated, doc_error = read_document(item["doc_ref"], args.max_doc_chars)
    base = dict(item)
    base.update(
        {
            "document_chars": original_chars,
            "document_truncated": truncated,
            "stepfun_base_url": args.base_url,
            "stepfun_model": args.model,
            "prompt_template": "full_doc_hop_answerability_v1",
            "created_at_unix": time.time(),
        }
    )
    if doc_error:
        base.update(
            {
                "error": doc_error,
                "stepfun": {},
                "comparison": compare_answer("", base["accepted_answers"]),
                "flags": ["doc_error", doc_error],
            }
        )
        return base

    messages = build_prompt(item, document_text, truncated, original_chars)
    last_error: str | None = None
    raw_text = ""
    raw_reasoning = ""
    raw_response: dict[str, Any] | None = None
    for attempt in range(1, args.attempts + 1):
        try:
            result = call_stepfun(args, messages)
            raw_response = result.get("response")
            raw_text = str(result.get("content") or "")
            raw_reasoning = str(result.get("reasoning") or "")
            parsed, parse_error = parse_json_object(raw_text)
            if parsed is None and raw_reasoning:
                parsed, parse_error = parse_json_object(raw_reasoning)
            if parsed is None:
                last_error = parse_error
                raise RuntimeError(parse_error or "parse_error")
            parse_warning = parsed.pop("_parse_warning", None)
            stepfun = normalize_stepfun_payload(parsed)
            comparison = compare_answer(stepfun.get("predicted_answer"), base["accepted_answers"])
            flags = flags_for_result(stepfun, comparison, truncated)
            if parse_warning:
                flags.append(str(parse_warning).split(":", 1)[0])
            base.update(
                {
                    "stepfun": stepfun,
                    "comparison": comparison,
                    "raw_stepfun_text": raw_text,
                    "raw_stepfun_reasoning": raw_reasoning,
                    "finish_reason": result.get("finish_reason"),
                    "usage": result.get("usage"),
                    "flags": flags,
                }
            )
            return base
        except Exception as exc:  # Keep per-row failures explicit in the output.
            last_error = str(exc)
            if attempt < args.attempts:
                time.sleep(min(8.0, 1.5 * attempt))

    stepfun = {
        "predicted_answer": "",
        "answerable_label": "no",
        "confidence": 0.0,
        "evidence_quote": "",
        "reason": last_error or "unknown_error",
        "suggested_rewrite": "",
        "suggested_answer": "",
    }
    base.update(
        {
            "error": last_error,
            "stepfun": stepfun,
            "comparison": compare_answer("", base["accepted_answers"]),
            "raw_stepfun_text": raw_text,
            "raw_stepfun_reasoning": raw_reasoning,
            "raw_response": raw_response,
            "flags": ["stepfun_error"],
        }
    )
    return base


def normalize_stepfun_payload(payload: dict[str, Any]) -> dict[str, Any]:
    label = str(payload.get("answerable_label") or "no").strip().lower()
    if label not in ALLOWED_LABELS:
        label = "no"
    confidence = payload.get("confidence", 0.0)
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        confidence_float = 0.0
    return {
        "predicted_answer": str(payload.get("predicted_answer") or "").strip(),
        "answerable_label": label,
        "confidence": max(0.0, min(1.0, confidence_float)),
        "evidence_quote": str(payload.get("evidence_quote") or "").strip(),
        "reason": str(payload.get("reason") or "").strip(),
        "suggested_rewrite": str(payload.get("suggested_rewrite") or "").strip(),
        "suggested_answer": str(payload.get("suggested_answer") or "").strip(),
    }


def flags_for_result(stepfun: dict[str, Any], comparison: dict[str, Any], truncated: bool) -> list[str]:
    flags: list[str] = []
    label = stepfun.get("answerable_label")
    predicted = normalize_for_exact(stepfun.get("predicted_answer"))
    if truncated:
        flags.append("doc_truncated")
    if label != "yes":
        flags.append(f"label_{label}")
    if predicted in {"notfound", "not_found", "not found"} or not predicted:
        flags.append("stepfun_not_found")
    if not comparison.get("exact_match"):
        flags.append("exact_mismatch")
    if comparison.get("numeric_match") and not comparison.get("exact_match"):
        flags.append("numeric_match_only")
    if stepfun.get("suggested_rewrite"):
        flags.append("suggested_rewrite")
    return flags


def write_review_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "audit_id",
        "chain_id",
        "hop_number",
        "hop_type",
        "company",
        "doc_id",
        "question",
        "gold_answer",
        "accepted_answers",
        "stepfun_answer",
        "answerable_label",
        "exact_match",
        "numeric_match",
        "flags",
        "evidence_quote",
        "reason",
        "suggested_rewrite",
        "doc_ref",
    ]
    suspicious = sorted(rows, key=review_sort_key)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in suspicious:
            stepfun = row.get("stepfun") or {}
            comparison = row.get("comparison") or {}
            writer.writerow(
                {
                    "audit_id": row.get("audit_id"),
                    "chain_id": row.get("chain_id"),
                    "hop_number": row.get("hop_number"),
                    "hop_type": row.get("hop_type"),
                    "company": row.get("company"),
                    "doc_id": row.get("doc_id"),
                    "question": row.get("question"),
                    "gold_answer": row.get("gold_answer"),
                    "accepted_answers": " | ".join(row.get("accepted_answers") or []),
                    "stepfun_answer": stepfun.get("predicted_answer"),
                    "answerable_label": stepfun.get("answerable_label"),
                    "exact_match": comparison.get("exact_match"),
                    "numeric_match": comparison.get("numeric_match"),
                    "flags": " | ".join(row.get("flags") or []),
                    "evidence_quote": stepfun.get("evidence_quote"),
                    "reason": stepfun.get("reason"),
                    "suggested_rewrite": stepfun.get("suggested_rewrite"),
                    "doc_ref": row.get("doc_ref"),
                }
            )


def review_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    flags = set(row.get("flags") or [])
    if "stepfun_error" in flags or "doc_error" in flags:
        rank = 0
    elif "label_underspecified" in flags or "label_ambiguous" in flags:
        rank = 1
    elif "stepfun_not_found" in flags:
        rank = 2
    elif "exact_mismatch" in flags and "numeric_match_only" not in flags:
        rank = 3
    elif "numeric_match_only" in flags:
        rank = 4
    else:
        rank = 5
    return rank, str(row.get("audit_id"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, int] = {}
    by_flag: dict[str, int] = {}
    by_company: dict[str, dict[str, int]] = {}
    by_hop_type: dict[str, dict[str, int]] = {}
    exact_matches = 0
    numeric_only = 0
    for row in rows:
        stepfun = row.get("stepfun") or {}
        comparison = row.get("comparison") or {}
        label = str(stepfun.get("answerable_label") or "missing")
        by_label[label] = by_label.get(label, 0) + 1
        if comparison.get("exact_match"):
            exact_matches += 1
        if comparison.get("numeric_match") and not comparison.get("exact_match"):
            numeric_only += 1
        for flag in row.get("flags") or []:
            by_flag[flag] = by_flag.get(flag, 0) + 1
        add_group_stat(by_company, str(row.get("company") or "unknown"), comparison.get("exact_match"))
        add_group_stat(by_hop_type, str(row.get("hop_type") or "unknown"), comparison.get("exact_match"))
    total = len(rows)
    return {
        "total_hops": total,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "numeric_match_only": numeric_only,
        "answerable_label_counts": by_label,
        "flag_counts": dict(sorted(by_flag.items(), key=lambda kv: (-kv[1], kv[0]))),
        "company_stats": by_company,
        "hop_type_stats": by_hop_type,
    }


def add_group_stat(groups: dict[str, dict[str, int]], key: str, exact_match: Any) -> None:
    group = groups.setdefault(key, {"total": 0, "exact_matches": 0})
    group["total"] += 1
    if exact_match:
        group["exact_matches"] += 1


def write_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_viewer(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    viewer_rows = [viewer_row(row) for row in sorted(rows, key=review_sort_key)]
    data_json = json.dumps({"summary": summary, "rows": viewer_rows}, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")
    html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PrivacyHopQA StepFun Hop Answerability Audit</title>
  <style>
    :root {{
      --ink: #172019;
      --muted: #5d6a61;
      --bg: #f4f0e6;
      --panel: #fffaf0;
      --line: #d8ceb9;
      --accent: #2d6b57;
      --bad: #a33b2f;
      --warn: #a86f13;
      --good: #2e7252;
      --soft: #e8dfcc;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(45,107,87,.18), transparent 32rem),
        linear-gradient(135deg, #f7f2e7, #eee5d4);
      font-family: Georgia, "Times New Roman", serif;
    }}
    header {{
      padding: 28px 34px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,250,240,.78);
      position: sticky;
      top: 0;
      z-index: 2;
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -.02em; }}
    .sub {{ color: var(--muted); max-width: 1100px; line-height: 1.35; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      padding: 22px 34px 6px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 10px 26px rgba(49,36,16,.08);
    }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .label {{ color: var(--muted); font-size: 13px; margin-top: 3px; }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr repeat(4, minmax(120px, 170px));
      gap: 10px;
      padding: 18px 34px;
    }}
    input, select {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 10px;
      padding: 10px 11px;
      font: 14px Georgia, "Times New Roman", serif;
    }}
    main {{ padding: 0 34px 34px; }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .plot {{
      background: rgba(255,250,240,.88);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }}
    .barrow {{
      display: grid;
      grid-template-columns: 160px 1fr 70px;
      gap: 10px;
      align-items: center;
      margin: 7px 0;
      font-size: 13px;
    }}
    .bar {{ height: 14px; background: var(--soft); border-radius: 20px; overflow: hidden; }}
    .fill {{ height: 100%; background: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0 8px;
      font-size: 13px;
    }}
    th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
      padding: 0 10px 4px;
      position: sticky;
      top: 113px;
      background: #f4f0e6;
      z-index: 1;
    }}
    td {{
      vertical-align: top;
      background: rgba(255,250,240,.92);
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 11px 10px;
      line-height: 1.32;
    }}
    td:first-child {{ border-left: 1px solid var(--line); border-radius: 12px 0 0 12px; }}
    td:last-child {{ border-right: 1px solid var(--line); border-radius: 0 12px 12px 0; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 7px;
      margin: 2px 2px 2px 0;
      background: var(--soft);
      color: var(--ink);
      white-space: nowrap;
    }}
    .bad {{ background: #f4d7d1; color: var(--bad); }}
    .warn {{ background: #f4e1bd; color: #7a4d05; }}
    .good {{ background: #d8eadf; color: var(--good); }}
    details {{ margin-top: 7px; }}
    summary {{ cursor: pointer; color: var(--accent); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .small {{ color: var(--muted); font-size: 12px; }}
    @media (max-width: 900px) {{
      .controls {{ grid-template-columns: 1fr; }}
      th {{ position: static; }}
      header {{ position: static; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>PrivacyHopQA StepFun Hop Answerability Audit</h1>
  <div class="sub">
    StepFun sees only the full gold document and the hop question. It does not see the gold answer or accepted variants.
    The audit then checks whether StepFun's short answer exactly matches an accepted answer, and highlights cases that look
    underspecified, ambiguous, unanswerable, numeric-only matched, or document-truncated.
  </div>
</header>
<section class="cards" id="cards"></section>
<section class="controls">
  <input id="search" placeholder="Search question, answer, doc id, reason...">
  <select id="flagFilter"><option value="">All flags</option></select>
  <select id="labelFilter"><option value="">All labels</option></select>
  <select id="companyFilter"><option value="">All companies</option></select>
  <select id="matchFilter">
    <option value="">All match states</option>
    <option value="exact">Exact match</option>
    <option value="numeric">Numeric-only match</option>
    <option value="mismatch">Exact mismatch</option>
  </select>
</section>
<main>
  <section class="plots">
    <div class="plot"><h3>Flags</h3><div id="flagPlot"></div></div>
    <div class="plot"><h3>Answerability labels</h3><div id="labelPlot"></div></div>
    <div class="plot"><h3>Exact match by hop type</h3><div id="hopPlot"></div></div>
  </section>
  <div class="small" id="visibleCount"></div>
  <table>
    <thead>
      <tr>
        <th>Hop</th>
        <th>Question</th>
        <th>Gold / StepFun</th>
        <th>Flags</th>
        <th>Evidence and notes</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
</main>
<script id="audit-data" type="application/json">__AUDIT_DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('audit-data').textContent);
const rows = DATA.rows;
const summary = DATA.summary;

function pct(x) {{ return (100 * x).toFixed(1) + '%'; }}
function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function card(label, value) {{ return `<div class="card"><div class="metric">${{esc(value)}}</div><div class="label">${{esc(label)}}</div></div>`; }}
function renderCards() {{
  const total = summary.total_hops || 0;
  document.getElementById('cards').innerHTML = [
    card('Total hop questions', total),
    card('Exact StepFun matches', `${{summary.exact_matches}} (${{pct(summary.exact_match_rate || 0)}})`),
    card('Numeric-only matches', summary.numeric_match_only || 0),
    card('Exact mismatches', total - (summary.exact_matches || 0)),
  ].join('');
}}
function optionize(id, values) {{
  const select = document.getElementById(id);
  values.forEach(v => {{
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  }});
}}
function uniq(values) {{ return [...new Set(values.filter(Boolean))].sort(); }}
function renderBarPlot(id, entries, maxValue) {{
  const max = maxValue || Math.max(1, ...entries.map(e => e[1]));
  document.getElementById(id).innerHTML = entries.map(([k, v]) => `
    <div class="barrow">
      <div title="${{esc(k)}}">${{esc(k)}}</div>
      <div class="bar"><div class="fill" style="width:${{Math.max(2, 100*v/max)}}%"></div></div>
      <div class="mono">${{v}}</div>
    </div>`).join('');
}}
function renderPlots() {{
  renderBarPlot('flagPlot', Object.entries(summary.flag_counts || {}).slice(0, 16));
  renderBarPlot('labelPlot', Object.entries(summary.answerable_label_counts || {}));
  const hopEntries = Object.entries(summary.hop_type_stats || {}).map(([k, v]) => [k + ' exact', Math.round(1000 * v.exact_matches / Math.max(1, v.total)) / 10]);
  renderBarPlot('hopPlot', hopEntries, 100);
}}
function rowMatches(row) {{
  const q = document.getElementById('search').value.trim().toLowerCase();
  const flag = document.getElementById('flagFilter').value;
  const label = document.getElementById('labelFilter').value;
  const company = document.getElementById('companyFilter').value;
  const match = document.getElementById('matchFilter').value;
  const hay = [row.audit_id, row.company, row.doc_id, row.question, row.gold_answer, row.accepted_answers.join(' '), row.stepfun_answer, row.reason, row.evidence_quote, row.suggested_rewrite].join(' ').toLowerCase();
  if (q && !hay.includes(q)) return false;
  if (flag && !row.flags.includes(flag)) return false;
  if (label && row.answerable_label !== label) return false;
  if (company && row.company !== company) return false;
  if (match === 'exact' && !row.exact_match) return false;
  if (match === 'numeric' && !(row.numeric_match && !row.exact_match)) return false;
  if (match === 'mismatch' && row.exact_match) return false;
  return true;
}}
function flagClass(flag) {{
  if (flag.includes('mismatch') || flag.includes('not_found') || flag.includes('error')) return 'bad';
  if (flag.includes('numeric') || flag.includes('ambiguous') || flag.includes('underspecified') || flag.includes('truncated') || flag.includes('rewrite')) return 'warn';
  return 'good';
}}
function renderRows() {{
  const visible = rows.filter(rowMatches);
  document.getElementById('visibleCount').textContent = `Showing ${{visible.length}} of ${{rows.length}} hops`;
  document.getElementById('rows').innerHTML = visible.slice(0, 1000).map(row => {{
    const flags = row.flags.map(f => `<span class="pill ${{flagClass(f)}}">${{esc(f)}}</span>`).join('');
    const matchPill = row.exact_match ? '<span class="pill good">exact match</span>' :
      (row.numeric_match ? '<span class="pill warn">numeric-only match</span>' : '<span class="pill bad">mismatch</span>');
    return `<tr>
      <td>
        <div><b>${{esc(row.chain_id)}} hop ${{esc(row.hop_number)}}</b></div>
        <div class="small">${{esc(row.hop_type)}} · ${{esc(row.company || '')}}</div>
        <div class="small mono">${{esc(row.doc_id || '')}}</div>
      </td>
      <td>
        <div>${{esc(row.question)}}</div>
        ${{row.existing_suggested_rewrite ? `<details><summary>Existing rewrite</summary>${{esc(row.existing_suggested_rewrite)}}</details>` : ''}}
      </td>
      <td>
        <div><b>Gold:</b> ${{esc(row.gold_answer)}}</div>
        <div><b>StepFun:</b> ${{esc(row.stepfun_answer)}}</div>
        <div>${{matchPill}} <span class="pill">${{esc(row.answerable_label)}}</span></div>
        <details><summary>Accepted variants (${{row.accepted_answers.length}})</summary><div class="mono">${{esc(row.accepted_answers.join(' | '))}}</div></details>
      </td>
      <td>${{flags}}</td>
      <td>
        <div><b>Evidence:</b> ${{esc(row.evidence_quote)}}</div>
        <div><b>Reason:</b> ${{esc(row.reason)}}</div>
        ${{row.suggested_rewrite ? `<div><b>Suggested rewrite:</b> ${{esc(row.suggested_rewrite)}}</div>` : ''}}
        <details><summary>Doc path</summary><code>${{esc(row.doc_ref)}}</code></details>
      </td>
    </tr>`;
  }).join('');
}}
function init() {{
  renderCards();
  renderPlots();
  optionize('flagFilter', uniq(rows.flatMap(r => r.flags)));
  optionize('labelFilter', uniq(rows.map(r => r.answerable_label)));
  optionize('companyFilter', uniq(rows.map(r => r.company)));
  ['search','flagFilter','labelFilter','companyFilter','matchFilter'].forEach(id => document.getElementById(id).addEventListener('input', renderRows));
  renderRows();
}}
init();
</script>
</body>
</html>
"""
    html_text = html_template.replace("{{", "{").replace("}}", "}").replace("__AUDIT_DATA_JSON__", data_json)
    path.write_text(html_text, encoding="utf-8")


def viewer_row(row: dict[str, Any]) -> dict[str, Any]:
    stepfun = row.get("stepfun") or {}
    comparison = row.get("comparison") or {}
    return {
        "audit_id": row.get("audit_id"),
        "chain_id": row.get("chain_id"),
        "hop_number": row.get("hop_number"),
        "hop_type": row.get("hop_type"),
        "company": row.get("company"),
        "doc_id": row.get("doc_id"),
        "doc_ref": row.get("doc_ref"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "accepted_answers": row.get("accepted_answers") or [],
        "stepfun_answer": stepfun.get("predicted_answer"),
        "answerable_label": stepfun.get("answerable_label"),
        "confidence": stepfun.get("confidence"),
        "evidence_quote": stepfun.get("evidence_quote"),
        "reason": stepfun.get("reason"),
        "suggested_rewrite": stepfun.get("suggested_rewrite"),
        "suggested_answer": stepfun.get("suggested_answer"),
        "exact_match": bool(comparison.get("exact_match")),
        "numeric_match": bool(comparison.get("numeric_match")),
        "flags": row.get("flags") or [],
        "existing_suggested_rewrite": row.get("existing_suggested_rewrite"),
        "question_issues": row.get("question_issues") or [],
        "document_truncated": row.get("document_truncated"),
        "document_chars": row.get("document_chars"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--viewer", default=DEFAULT_VIEWER)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--base-url", default=os.environ.get("STEPFUN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("STEPFUN_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("STEPFUN_MODEL", "step-3.5-flash"))
    parser.add_argument(
        "--api-mode",
        choices=["completion_prefill_no_think", "completion_template", "chat"],
        default="completion_prefill_no_think",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-doc-chars", type=int, default=180_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chain-id", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    review_csv_path = Path(args.review_csv)
    viewer_path = Path(args.viewer)
    summary_json_path = Path(args.summary_json)

    if args.rebuild_only:
        rows = read_jsonl(output_path)
        summary = write_summary(summary_json_path, rows)
        write_review_csv(review_csv_path, rows)
        write_viewer(viewer_path, rows, summary)
        print(f"Rebuilt review artifacts for {len(rows)} rows")
        print(f"viewer={viewer_path}")
        print(f"csv={review_csv_path}")
        return 0

    chain_ids = set(args.chain_id) if args.chain_id else None
    dataset_rows = read_jsonl(Path(args.dataset))
    items = flatten_hops(dataset_rows, chain_ids)
    if args.limit is not None:
        items = items[: args.limit]
    completed = set() if args.no_resume else load_completed(output_path)
    pending = [item for item in items if item["audit_id"] not in completed]
    print(
        f"Loaded {len(items)} hop audits from {args.dataset}; "
        f"{len(completed)} completed, {len(pending)} pending",
        flush=True,
    )

    started = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_item = {executor.submit(audit_one, item, args): item for item in pending}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(item)
                row.update({"error": str(exc), "flags": ["runner_error"]})
            append_jsonl(output_path, row)
            done += 1
            if done == 1 or done % 25 == 0 or done == len(pending):
                elapsed = max(1e-6, time.time() - started)
                print(
                    f"completed {done}/{len(pending)} pending "
                    f"({done / elapsed:.2f} rows/s)",
                    flush=True,
                )

    rows = read_jsonl(output_path)
    if chain_ids or args.limit is not None:
        wanted = {item["audit_id"] for item in items}
        rows = [row for row in rows if row.get("audit_id") in wanted]
    summary = write_summary(summary_json_path, rows)
    write_review_csv(review_csv_path, rows)
    write_viewer(viewer_path, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output={output_path}")
    print(f"review_csv={review_csv_path}")
    print(f"viewer={viewer_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
