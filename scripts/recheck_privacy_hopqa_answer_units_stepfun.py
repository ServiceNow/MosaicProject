#!/usr/bin/env python3
"""Ask StepFun to audit quantity/unit answer contracts and variants.

This script writes proposals only. It does not mutate the canonical dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


DEFAULT_DATASET = Path(
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/"
    "privacy_hopqa_finalized_2026-05-05.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/"
    "stepfun_answer_unit_variant_proposals.jsonl"
)
DEFAULT_STEPFUN_BASE_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000"
SYSTEM_PROMPT = (
    "You are a deterministic JSON audit tool for Privacy HopQA. "
    "Return exactly one JSON object in assistant content. Do not write analysis, markdown, or prose."
)

QUANTITY_RE = re.compile(
    r"\b(how many|how much|what percentage|what percent|what amount|what value|"
    r"nearest|million|billion|thousand|dollars?|usd|\$|percent|%|jobs?|stores?|"
    r"people|employees|customers|vehicles|years?|months?|days?)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:m|bn|b|million|billion|thousand|k|%)?", re.IGNORECASE)
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
}
MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
WORD_TO_INT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chain_id(row: dict[str, Any]) -> str:
    return str(row.get("chain_id") or row.get("id") or row.get("problem_id") or "")


def hop_number(hop: dict[str, Any], fallback: int) -> int:
    value = hop.get("hop_number") or hop.get("hop_index") or fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def truncate(value: Any, limit: int = 1800) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + " ... [truncated]"


def quantity_like(question: str, answer: str, variants: list[str]) -> bool:
    text = " ".join([question, answer, *variants]).lower()
    if QUANTITY_RE.search(text) or NUMBER_RE.search(text):
        return True
    return any(re.search(rf"\b{word}\b", text) for word in NUMBER_WORDS)


def compact_hop(hop: dict[str, Any], fallback_hop_number: int) -> dict[str, Any]:
    keys = [
        "hop_number",
        "hop_type",
        "type",
        "question",
        "answer",
        "accepted_answer_variants",
        "alternate_valid_answers",
        "document_id",
        "doc_id",
        "document_title",
        "title",
        "url",
        "source_url",
        "supporting_span",
        "answer_sentence",
        "source_excerpt",
        "company",
        "task_id",
    ]
    compact = {k: truncate(hop[k]) for k in keys if k in hop and hop[k] not in (None, "")}
    compact["hop_number"] = hop_number(hop, fallback_hop_number)
    for long_key in ("document_text", "document_body", "text", "content"):
        if long_key in hop and hop[long_key]:
            compact[long_key] = truncate(hop[long_key], 2600)
            break
    return compact


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def answer_number_forms(answer: str) -> tuple[list[str], int | None]:
    text = str(answer or "").strip()
    lower = text.lower().replace(",", "")
    forms = [text]
    if lower in WORD_TO_INT:
        number = WORD_TO_INT[lower]
        forms.append(str(number))
        return dedupe(forms), number

    numeric_text = re.sub(
        r"^(?:about|approx(?:imately)?|around|over|more than|less than|nearly|almost|at least|up to)\s+",
        "",
        lower.strip(),
    )
    if re.match(r"^\$?\s*\d+(?:\.\d+)?-[a-z]", numeric_text):
        return dedupe(forms), None
    match = re.match(r"\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|thousand|t|bn|b|m|k|%)?\b", numeric_text)
    if not match:
        return dedupe(forms), None
    value = float(match.group(1))
    scale = match.group(2) or ""
    multiplier = 1
    scale_word = ""
    short_scale = ""
    if scale in {"trillion", "t"}:
        multiplier = 1_000_000_000_000
        scale_word = "trillion"
        short_scale = "T"
    elif scale in {"million", "m"}:
        multiplier = 1_000_000
        scale_word = "million"
        short_scale = "M"
    elif scale in {"billion", "bn", "b"}:
        multiplier = 1_000_000_000
        scale_word = "billion"
        short_scale = "B"
    elif scale in {"thousand", "k"}:
        multiplier = 1_000
        scale_word = "thousand"
        short_scale = "K"
    value_text = f"{value:g}"
    if not scale_word and not value.is_integer():
        forms.append(value_text)
        return dedupe(forms), None
    whole = int(round(value * multiplier))
    if scale_word:
        forms.extend([f"{value_text} {scale_word}", f"{value_text}{short_scale}", f"{value_text}{short_scale.lower()}"])
    forms.extend([str(whole), f"{whole:,}"])
    if "$" in text or "dollar" in lower:
        money_forms = [
            f"${whole}",
            f"${whole:,}",
            f"{whole} dollars",
            f"{whole:,} dollars",
        ]
        if scale_word:
            money_forms.extend(
                [
                    f"${value_text} {scale_word}",
                    f"${value_text}{short_scale}",
                    f"{value_text} {scale_word} dollars",
                ]
            )
        forms.extend(money_forms)
    return dedupe(forms), whole


def is_composite_date_answer(question: str, answer: str) -> bool:
    lower_q = str(question or "").lower()
    lower_a = str(answer or "").lower()
    asks_composite_date = any(
        phrase in lower_q
        for phrase in (
            "month and year",
            "quarter and year",
            "date",
        )
    )
    has_month_year = bool(MONTH_RE.search(lower_a) and re.search(r"\b20\d{2}\b", lower_a))
    has_quarter_year = bool(re.search(r"\bq[1-4]\b", lower_a, re.IGNORECASE) and re.search(r"\b20\d{2}\b", lower_a))
    return asks_composite_date and (has_month_year or has_quarter_year)


def is_year_answer(question: str, answer: str) -> bool:
    text = str(answer or "").strip()
    lower_q = str(question or "").lower()
    return bool(
        re.fullmatch(r"\d{4}", text)
        and any(
            phrase in lower_q
            for phrase in (
                "year",
                "when did",
                "when was",
                "when were",
                "when is",
                "what year",
                "which year",
                "in what year",
                "by what year",
                "since what year",
                "starting from what year",
                "from what year",
            )
        )
    )


def singularize_unit(unit: str) -> str:
    text = str(unit or "").strip()
    if not text:
        return text
    words = text.split()
    last = words[-1]
    lower = last.lower()
    if lower == "people":
        singular = "person"
    elif lower.endswith("ies") and len(last) > 3:
        singular = last[:-3] + "y"
    elif lower.endswith("s") and not lower.endswith("ss") and len(last) > 1:
        singular = last[:-1]
    else:
        singular = last
    return " ".join([*words[:-1], singular])


COUNT_UNIT_STOP_WORDS = (
    "are",
    "were",
    "was",
    "is",
    "did",
    "do",
    "does",
    "will",
    "would",
    "could",
    "can",
    "should",
    "must",
    "live",
    "work",
    "worked",
    "use",
    "have",
    "has",
    "had",
    "created",
    "create",
    "affected",
    "affecting",
    "impacted",
    "impacting",
    "included",
    "including",
    "reported",
    "surveyed",
    "survey",
    "analyzed",
    "analyze",
    "conducted",
    "conduct",
    "documented",
    "document",
    "provided",
    "provide",
    "delivered",
    "deliver",
    "sold",
    "sell",
    "make",
    "makes",
    "made",
    "opened",
    "closed",
    "completed",
    "completing",
    "complete",
    "launched",
    "launch",
    "generated",
    "generate",
    "targeted",
    "target",
    "involving",
    "related",
    "participated",
    "participating",
    "play",
    "played",
    "competed",
    "scored",
    "spent",
    "recorded",
    "sent",
    "selected",
    "requested",
    "attended",
    "joined",
    "hit",
    "ended",
    "meet",
    "meets",
    "renovated",
    "renovate",
    "occurred",
    "sponsored",
    "sponsor",
    "established",
    "establish",
    "handled",
    "handle",
    "awarded",
    "award",
    "served",
    "serve",
    "received",
    "receive",
    "reached",
    "reach",
)


def inferred_count_units(question: str, answer: str) -> list[str]:
    q = str(question or "").strip()
    how_many = re.search(r"(?i)\bhow many\s+", q)
    if not how_many:
        return []
    stop_pattern = "|".join(re.escape(word) for word in COUNT_UNIT_STOP_WORDS)
    after_how_many = q[how_many.end() :]
    match = re.match(
        rf"(?i)(.+?)(?:\s+(?:{stop_pattern})\b|[?])",
        after_how_many,
    )
    if not match:
        return []
    raw_unit = re.sub(r"\b(the|a|an)\b", "", match.group(1), flags=re.IGNORECASE).strip()
    if raw_unit.lower().strip(" .?()[],:;") in {"is that", "is it", "is this"}:
        prefix = q[: how_many.start()]
        prefix_matches = re.findall(
            r"\bof\s+([^,;—-]+?)(?:\s+(?:with|were|was|are|is|that|who|which)\b|$)",
            prefix,
            flags=re.IGNORECASE,
        )
        if prefix_matches:
            raw_unit = prefix_matches[-1].strip()
    raw_unit = re.split(r"\s+(?:minimum|maximum|required|needed)\b", raw_unit, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if raw_unit.lower().startswith("total"):
        prefix = q[: how_many.start()]
        total_match = re.search(r"([A-Za-z][A-Za-z/-]+)\s+out of\s*$", prefix, flags=re.IGNORECASE)
        if total_match:
            raw_unit = total_match.group(1)
    raw_unit = re.split(r"\s+(?:at|in|from|for|by|after|before|during|through|while|with|over)\b", raw_unit, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    raw_unit = re.sub(r"^(?:19|20)\d{2}\s+", "", raw_unit).strip()
    raw_unit = raw_unit.strip(" .?()[],:;")
    if re.match(r"(?i)^(minutes|minute|hours|hour|days|day|weeks|week|months|month|years|year|words|word|tons|ton|acres|acre|zettabytes|zettabyte)\b", raw_unit):
        raw_unit = raw_unit.split(" of ", 1)[0].strip()
    if len(raw_unit.split()) > 8:
        return []
    if not raw_unit:
        return []
    _, number = answer_number_forms(answer)
    singular = singularize_unit(raw_unit)
    if number == 1 and singular:
        return [singular]
    return [raw_unit]


def candidate_variants(question: str, answer: str, current: list[str]) -> list[str]:
    if is_composite_date_answer(question, answer) or is_year_answer(question, answer):
        forms = [str(answer or "").strip()]
    else:
        forms = answer_number_forms(answer)[0]
    candidates = dedupe(current + forms)
    unit_phrases: list[str] = []
    count_units = inferred_count_units(question, answer)
    if count_units:
        unit_phrases.extend(count_units)
    def already_has_count_unit(value: str) -> bool:
        text = value.lower()
        text = re.sub(r"^(?:about|approx(?:imately)?|around|over|more than|less than|nearly|almost|at least|up to)\s+", "", text)
        text = re.sub(r"[$€£]?\d[\d,]*(?:\.\d+)?(?:\s|-)?", " ", text)
        text = re.sub(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b", " ", text)
        text = re.sub(r"\b(trillion|billion|million|thousand|percent|per cent|bn|[tbmk])\b", " ", text)
        return bool(re.search(r"[a-z]", text))

    for form in list(candidates):
        clean = form.strip()
        if not clean:
            continue
        for unit in unit_phrases:
            if unit.lower() not in clean.lower() and not already_has_count_unit(clean):
                candidates.append(f"{clean} {unit}")
    return dedupe(candidates)


def money_precision_phrase(answer: str) -> str:
    text = str(answer or "")
    lower = text.lower().replace(",", "")
    match = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?", lower)
    if not match:
        return "nearest dollar"
    value = float(match.group(1))
    scale = match.group(2) or ""
    if scale in {"million", "m"}:
        if not value.is_integer():
            return "nearest tenth of a million dollars"
        whole = int(value)
        if whole % 100 == 0:
            return "nearest hundred million dollars"
        if whole % 10 == 0:
            return "nearest ten million dollars"
        return "nearest million dollars"
    if scale in {"billion", "b"}:
        if not value.is_integer():
            return "nearest tenth of a billion dollars"
        return "nearest billion dollars"
    if scale in {"thousand", "k"}:
        if not value.is_integer():
            return "nearest tenth of a thousand dollars"
        return "nearest thousand dollars"
    return "nearest dollar"


def candidate_question_rewrites(question: str, answer: str) -> list[str]:
    q = str(question or "").strip()
    lower_answer = str(answer or "").lower()
    lower_q = q.lower()
    rewrites: list[str] = []
    if "$" in str(answer) or "dollar" in lower_answer:
        if "dollar" not in lower_q and "usd" not in lower_q:
            precision = money_precision_phrase(answer)
            if q.lower().startswith("how much is "):
                rewrites.append(f"How much, rounded to the {precision}, is " + q[len("How much is ") :])
            else:
                rewrites.append(q.rstrip("?") + f", rounded to the {precision}?")
    if "million" in lower_answer or re.search(r"\d+(?:\.\d+)?m\b", lower_answer):
        if "million" not in lower_q:
            if q.lower().startswith("how many new jobs will "):
                rewrites.append("How many new jobs, rounded to the nearest tenth of a million jobs, will " + q[len("How many new jobs will ") :])
            elif q.lower().startswith("how many "):
                rewrites.append(q.rstrip("?") + ", rounded to the nearest tenth of a million?")
    if "billion" in lower_answer or re.search(r"\d+(?:\.\d+)?b\b", lower_answer):
        if "billion" not in lower_q:
            rewrites.append(q.rstrip("?") + ", rounded to the nearest tenth of a billion?")
    return dedupe(rewrites)


def build_prompt(row: dict[str, Any], hop: dict[str, Any], fallback_hop_number: int) -> str:
    compact = compact_hop(hop, fallback_hop_number)
    evidence = (
        compact.get("supporting_span")
        or compact.get("answer_sentence")
        or compact.get("source_excerpt")
        or compact.get("document_text")
        or compact.get("document_body")
        or compact.get("text")
        or compact.get("content")
        or ""
    )
    current = dedupe(as_list(hop.get("accepted_answer_variants")) + as_list(hop.get("alternate_valid_answers")) + as_list(hop.get("answer")))
    candidates = candidate_variants(str(hop.get("question") or ""), str(hop.get("answer") or ""), current)
    question_rewrites = candidate_question_rewrites(str(hop.get("question") or ""), str(hop.get("answer") or ""))
    chain_lines = []
    hops = row.get("hops") or []
    for i, chain_hop in enumerate(hops):
        marker = " <-- REVIEW THIS HOP" if chain_hop is hop else ""
        chain_lines.append(
            f"{hop_number(chain_hop, i + 1)}. "
            f"[{chain_hop.get('hop_type') or chain_hop.get('type')}] "
            f"Q: {chain_hop.get('question')} | A: {chain_hop.get('answer')}{marker}"
        )
    return (
        "Return JSON only, one object.\n"
        "This QA hop is part of a multi-hop question chain. The gold answer is the canonical value that may condition the next hop. "
        "Accepted aliases are for evaluation only; do not reject a natural-language alias just because the canonical chaining value is shorter. "
        "Every accepted answer must still be valid for this hop and must not change the fact needed by later hops. "
        "Do not accept partial answers, over-broad answers, or answers that omit a requested date component/unit.\n"
        f"Full chain:\n" + "\n".join(chain_lines) + "\n"
        f"Question: {hop.get('question')}\n"
        f"Gold answer: {hop.get('answer')}\n"
        f"Evidence: {truncate(evidence, 2200)}\n"
        f"Candidate exact-match answers: {'; '.join(candidates)}\n"
        f"Candidate question rewrites: {'; '.join(question_rewrites) if question_rewrites else 'none'}\n"
        "The preferred answer is the gold answer. Unit-bearing forms are accepted aliases for evaluation completeness, not replacements for the preferred answer. "
        "Judge every candidate answer independently for whether it is valid for the question and the same fact as the gold answer. "
        "Candidates that add the counted unit, such as jobs/stores/dollars, are valid if the numeric fact is unchanged. "
        "For How many questions, include grammatical exact number plus counted-unit aliases unless they change the fact. "
        "Examples of valid aliases: 25 stores for how many stores, 3,000 jobs for how many jobs, 12 full ICC members for how many full ICC members. "
        "If the question asks for month and year, quarter and year, or a full date, reject year-only candidates. "
        "If candidate question rewrites is not none, choose one rewrite; otherwise use null.\n"
        'Return exactly: {"validity": {"candidate string": true_or_false}, "question_rewrite": string_or_null}.\n'
        "JSON:"
    )


def normalize_proposal(
    proposal: dict[str, Any],
    current_variants: list[str],
    candidate_list: list[str],
    question_rewrites: list[str],
    row: dict[str, Any],
    hop: dict[str, Any],
    fallback_hop_number: int,
) -> dict[str, Any]:
    normalized = dict(proposal)
    postprocess_notes: list[str] = []
    validity = normalized.get("validity") or normalized.get("candidate_validity")
    if not isinstance(validity, dict):
        top_level_validity = {
            candidate: normalized[candidate]
            for candidate in candidate_list
            if candidate in normalized and isinstance(normalized[candidate], (bool, str, int))
        }
        if top_level_validity:
            validity = top_level_validity
    valid_candidates = None
    rejected_candidates: list[str] = []
    if isinstance(validity, dict):
        valid_candidates = []
        for candidate in candidate_list:
            value = validity.get(candidate)
            if isinstance(value, str):
                is_valid = value.strip().lower() in {"true", "yes", "valid", "1"}
            else:
                is_valid = bool(value)
            if is_valid:
                valid_candidates.append(candidate)
            else:
                rejected_candidates.append(candidate)
    if valid_candidates is None:
        valid_candidates = normalized.get("valid_candidates")
        if valid_candidates is None:
            valid_candidates = normalized.get("selected_candidates") or normalized.get("valid") or normalized.get("variants")
    if isinstance(valid_candidates, list):
        accepted = dedupe(current_variants + [str(value) for value in valid_candidates])
    else:
        accepted = list(current_variants)
        postprocess_notes.append("no_valid_candidates_returned")
    normalized["valid_candidates"] = accepted
    normalized["accepted_answer_variants"] = accepted
    normalized["alternate_valid_answers"] = dedupe([value for value in accepted if value.lower() not in {v.lower() for v in current_variants}])
    rejected_keys = {value.lower() for value in rejected_candidates}
    normalized["rejected_existing_answer_variants"] = dedupe(
        [value for value in current_variants if value.lower() in rejected_keys]
    )
    if isinstance(valid_candidates, list) and len(accepted) > len(valid_candidates):
        postprocess_notes.append("preserved_existing_accepted_variants")
    normalized.setdefault("issue_types", [])
    normalized.setdefault("rationale", "")
    normalized.setdefault("confidence", 0.8)
    if normalized.get("question_rewrite") == "":
        normalized["question_rewrite"] = None
    if not normalized.get("question_rewrite") and question_rewrites:
        normalized["question_rewrite"] = question_rewrites[0]
        postprocess_notes.append("used_deterministic_question_rewrite_candidate")
    confidence = normalized.get("confidence")
    if isinstance(confidence, str):
        normalized["confidence"] = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(confidence.lower(), confidence)
    if "needs_edit" not in normalized:
        rewrite = normalized.get("question_rewrite")
        accepted = normalized.get("accepted_answer_variants") or []
        normalized["needs_edit"] = bool(rewrite) or len(accepted) > len(current_variants)
    normalized["chain_id"] = chain_id(row)
    normalized["hop_number"] = hop_number(hop, fallback_hop_number)
    normalized["postprocess_notes"] = postprocess_notes
    return normalized


def parse_chain_hop(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected CHAIN_ID:HOP_NUMBER")
    cid, hop = value.split(":", 1)
    try:
        return cid, int(hop)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hop number must be an integer") from exc


def select_targets(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    explicit_hops = set(args.chain_hop or [])
    explicit_chains = set(args.chain_id or [])
    targets: list[tuple[dict[str, Any], dict[str, Any], int]] = []

    for row in rows:
        cid = chain_id(row)
        if explicit_chains and cid not in explicit_chains:
            continue
        for i, hop in enumerate(row.get("hops") or []):
            hn = hop_number(hop, i + 1)
            if explicit_hops and (cid, hn) not in explicit_hops:
                continue
            question = str(hop.get("question") or "")
            answer = str(hop.get("answer") or "")
            variants = as_list(hop.get("accepted_answer_variants"))
            if not explicit_hops and not quantity_like(question, answer, variants):
                continue
            targets.append((row, hop, i + 1))
            if args.limit and len(targets) >= args.limit:
                return targets
    return targets


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start in reversed([match.start() for match in re.finditer(r"\{", text)]):
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise


async def call_stepfun(client: AsyncOpenAI, args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    if args.endpoint == "completion":
        response = await client.completions.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            stop=["\n", "\nReturn JSON only", "\nQuestion:", "\nReturn one JSON object", "\nTask:", "\nchain_id:"],
        )
        return parse_json_object(response.choices[0].text or "")

    kwargs = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }
    if args.response_format:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    text = message.content or ""
    if not text:
        reasoning = getattr(message, "reasoning", None) or ""
        raise RuntimeError(f"StepFun returned no assistant content; hidden reasoning chars={len(reasoning)}")
    return parse_json_object(text)


async def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.dataset)
    targets = select_targets(args, rows)
    if not targets:
        print("No matching hops found.")
        return 1

    if args.dry_run:
        for row, hop, fallback in targets:
            print(build_prompt(row, hop, fallback))
            print("\n" + "=" * 88 + "\n")
        return 0

    completed: set[tuple[str, int]] = set()
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                completed.add((str(record.get("chain_id")), int(record.get("hop_number"))))
    if completed:
        targets = [
            (row, hop, fallback)
            for row, hop, fallback in targets
            if (chain_id(row), hop_number(hop, fallback)) not in completed
        ]
        print(f"Resuming: skipped {len(completed)} completed proposal rows; {len(targets)} remain.")
    if not targets:
        print("All matching hops already completed.")
        return 0

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    client = AsyncOpenAI(api_key=args.api_key, base_url=base_url)

    output_mode = "a" if args.resume else "w"
    output_f = args.output.open(output_mode, encoding="utf-8")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(args.concurrency, 1))
    completed_count = 0

    async def process_target(row: dict[str, Any], hop: dict[str, Any], fallback: int) -> dict[str, Any]:
        prompt = build_prompt(row, hop, fallback)
        current_variants = dedupe(
            as_list(hop.get("accepted_answer_variants"))
            + as_list(hop.get("alternate_valid_answers"))
            + as_list(hop.get("answer"))
        )
        candidate_list = candidate_variants(
            str(hop.get("question") or ""),
            str(hop.get("answer") or ""),
            current_variants,
        )
        question_rewrites = candidate_question_rewrites(str(hop.get("question") or ""), str(hop.get("answer") or ""))
        last_error: Exception | None = None
        for attempt in range(1, args.attempts + 1):
            try:
                raw_proposal = await call_stepfun(client, args, prompt)
                break
            except Exception as exc:
                last_error = exc
                if attempt >= args.attempts:
                    raise
                await asyncio.sleep(min(2 * attempt, 10))
        else:
            raise RuntimeError("StepFun proposal failed") from last_error

        proposal = normalize_proposal(raw_proposal, current_variants, candidate_list, question_rewrites, row, hop, fallback)
        hn = hop_number(hop, fallback)
        return {
            "chain_id": chain_id(row),
            "hop_number": hn,
            "question": hop.get("question"),
            "answer": hop.get("answer"),
            "accepted_answer_variants_before": as_list(hop.get("accepted_answer_variants")),
            "alternate_valid_answers_before": as_list(hop.get("alternate_valid_answers")),
            "proposal": proposal,
        }

    async def run_one(row: dict[str, Any], hop: dict[str, Any], fallback: int) -> None:
        nonlocal completed_count
        async with sem:
            record = await process_target(row, hop, fallback)
        async with lock:
            output_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_f.flush()
            completed_count += 1
            total_done = len(completed) + completed_count
            total_targets = len(completed) + len(targets)
            proposal = record["proposal"]
            print(f"[{total_done}/{total_targets}] {record['chain_id']} hop {record['hop_number']}: needs_edit={proposal.get('needs_edit')}")
            print(f"  question: {record['question']}")
            print(f"  rewrite: {proposal.get('question_rewrite')}")
            print(f"  accepted variants: {proposal.get('accepted_answer_variants')}")
            print(f"  alternate valid answers: {proposal.get('alternate_valid_answers')}")
            print(f"  issue_types: {proposal.get('issue_types')}")

    try:
        await asyncio.gather(*(run_one(row, hop, fallback) for row, hop, fallback in targets))
    finally:
        output_f.close()

    print(f"Wrote {len(targets)} proposal rows to {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chain-id", action="append", help="Restrict to a chain id; may be repeated.")
    parser.add_argument(
        "--chain-hop",
        action="append",
        type=parse_chain_hop,
        help="Restrict to a specific CHAIN_ID:HOP_NUMBER; may be repeated.",
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_STEPFUN_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="step-3.5-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--response-format", action="store_true")
    parser.add_argument("--endpoint", choices=["completion", "chat"], default="completion")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
