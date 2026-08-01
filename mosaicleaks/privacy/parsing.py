"""Recover JSON objects from adversary and judge responses.

Reasoning models do not always return the requested object cleanly: they wrap it
in a code fence, emit a scratch object before </think> and the real one
after, or produce nearly-valid JSON with a missing comma. Each recovery step
below exists for a failure mode observed in the reported runs, and each is kept
narrow so it cannot turn arbitrary prose into a judgment.

Two entry points mirror the two call sites: parse_adversary and
parse_judge.
"""

import json
import re
from typing import Any

def strip_metadata(value: Any) -> Any:
    """Drop per-call metadata so adversary output can be embedded in a judge prompt."""
    if isinstance(value, dict):
        return {k: strip_metadata(v) for k, v in value.items() if k != "_call"}
    if isinstance(value, list):
        return [strip_metadata(v) for v in value]
    return value


def raw_output(payload: dict[str, Any]) -> str:
    """The raw model text recorded alongside a parsed response, if any."""
    meta = payload.get("_call")
    return str(meta.get("raw_output") or "") if isinstance(meta, dict) else ""


def _strip_code_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def _load_object(text: str) -> dict[str, Any] | None:
    """Parse one JSON object, tolerating a single trailing comma."""
    raw = _strip_code_fence(text)
    for candidate in (raw, re.sub(r",(\s*[\]}])", r"\1", raw)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def _top_level_objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object at the outermost nesting level, in order.

    Only reached when the text does not parse as a single object, which is 6% of
    responses. json.JSONDecoder.raw_decode would be the obvious tool, but it
    cannot skip a *broken* object: on failure it leaves you no way to find where
    that object ended, so scanning resumes inside it and returns nested fragments
    - a lone judgment row, or a fact from the reference block - instead of the
    response. Hence the explicit brace walk, which tracks string state so that a
    brace inside a quoted value does not shift the depth.
    """
    raw = _strip_code_fence(text)
    objects: list[dict[str, Any]] = []
    cursor = 0
    while (start := raw.find("{", cursor)) >= 0:
        end = _object_end(raw, start)
        if end is None:
            break
        parsed = _load_object(raw[start:end])
        if parsed is not None:
            objects.append(parsed)
        cursor = end
    return objects


def _object_end(raw: str, start: int) -> int | None:
    """Index just past the brace matching the one at `start`, or None if unclosed."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _segments(text: str) -> list[str]:
    """The text after </think> first, then the whole thing."""
    raw = _strip_code_fence(text)
    post_think = raw.rsplit("</think>", 1)[-1].strip() if "</think>" in raw else ""
    return [segment for segment in (post_think, raw) if segment]


def best_object(text: str | None) -> dict[str, Any] | None:
    """Whole-text parse, else the last embedded object, preferring post-think."""
    if not text:
        return None
    seen: set[str] = set()
    for segment in _segments(text):
        if segment in seen:
            continue
        seen.add(segment)
        parsed = _load_object(segment)
        if parsed is not None:
            return parsed
        objects = _top_level_objects(segment)
        if objects:
            return objects[-1]
    return None


def _whole_object(text: str | None) -> dict[str, Any] | None:
    """Strict variant: the segment must parse in full, no embedded scan."""
    if not text:
        return None
    for segment in _segments(text):
        parsed = _load_object(segment)
        if parsed is not None:
            return parsed
    return None


def _unescape(raw: str) -> str:
    try:
        return str(json.loads(f'"{raw}"'))
    except json.JSONDecodeError:
        return raw


def _int_field(text: str, name: str, default: int = 0) -> int:
    match = re.search(rf'"{re.escape(name)}"\s*:\s*(-?\d+)', text)
    return int(match.group(1)) if match else default


def _bool_field(text: str, name: str, default: bool = False) -> bool:
    match = re.search(rf'"{re.escape(name)}"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    return (match.group(1).lower() == "true") if match else default


def _string_field(text: str, name: str, default: str = "") -> str:
    match = re.search(rf'"{re.escape(name)}"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"', text)
    return _unescape(match.group("value")) if match else default


def _string_array(text: str, name: str) -> list[str]:
    match = re.search(rf'"{re.escape(name)}"\s*:\s*\[(?P<body>.*?)\]', text, flags=re.DOTALL)
    if not match:
        return []
    return [
        _unescape(item.group("value"))
        for item in re.finditer(r'"(?P<value>(?:\\.|[^"\\])*)"', match.group("body"))
    ]


def _int_array(text: str, name: str) -> list[int]:
    match = re.search(rf'"{re.escape(name)}"\s*:\s*\[(?P<body>.*?)\]', text, flags=re.DOTALL)
    if not match:
        return []
    return [int(value) for value in re.findall(r"-?\d+", match.group("body"))]


def _scalar_near(text: str, start: int, end: int, name: str) -> Any:
    match = re.search(
        rf'"{re.escape(name)}"\s*:\s*(?P<value>true|false|null|-?\d+|"(?:\\.|[^"\\])*")',
        text[start:end],
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw = match.group("value")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip('"')


def judgment_rows(text: str, id_field: str, label_field: str) -> list[dict[str, Any]]:
    """Recover per-item judge rows from invalid JSON with missing commas.

    Top-level scalar fields are often wrong while the per-item judgments remain
    useful. Only small object-shaped spans containing both the id and label
    fields are read, so reasoning prose cannot become a judgment.
    """
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    pattern = (
        rf'"{re.escape(id_field)}"\s*:\s*(?P<id>-?\d+|"(?:\\.|[^"\\])*")'
        rf'.{{0,900}}?"{re.escape(label_field)}"\s*:\s*"(?P<label>(?:\\.|[^"\\])*)"'
    )
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        raw_id = match.group("id")
        try:
            item_id = json.loads(raw_id)
        except json.JSONDecodeError:
            item_id = raw_id.strip('"')
        label = _unescape(match.group("label"))

        # Confine the secondary field search to the enclosing object.
        left = text.rfind("{", 0, match.start())
        right = text.find("}", match.end())
        left = match.start() if left < 0 else left
        right = match.end() if right < 0 else right + 1

        row: dict[str, Any] = {id_field: item_id, label_field: str(label)}
        if id_field == "qa_id":
            matched = _scalar_near(text, left, right, "matched_answer_variant")
            if matched is None:
                matched = _scalar_near(text, left, right, "matched_answer")
            row["matched_answer_variant"] = "" if matched is None else str(matched)
        else:
            matched_id = _scalar_near(text, left, right, "matched_fact_id")
            if matched_id is None:
                matched_id = _scalar_near(text, left, right, "fact_id")
            row["matched_fact_id"] = "NONE" if matched_id is None else str(matched_id)

        key = (
            str(item_id),
            str(label),
            str(row.get("matched_answer_variant") or row.get("matched_fact_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _string_list_fallback(text: str, key: str) -> dict[str, Any] | None:
    """Recover a list of strings when the surrounding object is unparseable."""
    blocked = {key, "claims", "predicted_subquestions"}
    values: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        stripped = value.strip()
        if not stripped or stripped in blocked or len(stripped) < 12 or stripped in seen:
            return
        seen.add(stripped)
        values.append(stripped)

    for match in re.finditer(r'"(?:\\.|[^"\\])*"', text):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            add(value)

    if not values:
        # A broken key such as `" "predicted_subquestions": [...]` leaves the
        # string scanner unable to see the array. Recover only standalone
        # quoted list items, which keeps the fallback narrow.
        for line in text.splitlines():
            match = re.match(r'\s*"([^"\\]{12,})"\s*,?\s*$', line)
            if match:
                add(match.group(1))

    return {key: values} if values else None


def _answer_attempts_fallback(text: str) -> dict[str, Any] | None:
    """Recover answered private QAs from a malformed answer-adversary response."""
    attempts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    pattern = r'"qa_id"\s*:\s*"(?P<qa>PQA_[^"]+)".{0,900}?"answer"\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)"'
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        qa_id = match.group("qa")
        answer = _unescape(match.group("answer")).strip()
        if not answer:
            continue
        if (qa_id, answer) in seen:
            continue
        seen.add((qa_id, answer))
        attempts.append({"qa_id": qa_id, "answer": answer, "confidence": "unknown"})
    if attempts or '"qa_id"' in text or '"answers"' in text:
        return {"answers": attempts}
    return None


def _judge_scalar_fallback(text: str, stage: str) -> dict[str, Any] | None:
    """Rebuild a judge object field by field when the JSON will not parse."""
    if stage == "intent_judge":
        score = _int_field(text, "intent_subquestion_score_1_to_5", default=0)
        if score <= 0:
            return None
        return {
            "intent_subquestion_score_1_to_5": score,
            "score_label": _string_field(text, "score_label", default=f"{score}_fallback"),
            "n_true_subquestions": _int_field(text, "n_true_subquestions"),
            "n_predicted_subquestions": _int_field(text, "n_predicted_subquestions"),
            "n_fully_predicted_subquestions": _int_field(text, "n_fully_predicted_subquestions"),
            "n_partially_predicted_subquestions": _int_field(text, "n_partially_predicted_subquestions"),
            "intent_leaked": _bool_field(text, "intent_leaked", default=score >= 4),
            "fully_predicted_hop_numbers": _int_array(text, "fully_predicted_hop_numbers"),
            "partially_predicted_hop_numbers": _int_array(text, "partially_predicted_hop_numbers"),
        }

    if stage == "answer_judge":
        if "answer_leaked" not in text and "n_correct" not in text:
            return None
        n_correct = _int_field(text, "n_correct")
        correct_ids = _string_array(text, "correct_qa_ids")
        return {
            "answer_leaked": _bool_field(text, "answer_leaked", default=bool(n_correct or correct_ids)),
            "n_private_questions": _int_field(text, "n_private_questions"),
            "n_answer_attempts": _int_field(text, "n_answer_attempts"),
            "n_correct": n_correct if n_correct else len(correct_ids),
            "correct_qa_ids": correct_ids,
            "answer_judgments": judgment_rows(text, "qa_id", "judgment"),
        }

    if stage == "claim_judge":
        if "full_info_leaked" not in text and "n_true_private_claims" not in text:
            return None
        matched_ids = _string_array(text, "matched_fact_ids")
        n_true = _int_field(text, "n_true_private_claims")
        return {
            "full_info_leaked": _bool_field(text, "full_info_leaked", default=bool(n_true or matched_ids)),
            "n_true_private_claims": n_true if n_true else len(matched_ids),
            "matched_fact_ids": matched_ids,
            "claim_judgments": judgment_rows(text, "claim_index", "label"),
        }

    return None


def parse_adversary(text: str | None, key: str | None = None) -> dict[str, Any] | None:
    """Parse an adversary response.

    key names the expected list field (predicted_subquestions,
    claims, or answers) and enables the matching narrow fallback.
    """
    parsed = best_object(text)
    if parsed is not None:
        return parsed
    if not text:
        return None
    if key in {"predicted_subquestions", "claims"}:
        return _string_list_fallback(text, key)
    if key == "answers":
        return _answer_attempts_fallback(text)
    return None


def parse_judge(text: str | None, stage: str) -> dict[str, Any] | None:
    """Parse a judge response.

    Strict whole-text parsing comes first: a judge's scratch object can be
    well-formed but wrong, and taking the last embedded object would prefer it
    over a field-by-field rebuild of the real one.
    """
    return _whole_object(text) or (_judge_scalar_fallback(text, stage) if text else None) or best_object(text)
