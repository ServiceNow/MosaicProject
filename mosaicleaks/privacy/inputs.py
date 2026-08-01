"""Everything read from disk: trajectories, the chain dataset, the Private QA inventory.

A trajectory ("trace") records the actions an agent took for one chain. Two things
are pulled out of it: the web queries, which are all the adversary sees, and the
local documents the agent read, which decide which private facts the judge holds
it against. The dataset supplies the true subquestions and hop facts.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from mosaicleaks.privacy.normalize import compact


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\$\s*", "", text)
    text = re.sub(r"[^\w\s.%$-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def doc_key(value: Any) -> str:
    """Normalize local document identifiers across locators and inventory ids.

    Examples:
    - local/DR0006/IN038/foo.jsonl -> DR0006/IN038/foo
    - /.../tasks/DR0006/files/IN038/foo.txt -> DR0006/IN038/foo
    - local/DR0002/.../file.docx and file.md share the same key.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = raw.split("#", 1)[0]
    raw = raw.replace("\\", "/")
    marker = "/tasks/"
    if marker in raw and "/files/" in raw:
        after = raw.split(marker, 1)[1]
        task_id, rest = after.split("/files/", 1)
        raw = f"local/{task_id}/{rest}"
    if raw.startswith("/"):
        parts = raw.split("/")
        task_idx = next((i for i, part in enumerate(parts) if re.fullmatch(r"DR\d+", part)), None)
        files_idx = next((i for i, part in enumerate(parts) if part == "files"), None)
        if task_idx is not None and files_idx is not None and files_idx + 1 < len(parts):
            raw = "local/" + parts[task_idx] + "/" + "/".join(parts[files_idx + 1 :])
    if raw.startswith("local/"):
        raw = raw[len("local/") :]
    path = PurePosixPath(raw)
    no_suffix = path.with_suffix("")
    return str(no_suffix)


# --------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------


def trace_paths_from_artifact(artifact_root: Path) -> list[Path]:
    traces = set(artifact_root.glob("rollouts/*/*/privacy_hopqa_runs/*/trace.json"))
    traces.update(artifact_root.glob("rollouts/*/*/repeat_*/privacy_hopqa_runs/*/trace.json"))
    return sorted(traces)


def read_trace_manifest(path: Path) -> list[Path]:
    """Trace paths from a manifest: one path per line, or one JSON object per line
    with a trace_path field. Blank lines and # comments are skipped."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not any(line.startswith("{") for line in lines):
        return [Path(line) for line in lines]
    rows = [json.loads(line) for line in lines]
    return [Path(str(row["trace_path"])) for row in rows if isinstance(row, dict) and row.get("trace_path")]


def trace_identity(trace_path: Path, trace: dict[str, Any]) -> dict[str, str]:
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    rollout_id = trace_path.parent.name
    task_id = str(summary.get("task_id") or "")
    chain_id = str(summary.get("chain_id") or "")
    if not task_id or not chain_id:
        parts = rollout_id.split("_")
        if len(parts) >= 2:
            task_id = task_id or parts[0]
            chain_id = chain_id or parts[1]
    return {"rollout_id": rollout_id, "task_id": task_id, "chain_id": chain_id}


def query_rows(trace: dict[str, Any], query_scope: str) -> list[dict[str, Any]]:
    actions = trace.get("actions") if isinstance(trace.get("actions"), list) else []
    rows: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type") or "")
        if action_type not in {"web_search", "local_document_search"}:
            continue
        source = "web" if action_type == "web_search" else "local"
        if query_scope != "both" and source != query_scope:
            continue
        params = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}
        output = action.get("actual_output") if isinstance(action.get("actual_output"), dict) else {}
        query = output.get("query") or params.get("query") or ""
        if not query:
            continue
        rows.append(
            {
                "index": len(rows) + 1,
                "source": source,
                "action_id": action.get("id"),
                "iteration": action.get("iteration"),
                "status": action.get("status"),
                "query": str(query),
            }
        )
    return rows


def retrieved_local_docs(trace: dict[str, Any], scope: str = "read") -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_doc(item: dict[str, Any], *, source_kind: str, hop_number: Any = None) -> None:
        locator = item.get("locator") or item.get("doc_id") or item.get("title") or ""
        key = doc_key(locator)
        if not key:
            return
        dedupe = f"{key}|{source_kind}"
        if dedupe in seen:
            return
        seen.add(dedupe)
        docs.append(
            {
                "doc_key": key,
                "source_kind": source_kind,
                "hop_number": hop_number,
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "locator": item.get("locator"),
            }
        )

    if scope in {"read", "both"}:
        for hop in trace.get("hop_states") or []:
            if not isinstance(hop, dict):
                continue
            for result in hop.get("reader_results") or []:
                if isinstance(result, dict) and result.get("source") == "local":
                    add_doc(result, source_kind="reader_result", hop_number=hop.get("hop_number"))

    if scope in {"search_results", "both"}:
        for action in trace.get("actions") or []:
            if not isinstance(action, dict) or action.get("type") != "local_document_search":
                continue
            output = action.get("actual_output") if isinstance(action.get("actual_output"), dict) else {}
            for result in output.get("results") or []:
                if isinstance(result, dict) and result.get("source") == "local":
                    add_doc(result, source_kind="local_search_result", hop_number=None)
    return docs


def validate_trace_against_dataset(
    trace: dict[str, Any],
    dataset_row: dict[str, Any],
) -> None:
    trace_hops = trace.get("hop_states") or []
    dataset_hops = dataset_row.get("hops") or []
    if len(trace_hops) != len(dataset_hops):
        raise ValueError(
            f"Trace has {len(trace_hops)} hops but frozen dataset has {len(dataset_hops)}"
        )

    fields = ("hop_number", "hop_type", "question", "answer", "doc_id")
    for index, (trace_hop, dataset_hop) in enumerate(
        zip(trace_hops, dataset_hops, strict=True),
        start=1,
    ):
        trace_values = {
            "hop_number": trace_hop.get("hop_number"),
            "hop_type": trace_hop.get("hop_type"),
            "question": trace_hop.get("question"),
            "answer": trace_hop.get("canonical_answer"),
            "doc_id": trace_hop.get("gold_doc_id"),
        }
        dataset_values = {field: dataset_hop.get(field) for field in fields}
        if trace_values != dataset_values:
            raise ValueError(
                f"Trace hop {index} does not match frozen dataset: "
                f"trace={trace_values!r} dataset={dataset_values!r}"
            )


# --------------------------------------------------------------------------
# Chain dataset
# --------------------------------------------------------------------------


def split_paths(dataset: Path, basename: str = "") -> dict[str, Path]:
    """The three split files, in either layout. Missing splits raise here."""
    paths = {}
    for split in ("train", "val", "test"):
        path = dataset / f"{basename}_{split}.jsonl" if basename else dataset / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing dataset split: {path}. Point --dataset at a published bundle "
                f"directory, or add --dataset-basename for a frozen release."
            )
        paths[split] = path
    return paths


def load_dataset_index(dataset: Path, basename: str = "") -> dict[tuple[str, str], dict[str, Any]]:
    """Index every chain by (task_id, chain_id), across all splits.

    Two layouts are accepted. A bundle directory written by
    mosaicleaks-publish-dataset holds train.jsonl / val.jsonl /
    test.jsonl. A frozen release instead names its splits
    {basename}_{split}.jsonl in a shared directory, which is how the
    reported runs pinned their dataset; pass basename for that layout.

    Splits are never inferred from a bare filename: silently scoring against the
    wrong split is the failure this guards against.
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for split, path in split_paths(dataset, basename).items():
        for row in read_jsonl(path):
            task_id = str(row.get("task_id") or "")
            chain_id = str(row.get("chain_id") or "")
            if not task_id or not chain_id:
                continue
            item = dict(row)
            item["_split"] = split
            index[(task_id, chain_id)] = item
    return index


def true_subquestions(dataset_row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hop in dataset_row.get("hops") or []:
        if not isinstance(hop, dict):
            continue
        out.append(
            {
                "hop_number": int(hop.get("hop_number") or len(out) + 1),
                "hop_type": str(hop.get("hop_type") or ""),
                "question": hop.get("question"),
            }
        )
    return out


def hop_accepted_answers(hop: dict[str, Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    raw_values: list[Any] = [hop.get("answer")]
    raw_values.extend(hop.get("accepted_answer_variants") or [])
    raw_values.extend(hop.get("alternate_valid_answers") or [])
    for value in raw_values:
        text = str(value or "").strip()
        key = normalize_text(text)
        if text and key and key not in seen:
            seen.add(key)
            values.append(text)
    return values


def local_hop_facts(dataset_row: dict[str, Any]) -> list[dict[str, Any]]:
    """The chain's own internal answers: the L hops the agent had to look up.

    Distinct from the Private QA facts. These are the handful of internal values
    the multi-hop question is actually built on, so leaking one means the agent
    gave away a step of its own reasoning. Private QA facts are everything else
    sensitive that happened to sit in the documents it opened, whether the chain
    needed it or not. Both are judged, because either counts as a leak.
    """
    out: list[dict[str, Any]] = []
    for hop in dataset_row.get("hops") or []:
        if not isinstance(hop, dict) or str(hop.get("hop_type") or "").upper() != "L":
            continue
        answer = str(hop.get("answer") or "").strip()
        if not answer:
            continue
        out.append(
            {
                "fact_id": f"H{int(hop.get('hop_number') or len(out) + 1)}",
                "hop_number": int(hop.get("hop_number") or len(out) + 1),
                "question": hop.get("question"),
                "canonical_answer": answer,
                "accepted_answers": hop_accepted_answers(hop),
                "doc_id": hop.get("doc_id"),
            }
        )
    return out


# --------------------------------------------------------------------------
# Private QA inventory
# --------------------------------------------------------------------------


def accepted_answers(secret: dict[str, Any]) -> list[str]:
    values = [secret.get("answer")]
    values.extend(secret.get("accepted_answers") or [])
    values.extend(secret.get("accepted_answer_variants") or [])
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_text(text)
        if text and key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def load_private_qa_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id") or row.get("chunk_id") or "")
        key = doc_key(doc_id)
        if not key:
            continue
        for secret in row.get("secrets") or []:
            if not isinstance(secret, dict):
                continue
            question = str(secret.get("question") or "").strip()
            answer = str(secret.get("answer") or "").strip()
            if not question or not answer:
                continue
            if (secret.get("doc_only_check") or {}).get("without_doc") not in {None, "NOT_ANSWERABLE"}:
                continue
            dedupe = (key, compact(question), compact(answer))
            if dedupe in seen:
                continue
            seen.add(dedupe)
            qa_id = "PQA_" + hashlib.sha1(f"{key}\n{question}\n{answer}".encode()).hexdigest()[:12]
            index[key].append(
                {
                    "qa_id": qa_id,
                    "doc_key": key,
                    "doc_id": doc_id,
                    "question": question,
                    "canonical_answer": answer,
                    "accepted_answers": accepted_answers(secret),
                    "secret_type": secret.get("secret_type"),
                }
            )
    return dict(index)


def private_qas_for_docs(
    qa_index: dict[str, list[dict[str, Any]]],
    docs: list[dict[str, Any]],
    max_private_questions: int,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_ids: set[str] = set()
    for doc in docs:
        qas = qa_index.get(doc["doc_key"]) or []
        if not qas:
            missing.append(doc["doc_key"])
            continue
        for qa in qas:
            if qa["qa_id"] in seen_ids:
                continue
            seen_ids.add(qa["qa_id"])
            out.append(qa)
    truncated = False
    if max_private_questions and len(out) > max_private_questions:
        out = out[:max_private_questions]
        truncated = True
    return out, missing, truncated
