"""Stage 4: Filter chains with LLM judge scoring."""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trajectory_generation.src.llm import LLMClient

from trajectory_generation import config

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are judging multi-hop research chains.

A chain consists of numbered questions where each answer feeds into the next via back-references (shown as "(N)"). Score the chain 1-10 based on:

- Search dependency: For each hop, imagine the agent sees the question but does NOT know the previous hop's answer (the back-reference is blank). Could it still write a search query that finds the right document? If YES, the hop has weak dependency and the chain is less valuable. Strong chains require the previous answer to formulate viable searches.
  BAD: "In (1), which company partnered with Northwind Robotics on the Orion rollout?" → If "Northwind Robotics Orion rollout partnership" already finds the source, then (1) is decorative.
  GOOD: "Which company had (1) marketshare in the arcade market in 2006?" → Without (1)=60%, the query matches every arcade company. Previous answer is essential.
- No document self-references: Reject "according to the document", "the report", "this paper", etc.
- Answer uniqueness: Each question should have exactly one defensible answer.
- Answer verifiability: Concrete, unambiguous values. Good: "2022", "15%", "FDA". Bad: "the next month", "approximately half".
- Entity identity: When an answer feeds into the next as a back-reference, ensure the same string doesn't denote different entities in different docs (e.g. acronyms).

Examples:
Score 10: Every hop's search depends on the previous answer, no self-refs, every answer uniquely verifiable.
Score 7: Strong dependency with minor concerns.
Score 2: Hops independent; e.g. "What book is at (1) discount?" where (1) is "30%" — thousands of books match, connection is coincidence.

Respond in this exact format:
SCORE: <integer 1-10>
JUSTIFICATION: <1-2 sentences>"""


def _build_user_message(chain: dict) -> str:
    parts = [
        f"Numbered questions:\n{chain.get('numbered_questions', '')}",
        f"\nGlobal answer: {chain.get('global_answer', '')}",
        "\nHops:",
    ]
    for hop in chain.get("hops", []):
        doc_short = hop["doc_id"].split("/")[-1][:60] if hop.get("doc_id") else "?"
        quote = hop.get("quote", "")[:config.QUOTE_DISPLAY_LEN]
        parts.append(
            f"  Hop {hop['hop_number']}: Q: {hop['question']}\n"
            f"    A: {hop['answer']}\n    Quote: \"{quote}\""
        )
    return "\n".join(parts)


def _parse_score(response: str) -> tuple[int, str]:
    score, justification = None, ""
    for line in response.splitlines():
        s = line.strip().strip("*")
        if s.upper().startswith("SCORE:"):
            m = re.search(r'\d+', s.split(":", 1)[1])
            if m:
                score = max(1, min(10, int(m.group())))
        elif s.upper().startswith("JUSTIFICATION:"):
            justification = s.split(":", 1)[1].strip()
    if score is None:
        m = re.search(r'(?i)score\s*[:=]\s*(\d{1,2})', response)
        if m:
            score = max(1, min(10, int(m.group(1))))
    if score is None:
        raise ValueError(f"Could not parse score from: {response[:200]}")
    return score, justification


def _is_valid_chain(chain: dict) -> bool:
    if not chain.get("valid", False):
        return False
    v = chain.get("verification", {})
    return not v or v.get("is_valid", False)


def run_filter(
    chains_path: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key: str | None,
    threshold: int | None = None,
    workers: int | None = None,
) -> Path:
    """Score and filter chains. Returns path to filtered output."""
    threshold = threshold if threshold is not None else config.FILTER_SCORE_THRESHOLD
    workers = workers if workers is not None else config.FILTER_MAX_WORKERS
    chains = []
    with open(chains_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                chains.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in %s line %d: %s", chains_path, i + 1, e)

    to_score = [(i, c) for i, c in enumerate(chains) if _is_valid_chain(c)]
    llm = LLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        min_max_tokens=config.DEFAULT_MAX_TOKENS,
    )

    results: dict[int, dict] = {}
    lock = threading.Lock()

    def _score_one(idx: int, chain: dict) -> None:
        try:
            msgs = [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": _build_user_message(chain)},
            ]
            raw = llm.chat(
                msgs,
                temperature=config.DEFAULT_TEMPERATURE,
                max_tokens=config.SHORT_CALL_MAX_TOKENS,
                top_p=config.DEFAULT_TOP_P,
                top_k=config.DEFAULT_TOP_K,
                use_min_max_tokens=False,
            )
            score, justification = _parse_score(raw)
            with lock:
                results[idx] = {"score": score, "justification": justification}
        except Exception as e:
            logger.error("Score failed for chain %s: %s", chain.get("chain_id"), e)
            with lock:
                results[idx] = {"score": 0, "justification": str(e)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_one, idx, chain) for idx, chain in to_score]
        for f in futures:
            f.result()

    filtered_path = output_dir / "chains_filtered.jsonl"
    scored_path = output_dir / "chains_scored.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_passed = 0
    with open(filtered_path, "w") as f_filt, open(scored_path, "w") as f_scored:
        for i, chain in enumerate(chains):
            if i in results:
                chain["posthoc"] = results[i]
                if results[i]["score"] >= threshold:
                    n_passed += 1
                    f_filt.write(json.dumps(chain, ensure_ascii=False) + "\n")
            f_scored.write(json.dumps(chain, ensure_ascii=False) + "\n")

    logger.info("Filtered: %d/%d passed (threshold %d)", n_passed, len(to_score), threshold)
    return filtered_path
