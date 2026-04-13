"""Stage 5: Reword questions to fix doc references and vague phrasing."""

import json
import logging
from pathlib import Path

from trajectory_generation.src.format_questions import format_numbered_questions
from trajectory_generation.src.llm import LLMClient

from trajectory_generation import config
from trajectory_generation.prompts import REWORD_PROMPT

logger = logging.getLogger(__name__)


def _format_prev_hops(prev_hops: list[dict]) -> str:
    """Format previous hops as Q/A/quote for context."""
    if not prev_hops:
        return "(none — this is the first question)"
    lines = []
    for j, h in enumerate(prev_hops, 1):
        q = h.get("question", "")
        a = h.get("answer", "")
        quote = h.get("quote", "")
        lines.append(f"Hop {j}: Q: {q}\n     A: {a}")
        if quote:
            lines.append(f"     Quote: {quote[:300]}{'...' if len(quote) > 300 else ''}")
    return "\n".join(lines)


def _reword_question(question: str, prev_hops: list[dict], llm: LLMClient) -> str:
    """Rewrite a single question to fix doc refs, vague phrasing, and improve linearity."""
    prev_context = _format_prev_hops(prev_hops)
    prompt = REWORD_PROMPT.format(prev_hops=prev_context, question=question)
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.SHORT_CALL_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
        top_k=config.DEFAULT_TOP_K,
        use_min_max_tokens=False,
    )
    return raw.strip().strip('"').strip()


def run_reword(
    chains_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
    api_key: str | None,
) -> Path:
    """Reword each hop's question. Returns path to chains_final.jsonl."""
    chains = []
    with open(chains_path) as f:
        for line in f:
            if line.strip():
                chains.append(json.loads(line))

    llm = LLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        min_max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f_out:
        for chain in chains:
            hops = chain.get("hops", [])
            for i, hop in enumerate(hops):
                q = hop.get("question", "")
                if q:
                    try:
                        prev = hops[:i]
                        fixed = _reword_question(q, prev, llm)
                        if fixed and len(fixed) > 5:
                            hop["question"] = fixed
                    except Exception as e:
                        logger.warning("Reword failed for Q %r: %s", q[:50], e)

            try:
                hop_dicts = [
                    {"question": h["question"], "answer": h["answer"], "hop_number": h["hop_number"]}
                    for h in hops
                ]
                numbered = format_numbered_questions(hop_dicts)
                chain["numbered_questions"] = numbered
                chain["global_question"] = numbered
            except ValueError as e:
                logger.warning("format_numbered_questions failed for chain %s: %s",
                              chain.get("chain_id", "?"), e)

            f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")

    logger.info("Reworded %d chains → %s", len(chains), output_path)
    return output_path
