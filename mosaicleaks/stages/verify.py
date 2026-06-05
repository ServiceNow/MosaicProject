"""Stage 3: Verify chains.

Build stage already runs verification. This stage passes through chains
that have verification, optionally re-verifying if requested.
"""

import json
import logging
from pathlib import Path

from mosaicleaks.src.types import ChainState, HopRecord
from mosaicleaks.src.verify_chain import verify_chain

logger = logging.getLogger(__name__)


def run_verify(
    chains_path: Path,
    output_path: Path,
    reverify: bool = False,
    llm=None,
) -> Path:
    """Pass through chains to verified output. Re-verify if requested."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(chains_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            chain = json.loads(line)
            if reverify and llm:
                hops = [
                    HopRecord(
                        hop_number=h["hop_number"],
                        hop_type=h["hop_type"],
                        question=h["question"],
                        answer=h["answer"],
                        doc_id=h["doc_id"],
                        doc_text=h.get("doc_text", ""),
                        quote=h.get("quote", ""),
                    )
                    for h in chain["hops"]
                ]
                state = ChainState(
                    pattern=chain["pattern"],
                    hop_history=hops,
                    global_question=chain.get("global_question", ""),
                    global_answer=chain.get("global_answer", ""),
                )
                verification, _ = verify_chain(state, llm)
                chain["verification"] = {
                    "is_valid": verification.is_valid,
                    "no_docs_pass": verification.no_docs_pass,
                    "first_only_pass": verification.first_only_pass,
                    "last_only_pass": verification.last_only_pass,
                    "all_docs_pass": verification.all_docs_pass,
                    "all_docs_answer": verification.all_docs_answer,
                }
                chain["valid"] = verification.is_valid
            f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Verified %d chains -> %s", n, output_path)
    return output_path
