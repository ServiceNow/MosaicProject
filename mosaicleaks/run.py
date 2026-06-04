#!/usr/bin/env python3
"""Run the MosaicLeaks dataset generation pipeline.

Stages: 0 load -> 1 build -> 2 verify -> 3 filter -> 4 reword

Logs go to ~/mosaicleaks_runs/<run_id>/
Resume with --from-stage N.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from mosaicleaks import config
from mosaicleaks.stages.load import run_load, load_state, save_state
from mosaicleaks.stages.build import run_build
from mosaicleaks.stages.verify import run_verify
from mosaicleaks.stages.filter import run_filter
from mosaicleaks.stages.reword import run_reword


def setup_logging(run_dir: Path, verbose: bool) -> None:
    log_file = run_dir / "pipeline.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    for lib in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build, verify, filter, and reword multi-hop chains."
    )
    p.add_argument("--from-stage", type=int, default=0, help="Start from stage N (0-4).")
    p.add_argument("--output-dir", type=Path, default=None, help="Run directory (default: ~/mosaicleaks_runs/run_<ts>).")
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--base-url", default=config.DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=None)

    p.add_argument("--spacy-model", default=config.SPACY_MODEL, help="spaCy NER model.")
    p.add_argument("--chunks-local", type=Path, default=config.CHUNKS_LOCAL)
    p.add_argument("--chunks-web", nargs="+", type=Path, default=[config.BROWSECOMP_WEB_CHUNKS])
    p.add_argument("--curated-web-chunks", nargs="+", type=Path, default=[config.CURATED_WEB_CHUNKS])
    p.add_argument("--secrets", type=Path, default=config.SECRET_INVENTORY)
    p.add_argument("--curated-web-docs", type=Path, default=config.CURATED_WEB_DOCS)
    p.add_argument("--browsecomp-search-url", default=config.BROWSECOMP_SEARCH_URL, help="Optional shared web retriever endpoint.")
    p.add_argument("--task", default=None, help="Only run one task ID.")
    p.add_argument("--company", default=None, help="Only run tasks that match this company name.")

    p.add_argument("--patterns", nargs="+", default=config.DEFAULT_PATTERNS)
    p.add_argument("--per-task", type=int, default=None)
    p.add_argument("--max-chains", type=int, default=config.MAX_CHAINS_DEFAULT)
    p.add_argument("--seeds-from", type=Path, default=None, help="Load seed records from JSONL.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed-schedule", choices=["round_robin", "grouped"], default=config.DEFAULT_SEED_SCHEDULE)
    p.add_argument("--num-shards", type=int, default=config.DEFAULT_NUM_SHARDS)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=config.DEFAULT_MAX_TOKENS,
        help="Max output tokens for long generation calls (default: %(default)s).",
    )
    p.add_argument(
        "--short-max-tokens",
        type=int,
        default=config.SHORT_CALL_MAX_TOKENS,
        help="Max output tokens for ranking, checking, verification, and rewording calls (default: %(default)s).",
    )
    p.add_argument(
        "--prompt-reasoning-mode",
        choices=["step_by_step", "brief_options"],
        default=config.PROMPT_REASONING_MODE,
        help="How much deliberation wording to include in generation prompts.",
    )
    args = p.parse_args()

    config.DEFAULT_MAX_TOKENS = args.max_tokens
    config.SHORT_CALL_MAX_TOKENS = args.short_max_tokens
    config.SEMANTIC_JUDGE_MAX_TOKENS = args.short_max_tokens
    config.BROWSECOMP_SEARCH_URL = args.browsecomp_search_url
    config.PROMPT_REASONING_MODE = args.prompt_reasoning_mode

    run_dir = args.output_dir
    if run_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = config.LOG_ROOT / f"run_{ts}"
    run_dir = Path(run_dir)

    setup_logging(run_dir, args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Run dir: %s", run_dir)

    # Stage 0: Load
    state_0_path = run_dir / "state_0_load.json"
    if args.from_stage <= 0:
        loaded = run_load(
            chunks_local=args.chunks_local,
            chunks_web=args.chunks_web,
            curated_web_chunks=args.curated_web_chunks,
            secret_inventory=args.secrets,
            entity_index_dir=config.ENTITY_INDEX_DIR,
            spacy_model=args.spacy_model,
            curated_web_docs=args.curated_web_docs,
            task_id=args.task,
            company=args.company,
        )
        save_state(loaded, state_0_path)
        logger.info("Stage 0 done: %d tasks", loaded["manifest"]["n_tasks"])
    else:
        loaded = load_state(state_0_path)

    # Stage 1: Build
    if args.from_stage <= 1:
        run_build(
            run_dir=run_dir,
            load_state=loaded,
            patterns=args.patterns,
            per_task=args.per_task,
            max_chains=args.max_chains,
            seed=args.seed,
            seeds_from=args.seeds_from,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            workers=args.workers,
            resume=args.resume,
            seed_schedule=args.seed_schedule,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        logger.info("Stage 1 done")

    # Stage 2: Verify
    chains_path = run_dir / "chains.jsonl"
    verified_path = run_dir / "chains_verified.jsonl"
    if args.from_stage <= 2:
        run_verify(chains_path, verified_path)
        logger.info("Stage 2 done")

    # Stage 3: Filter
    if args.from_stage <= 3:
        run_filter(
            chains_path=verified_path,
            output_dir=run_dir,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            threshold=config.FILTER_SCORE_THRESHOLD,
            workers=min(config.FILTER_MAX_WORKERS, args.workers + 2),
        )
        logger.info("Stage 3 done")

    # Stage 4: Reword
    filtered_path = run_dir / "chains_filtered.jsonl"
    final_path = run_dir / "chains_final.jsonl"
    if args.from_stage <= 4 and filtered_path.exists():
        run_reword(
            chains_path=filtered_path,
            output_path=final_path,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        logger.info("Stage 4 done")

    logger.info("Pipeline complete. Output: %s", run_dir)
    return 0


if __name__ == "__main__":
    main()
