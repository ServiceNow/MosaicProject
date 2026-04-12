"""Configuration and default paths for trajectory generation."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

# Run output: logs and checkpoints go here.
LOG_ROOT = _env_path("MOSAIC_TRAJ_LOG_ROOT", Path.home() / "trajectory_generation_runs")

# Package-owned data paths.
CHUNKS_LOCAL = _env_path("MOSAIC_TRAJ_CHUNKS_LOCAL", DATA_DIR / "chunks_local.jsonl")
BROWSECOMP_WEB_CHUNKS = _env_path("MOSAIC_TRAJ_CHUNKS_WEB", DATA_DIR / "chunks_web.jsonl")
CURATED_WEB_CHUNKS = _env_path("MOSAIC_TRAJ_CURATED_WEB_CHUNKS", DATA_DIR / "chunks_curated_web.jsonl")
CURATED_WEB_DOCS = _env_path("MOSAIC_TRAJ_CURATED_WEB_DOCS", DATA_DIR / "docs_curated_web.jsonl")
SECRET_INVENTORY = _env_path("MOSAIC_TRAJ_SECRET_INVENTORY", DATA_DIR / "secret_inventory.jsonl")
ENTITY_INDEX_DIR = _env_path("MOSAIC_TRAJ_ENTITY_INDEX_DIR", DATA_DIR)
TASKS_DIR = _env_path("MOSAIC_TRAJ_TASKS_DIR", DATA_DIR / "tasks")
SPACY_MODEL = os.getenv("MOSAIC_TRAJ_SPACY_MODEL", "en_core_web_trf")

# BrowseComp retrieval settings.
BROWSECOMP_EMBEDDING_INDEX_GLOB = os.getenv("MOSAIC_TRAJ_BROWSECOMP_INDEX_GLOB") or str(
    Path.home() / "BrowseComp-Plus" / "indexes" / "qwen3-embedding-4b" / "corpus.shard*_of_*.pkl"
)
BROWSECOMP_EMBEDDING_MODEL = os.getenv("MOSAIC_TRAJ_BROWSECOMP_MODEL", "Qwen/Qwen3-Embedding-4B")
BROWSECOMP_DATASET_NAME = os.getenv("MOSAIC_TRAJ_BROWSECOMP_DATASET", "Tevatron/browsecomp-plus-corpus")
BROWSECOMP_EMBEDDING_DEVICE = (
    os.getenv("MOSAIC_TRAJ_EMBEDDING_DEVICE")
    or os.getenv("DRBENCH_EMBEDDING_DEVICE")
    or "cuda"
)
BROWSECOMP_BM25_RECALL_K = int(os.getenv("MOSAIC_TRAJ_BROWSECOMP_BM25_RECALL_K", "200"))
BROWSECOMP_DENSE_RECALL_K = int(os.getenv("MOSAIC_TRAJ_BROWSECOMP_DENSE_RECALL_K", "200"))
BROWSECOMP_SEARCH_URL = (
    os.getenv("MOSAIC_TRAJ_BROWSECOMP_SEARCH_URL")
    or os.getenv("TRAJ_BROWSECOMP_SEARCH_URL")
)
BROWSECOMP_SEARCH_TIMEOUT = float(
    os.getenv("MOSAIC_TRAJ_BROWSECOMP_SEARCH_TIMEOUT")
    or os.getenv("TRAJ_BROWSECOMP_SEARCH_TIMEOUT")
    or "120"
)

# Default model for chain generation.
DEFAULT_MODEL = os.getenv("MOSAIC_TRAJ_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_BASE_URL = os.getenv("MOSAIC_TRAJ_BASE_URL", "http://127.0.0.1:8000/v1")

# Supported patterns (L=local, W=web).
DEFAULT_PATTERNS = ["LW", "LWL"]
MAX_CHAINS_DEFAULT = 30

# Seed generation: upper bound for random seed (exclusive).
MAX_RANDOM_SEED = 2**31

# Filter stage.
FILTER_SCORE_THRESHOLD = 7
FILTER_MAX_WORKERS = 4

# Display truncation (e.g. quote length in filter judge).
QUOTE_DISPLAY_LEN = 300

# Build stage: BM25 retrieval cap when finding bridge candidates.
RETRIEVAL_K = 50
BUILD_MAX_CANDIDATES = 20
RRF_K = 60
BROWSECOMP_WEB_WEIGHT = 1.0
CURATED_WEB_WEIGHT = 2.0

# Bridge discovery / search strategy.
ENABLE_FAST_PATH = False
LOOKAHEAD_MODE = "beam"  # off | score_only | beam
LOOKAHEAD_DEPTH = 1
BEAM_WIDTH = 3

# Verification strictness.
STRICT_VERIFY_ANSWER_MATCH = True

# Long-form generation budget.
# Historical Step-based runs typically used 16k here; local Qwen runs usually
# used 8k. Keep the local-friendly default in code and let the run scripts raise
# it when they target the Step endpoint.
DEFAULT_MAX_TOKENS = 8192

# Short structured calls use a separate explicit cap so the visible values are
# honest instead of tiny placeholders like 150 or 300 that later get inflated
# by the client floor.
SHORT_CALL_MAX_TOKENS = 8192

# Semantic transition judge.
SEMANTIC_JUDGE_TEMPERATURE = 0.0
SEMANTIC_JUDGE_MAX_TOKENS = SHORT_CALL_MAX_TOKENS

# Concurrent prompt batch size for generation-heavy stages.
GENERATION_BATCH_SIZE = 16

DEFAULT_SEED_SCHEDULE = "round_robin"
DEFAULT_NUM_SHARDS = 1

# Prompt reasoning style for generation prompts.
# "step_by_step" encourages more explicit deliberation.
# "brief_options" asks the model to consider a few options briefly and then
# output only the final concise block.
PROMPT_REASONING_MODE = "step_by_step"

# LLM sampling: normal temps for generative tasks.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
