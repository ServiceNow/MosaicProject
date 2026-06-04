"""Runtime configuration for MosaicLeaks dataset generation.

The defaults are intentionally local and portable. Large corpora, task files,
embedding indexes, and generated outputs should be supplied through CLI flags or
``MOSAICLEAKS_*`` environment variables.
"""

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

# Run output: logs and checkpoints go here.
LOG_ROOT = _env_path("MOSAICLEAKS_RUN_ROOT", Path.home() / "mosaicleaks_runs")

# Package-owned data paths.
CHUNKS_LOCAL = _env_path("MOSAICLEAKS_CHUNKS_LOCAL", DATA_DIR / "chunks_local.jsonl")
BROWSECOMP_WEB_CHUNKS = _env_path("MOSAICLEAKS_CHUNKS_WEB", DATA_DIR / "chunks_web.jsonl")
CURATED_WEB_CHUNKS = _env_path("MOSAICLEAKS_CURATED_WEB_CHUNKS", DATA_DIR / "chunks_curated_web.jsonl")
CURATED_WEB_DOCS = _env_path("MOSAICLEAKS_CURATED_WEB_DOCS", DATA_DIR / "docs_curated_web.jsonl")
SECRET_INVENTORY = _env_path("MOSAICLEAKS_SECRET_INVENTORY", DATA_DIR / "secret_inventory.jsonl")
ENTITY_INDEX_DIR = _env_path("MOSAICLEAKS_ENTITY_INDEX_DIR", DATA_DIR)
TASKS_DIR = _env_path("MOSAICLEAKS_TASKS_DIR", DATA_DIR / "tasks")
SPACY_MODEL = os.getenv("MOSAICLEAKS_SPACY_MODEL", "en_core_web_trf")

# BrowseComp retrieval settings.
BROWSECOMP_EMBEDDING_INDEX_GLOB = os.getenv("MOSAICLEAKS_BROWSECOMP_INDEX_GLOB", "")
BROWSECOMP_EMBEDDING_MODEL = os.getenv("MOSAICLEAKS_BROWSECOMP_MODEL", "Qwen/Qwen3-Embedding-4B")
BROWSECOMP_DATASET_NAME = os.getenv("MOSAICLEAKS_BROWSECOMP_DATASET", "Tevatron/browsecomp-plus-corpus")
BROWSECOMP_BM25_RECALL_K = int(os.getenv("MOSAICLEAKS_BROWSECOMP_BM25_RECALL_K", "100"))
BROWSECOMP_DENSE_RECALL_K = int(os.getenv("MOSAICLEAKS_BROWSECOMP_DENSE_RECALL_K", "100"))
# Default OpenAI-compatible endpoint for chain generation. The model string is
# just an endpoint setting; MosaicLeaks does not import provider-specific code.
DEFAULT_MODEL = os.getenv("MOSAICLEAKS_LLM_MODEL", "step-3.5-flash")
DEFAULT_BASE_URL = os.getenv("MOSAICLEAKS_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
BROWSECOMP_EMBEDDING_BASE_URL = (
    os.getenv("MOSAICLEAKS_BROWSECOMP_EMBEDDING_BASE_URL")
    or DEFAULT_BASE_URL
)

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

# Long-form generation budget. Some OpenAI-compatible endpoints spend part of
# this budget on internal reasoning before returning structured output.
DEFAULT_MAX_TOKENS = 16384

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
