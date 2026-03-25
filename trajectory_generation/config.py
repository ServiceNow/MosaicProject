"""Configuration and default paths for trajectory generation."""

from pathlib import Path

# Run output: logs and checkpoints go here
LOG_ROOT = Path.home() / "detailed_trajectory_logs"

# Data paths — symlink or copy data files into trajectory_generation/data/
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

CHUNKS_LOCAL = DATA_DIR / "chunks_local.jsonl"
CHUNKS_WEB = [DATA_DIR / "chunks_web.jsonl"]
SECRET_INVENTORY = DATA_DIR / "secret_inventory.jsonl"
ENTITY_INDEX_DIR = DATA_DIR
TASKS_DIR = REPO_ROOT / "drbench" / "data" / "tasks"
SPACY_MODEL = "en_core_web_trf"

# Default model for chain generation
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# Supported patterns (L=local, W=web)
DEFAULT_PATTERNS = ["LW", "LWL"]
MAX_CHAINS_DEFAULT = 30

# Seed generation: upper bound for random seed (exclusive)
MAX_RANDOM_SEED = 2**31

# Filter stage
FILTER_SCORE_THRESHOLD = 7
FILTER_MAX_WORKERS = 4

# Display truncation (e.g. quote length in filter judge)
QUOTE_DISPLAY_LEN = 300

# Build stage: BM25 retrieval cap when finding bridge candidates
RETRIEVAL_K = 50

# LLM output limit (some models are verbose; use 16k for all calls)
DEFAULT_MAX_TOKENS = 16384

# LLM sampling: normal temps for generative tasks
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
