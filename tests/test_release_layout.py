import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "mosaicleaks" / "data"


def test_data_directory_contains_only_readme() -> None:
    data_files = sorted(path.relative_to(DATA_DIR).as_posix() for path in DATA_DIR.rglob("*") if path.is_file())
    assert data_files == ["README"]


def test_portable_config_defaults(monkeypatch) -> None:
    for name in [
        "MOSAICLEAKS_RUN_ROOT",
        "MOSAICLEAKS_BROWSECOMP_INDEX_GLOB",
        "MOSAICLEAKS_LLM_MODEL",
        "MOSAICLEAKS_LLM_BASE_URL",
    ]:
        monkeypatch.delenv(name, raising=False)

    from mosaicleaks import config

    config = importlib.reload(config)
    assert config.LOG_ROOT == Path.home() / "mosaicleaks_runs"
    assert config.BROWSECOMP_EMBEDDING_INDEX_GLOB == ""
    assert config.DEFAULT_MODEL == "step-3.5-flash"
    assert config.DEFAULT_BASE_URL == "http://127.0.0.1:8000/v1"
