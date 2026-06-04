import importlib
import os
from pathlib import Path
import subprocess


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


def test_console_script_targets_import() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    in_scripts = False
    targets = []
    for line in pyproject:
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and stripped.startswith("["):
            break
        if in_scripts and "=" in stripped:
            _name, value = stripped.split("=", 1)
            targets.append(value.strip().strip('"'))

    assert targets
    for target in targets:
        module_name, attr_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert hasattr(module, attr_name), target


def test_training_dry_run_allows_placeholder_bundle(tmp_path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MOSAICLEAKS_DATASET_BUNDLE": str(tmp_path / "missing_bundle"),
            "MODEL_PATH": "example-model",
            "REQUIRE_RETRIEVAL_HELPER": "0",
            "DRY_RUN": "1",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/launch_training.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pipelinerl.launch" in result.stdout
