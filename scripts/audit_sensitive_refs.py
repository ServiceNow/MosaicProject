#!/usr/bin/env python3
"""Fail if release-blocking internal references appear in tracked text files."""

import argparse
from pathlib import Path


DEFAULT_PATTERNS = [
    "/home/toolkit",
    "/mnt/llmd",
    "snow.research",
    "toolkit-sp",
    "openkey.env",
    "alexg",
    " eai ",
    "EAI",
    "WANDB",
    "wandb.",
    "MosaicProject",
    "trajectory_generation",
]

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "mosaicleaks.egg-info",
}

SKIP_FILES = {
    ".gitignore",
    "audit_sensitive_refs.py",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}:
            yield path


def find_hits(root: Path, patterns: list[str]) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for path in iter_text_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for pattern in patterns:
                if pattern in line:
                    hits.append((path, line_no, pattern))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--pattern", action="append", default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    patterns = args.pattern or DEFAULT_PATTERNS
    hits = find_hits(root, patterns)
    for path, line_no, pattern in hits:
        print(f"{path.relative_to(root)}:{line_no}: {pattern}")
    if hits:
        raise RuntimeError(f"found {len(hits)} blocked reference(s)")


if __name__ == "__main__":
    main()
