#!/usr/bin/env python3
"""Publish the finalized Privacy HopQA dataset bundle into MosaicProject.

This copies the finalized Stage M export, stats, uniqueness analysis, plots, and
HTML viewer from the DRBench quality-pass workspace into the canonical dataset
bundle:

  data_files/privacy_hopqa_final_dataset/

An optional --mirror-out path can be supplied for temporary local mirrors.

The script refuses to publish while the source export is marked NOT_READY.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/toolkit")
DEFAULT_SOURCE = ROOT / "nice_code/drbench/chain_review/quality_pass_2026-04-30/proposed_annotations"
DEFAULT_OUT = ROOT / "my_code/MosaicProject/data_files/privacy_hopqa_final_dataset"

REQUIRED = [
    "privacy_hopqa_finalized_2026-05-05.jsonl",
    "privacy_hopqa_finalized_2026-05-05.meta.json",
    "dataset_stats.json",
    "dataset_analysis.json",
    "question_uniqueness_stats.json",
    "dataset_viewer.html",
]


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--mirror-out", default="")
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    mirror_out = Path(args.mirror_out) if args.mirror_out else None
    not_ready = source / "NOT_READY.txt"
    if not_ready.exists() and not args.allow_not_ready:
        raise SystemExit(f"Refusing to publish: {not_ready} exists")

    missing = [name for name in REQUIRED if not (source / name).exists()]
    if missing:
        raise SystemExit(f"Missing required source files: {missing}")

    out.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(source / name, out / name)
    if (source / "plots").exists():
        shutil.copytree(source / "plots", out / "plots", dirs_exist_ok=True)

    rows = count_jsonl(out / "privacy_hopqa_finalized_2026-05-05.jsonl")
    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    uniqueness = json.loads((out / "question_uniqueness_stats.json").read_text(encoding="utf-8"))
    analysis = json.loads((out / "dataset_analysis.json").read_text(encoding="utf-8"))
    manifest = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "out": str(out),
        "mirror_out": str(mirror_out) if mirror_out else None,
        "viewer": "dataset_viewer.html",
        "jsonl": "privacy_hopqa_finalized_2026-05-05.jsonl",
        "n_rows": rows,
        "n_chains_stats": stats.get("n_chains"),
        "n_hops_total": stats.get("n_hops_total"),
        "n_chains_with_local_to_web_transition": (analysis.get("transition_summary") or {}).get("has_local_to_web"),
        "n_effective_doc_hops_distribution": (analysis.get("effective_doc_hops_per_chain") or {}).get("distribution"),
        "n_hops_with_alternate_valid_answers": analysis.get("n_hops_with_alternate_valid_answers"),
        "n_approx_unique_question_clusters": uniqueness.get("n_approx_unique_clusters"),
        "duplicate_rate_approx": uniqueness.get("duplicate_rate_approx"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Privacy HopQA Final Dataset\n\n"
        "Open `dataset_viewer.html` from a local HTTP server in this directory.\n\n"
        "```bash\n"
        "cd /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset\n"
        "python3 -m http.server 8765\n"
        "```\n\n"
        "Then visit `http://localhost:8765/dataset_viewer.html`.\n\n"
        "Files:\n\n"
        "- `privacy_hopqa_finalized_2026-05-05.jsonl`: finalized dataset rows.\n"
        "- `dataset_stats.json`: task/company/pattern/hop/doc statistics.\n"
        "- `dataset_analysis.json`: derived transition, effective-hop, and answer-variant statistics.\n"
        "- `question_uniqueness_stats.json`: approximate near-duplicate question clusters.\n"
        "- `plots/`: matplotlib/seaborn distribution plots used by the viewer.\n"
        "- `manifest.json`: publish provenance.\n",
        encoding="utf-8",
    )
    if mirror_out:
        mirror_out.mkdir(parents=True, exist_ok=True)
        for item in out.iterdir():
            target = mirror_out / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
