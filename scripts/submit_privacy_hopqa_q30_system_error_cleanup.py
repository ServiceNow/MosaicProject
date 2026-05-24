#!/usr/bin/env python3
"""Submit targeted cleanup jobs for final q30 rollout system errors.

The final paper viewer treats infrastructure/system rollout errors as
incomplete attempts. This submitter finds unresolved system-error identities in
the current final-eval artifact roots and submits resumable rollout jobs that
target only the affected chain IDs. The underlying rollout runner skips any
``(split, chain_id, repeat)`` identity that already has a terminal row, so a
chain ID that failed in only one repeat will not be regenerated in other
completed repeats.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import submit_privacy_hopqa_q30_external_final_rollouts as external
import submit_privacy_hopqa_q30_final_eval_runner as final


DEFAULT_TARGETS: list[dict[str, Any]] = [
    {
        "runner": "final",
        "artifact_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_final_eval_privacy_prompt_16kgen_32kctx_helperfe_20260522"
        ),
        "run_tag": "final_eval_privacy_prompt_16kgen_32kctx_helperfe_20260522",
    },
    {
        "runner": "external",
        "artifact_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_external_final_rollouts_16kgen_32kctx_helperfe_20260522"
        ),
        "run_tag": "external_final_rollouts_16kgen_32kctx_helperfe_20260522",
    },
    {
        "runner": "external",
        "artifact_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_external_final_rollouts_privacy_prompt_16kgen_32kctx_helperfe_20260522"
        ),
        "run_tag": "external_final_rollouts_privacy_prompt_16kgen_32kctx_helperfe_20260522",
    },
    {
        "runner": "final",
        "artifact_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_final_eval_privreward_tau05_s50_16kgen_32kctx_helperfe_20260523"
        ),
        "run_tag": "final_eval_privreward_tau05_s50_16kgen_32kctx_helperfe_20260523",
    },
    {
        "runner": "final",
        "artifact_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_final_eval_privreward_tau05_s50_privacy_prompt_16kgen_32kctx_helperfe_20260523"
        ),
        "run_tag": "final_eval_privreward_tau05_s50_privacy_prompt_16kgen_32kctx_helperfe_20260523",
    },
]


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("split") or ""), int(row.get("repeat") or 0), str(row.get("chain_id") or ""))


def unresolved_system_errors(model_root: Path) -> list[dict[str, Any]]:
    rollout_path = model_root / "tables" / "rollout_index.jsonl"
    system_error_path = model_root / "tables" / "rollout_system_errors.jsonl"
    completed = {identity(row) for row in read_jsonl(rollout_path)}
    latest_errors: dict[tuple[str, int, str], tuple[str, int, dict[str, Any]]] = {}
    for index, row in enumerate(read_jsonl(system_error_path)):
        key = identity(row)
        timestamp = str(row.get("timestamp") or "")
        current = latest_errors.get(key)
        if current is None or (timestamp, index) >= (current[0], current[1]):
            latest_errors[key] = (timestamp, index, row)
    unresolved = [row for key, (_timestamp, _index, row) in latest_errors.items() if key not in completed]
    return sorted(unresolved, key=lambda row: identity(row))


def load_manifest(artifact_root: Path) -> dict[str, Any]:
    for name in ("final_eval_manifest.json", "external_final_rollouts_manifest.json"):
        path = artifact_root / name
        if path.exists():
            return read_json(path, {})
    return {}


def model_maps() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "final": {str(model["model_label"]): dict(model) for model in final.MODELS},
        "external": external.model_by_label(),
    }


def base_rollout_args(manifest: dict[str, Any]) -> dict[str, Any]:
    args = dict(final.COMMON_ROLLOUT_ARGS)
    manifest_args = manifest.get("rollout_args")
    if isinstance(manifest_args, dict):
        args.update(manifest_args)
    args["splits"] = args.get("splits") or ["test"]
    args["repeats"] = int(args.get("repeats") or 3)
    args["case_attempts"] = 1
    args["case_retry_delay_s"] = 0
    return args


def append_chain_ids(command: str, chain_ids: list[str]) -> str:
    parts: list[Any] = []
    for chain_id in chain_ids:
        parts.extend(["--chain-id", chain_id])
    return command + " " + final.shell_join(parts)


def chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return [values]
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def build_rollout_command(
    runner: str,
    model: dict[str, Any],
    artifact_root: Path,
    manifest: dict[str, Any],
    chain_ids: list[str],
) -> str:
    rollout_args = base_rollout_args(manifest)
    dataset_root = Path(manifest.get("dataset_root") or final.DATASET_ROOT)
    dataset_basename = str(manifest.get("dataset_basename") or final.DATASET_BASENAME)
    helper_url = str(manifest.get("helper_url") or final.HELPER_URL)
    if runner == "final":
        command = final.rollout_command(model, artifact_root, dataset_root, dataset_basename, helper_url, rollout_args)
    elif runner == "external":
        command = external.rollout_command(model, artifact_root, dataset_root, dataset_basename, helper_url, rollout_args)
    else:
        raise ValueError(f"unknown runner: {runner}")
    return append_chain_ids(command, chain_ids)


def submit(command: list[str], dry_run: bool, quiet: bool) -> None:
    if not quiet:
        print(final.shell_join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL if quiet else None)


def write_manifest(path: Path, rows: list[dict[str, Any]], dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "jobs": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name-suffix", default="sysfix")
    parser.add_argument("--account", default=final.ACCOUNT)
    parser.add_argument("--bid", type=int, default=7)
    parser.add_argument("--only-model-label", action="append", default=[])
    parser.add_argument("--skip-model-label", action="append", default=[])
    parser.add_argument("--only-artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--only-chain-id", action="append", default=[])
    parser.add_argument("--quiet", action="store_true", help="Do not print full EAI submit commands.")
    parser.add_argument(
        "--max-chain-ids-per-job",
        type=int,
        default=0,
        help="Split each model cleanup into shard jobs with at most this many target chain IDs.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(final.WORKDIR)
        / "runtime_analysis"
        / "privacy_hopqa_q30_system_error_cleanup_20260523"
        / "cleanup_submit_manifest.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maps = model_maps()
    only_roots = {str(path.resolve()) for path in args.only_artifact_root}
    only_models = set(args.only_model_label)
    skip_models = set(args.skip_model_label)
    only_chain_ids = set(args.only_chain_id)
    manifest_rows: list[dict[str, Any]] = []

    for target in DEFAULT_TARGETS:
        artifact_root = Path(target["artifact_root"])
        if only_roots and str(artifact_root.resolve()) not in only_roots:
            continue
        runner = str(target["runner"])
        manifest = load_manifest(artifact_root)
        available_models = maps[runner]
        model_runs_root = artifact_root / "model_runs"
        if not model_runs_root.exists():
            continue
        for model_root in sorted(path for path in model_runs_root.iterdir() if path.is_dir()):
            model_label = model_root.name
            if only_models and model_label not in only_models:
                continue
            if model_label in skip_models:
                continue
            missing_rows = unresolved_system_errors(model_root)
            if only_chain_ids:
                missing_rows = [row for row in missing_rows if str(row.get("chain_id")) in only_chain_ids]
            if not missing_rows:
                continue
            if model_label not in available_models:
                raise ValueError(f"No {runner} model config for {model_label} in {artifact_root}")
            chain_ids = sorted({str(row.get("chain_id")) for row in missing_rows})
            chain_id_chunks = chunked(chain_ids, args.max_chain_ids_per_job)
            for shard_index, shard_chain_ids in enumerate(chain_id_chunks, start=1):
                shard_missing_rows = [row for row in missing_rows if str(row.get("chain_id")) in set(shard_chain_ids)]
                repeat_counts = Counter(int(row.get("repeat") or 0) for row in shard_missing_rows)
                error_counts = Counter(str(row.get("error_type") or "") for row in shard_missing_rows)
                command = build_rollout_command(
                    runner,
                    dict(available_models[model_label]),
                    artifact_root,
                    manifest,
                    shard_chain_ids,
                )
                suffix = args.job_name_suffix
                if suffix and not suffix.startswith("_"):
                    suffix = "_" + suffix
                if len(chain_id_chunks) > 1:
                    suffix = f"{suffix}_p{shard_index:02d}of{len(chain_id_chunks):02d}"
                job_name = f"phq30_{target['run_tag']}_{model_label}{suffix}"
                model = available_models[model_label]
                submit_cmd = final.base_submit_cmd(
                    name=job_name,
                    cpu=int(model["cpu"]),
                    mem=int(model["mem"]),
                    gpu=int(model.get("gpu", 0)),
                    bid=args.bid,
                    account=args.account,
                )
                submit_cmd.extend(["--", "/bin/bash", "-lc", command])
                manifest_rows.append(
                    {
                        "artifact_root": str(artifact_root),
                        "runner": runner,
                        "model_label": model_label,
                        "missing_identities": len(shard_missing_rows),
                        "target_chain_ids": len(shard_chain_ids),
                        "repeat_counts": dict(sorted(repeat_counts.items())),
                        "error_counts": dict(sorted(error_counts.items())),
                        "job_name": job_name,
                        "shard_index": shard_index,
                        "shard_count": len(chain_id_chunks),
                        "submit_command": submit_cmd,
                    }
                )
                submit(submit_cmd, args.dry_run, args.quiet)

    write_manifest(args.manifest_output, manifest_rows, args.dry_run)
    total = sum(int(row["missing_identities"]) for row in manifest_rows)
    print(f"planned_jobs={len(manifest_rows)} missing_identities={total}", flush=True)
    print(f"manifest={args.manifest_output}", flush=True)


if __name__ == "__main__":
    main()
