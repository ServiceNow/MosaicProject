# privacy_hopqa helper fix and benchmark, 2026-05-06

## Context

The existing privacy helper showed client-side `Server disconnected` errors during 4B/8B privacy_hopqa training. Its logs also showed uvicorn child worker churn with import failures such as `ModuleNotFoundError: No module named 'argparse'`.

A deeper check showed this was not a simple missing-stdlib issue. The shared `pipeline-rl` conda environment had drifted after the helper job started:

- `pandas` was only visible from user-site packages, but helper jobs set `PYTHONNOUSERSITE=1`.
- Fresh helper jobs using vLLM 0.18 rejected old embedding server flags.
- Fresh helper jobs using pip `faiss-gpu-cu12` failed on H100 with CUDA error 209.

The old helper could keep serving some requests because already-started workers had loaded state, but fresh/replacement workers were not reliable.

## Fixes applied

Environment fixes in `pipeline-rl`:

- Installed `pandas`, `pytz`, and `tzdata` into the conda env, not user-site.
- Installed `faiss-cpu` into the conda env.
- Restored CPU FAISS after testing GPU FAISS wheels.

Code/config fixes:

- Updated `trajectory_generation/src/retrieval/privacy_helper_stack.py` for vLLM 0.18 embedding server flags:
  - `--runner pooling`
  - `--convert embed`
  - `--no-enable-log-requests`
- Updated `privacy_helper_1gpu.yaml` and `privacy_helper_2gpu.yaml` to use CPU FAISS.
- Set default helper worker count to `workers=4`.

## GPU FAISS result

Available pip `faiss-gpu-cu12` wheels are not usable in this environment for the helper:

- `1.14.1.post1`, `1.14.1`, and `1.13.2` import and see the GPU, but abort on H100 with CUDA error 209: `no kernel image is available for execution on the device`.
- `1.12.0` is incompatible with the current NumPy/Python stack.

Until a known H100-compatible FAISS build is available, use CPU FAISS.

## Benchmark summary

Benchmark artifacts are in `runtime_analysis/helper_benchmarks/`.

Request mix:

- `mixed_c24_n96`: 48 BrowseComp searches + 48 local searches at concurrency 24.
- `mixed_c64_n256`: 128 BrowseComp searches + 128 local searches at concurrency 64.

Results on fresh CPU-FAISS helpers:

| config | stress wall | BrowseComp p50/p95 | local p50/p95 | failures | notes |
|---|---:|---:|---:|---:|---|
| `workers=1` | `10.45s` | `4.61s / 6.72s` | `0.27s / 2.55s` | `0` | stable, slower |
| `workers=4` | `6.56s` | `2.57s / 4.47s` | `0.12s / 1.61s` | `0` | selected |
| `workers=8` | `5.68s` | `2.25s / 3.43s` | `0.29s / 0.91s` | `0` | fastest, but logged child restarts |

Decision for the original low-concurrency run: use `workers=4`. It was stable in request tests and did not show the worker restart behavior observed under `workers=8`.

For high-concurrency Privacy HopQA training, the 2-GPU helper spec now uses a
larger single endpoint: `workers=16`, `threadpool_tokens=16`, `cpu=256`, and
`mem=768`. The threadpool cap is intentionally part of the sizing: uvicorn
workers are process-level sharding, while FastAPI sync handlers also use a
per-worker threadpool. Without a cap, increasing workers can create too many
simultaneous CPU-heavy searches inside one helper job.

## Selected helper for next privacy_hopqa runs

Current recommended helper job:

- Job ID: `c42f6748-1451-481e-9690-359ce33fd7b2`
- Internal URL: `http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-phcpuw4final-cca9c1ce:8012`
- Config: one GPU for vLLM embeddings, CPU FAISS, `workers=4`

Final smoke check:

- 64 requests at concurrency 24 succeeded.
- No request failures.
- No helper traceback/import/child-death log entries after the smoke.

## Recommended next workflow

Before restarting full training:

1. Point privacy_hopqa to the selected helper URL.
2. Run a small smoke rollout for both base models with the new prompt and helper.
3. Confirm 4B no longer emits mostly empty `[]` hop plans.
4. Confirm helper errors stay near zero.
5. Confirm traces show actual searches and document reads.
6. If smoke passes, start fresh 4B and 8B training runs with new names, new output dirs, and new W&B project names.
7. Start/update the live viewer for the new runs.
8. Keep the old unstable helper only until the new smoke is complete, then kill it.

Initial-state evaluation should be separate from training:

- Run base 4B and base 8B on train/val/test splits.
- Save full traces and viewer artifacts.
- Run StepFun privacy evaluation for each rollout.
- Use train labels for the future reward model, val for tuning/thresholds, and the held-out company test split only for final generalization evaluation.
