# privacy helper benchmark summary 2026-05-06

Fixes applied:
- Updated `trajectory_generation/src/retrieval/privacy_helper_stack.py` for vLLM 0.18 embedding server flags: `--runner pooling --convert embed` and `--no-enable-log-requests`.
- Installed required packages into the `pipeline-rl` conda env with `PYTHONNOUSERSITE=1`: `pandas`, `pytz`, `tzdata`, and `faiss-cpu`.
- Avoided pip `faiss-gpu-cu12` on H100: tested available wheels and they abort with CUDA error 209 (`no kernel image is available for execution on the device`) or are incompatible with the current NumPy stack.
- Set helper YAML defaults to CPU FAISS and `workers=4`.

Selected helper for training smoke/new runs:
- Job: `c42f6748-1451-481e-9690-359ce33fd7b2`
- Internal URL: `http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-phcpuw4final-cca9c1ce:8012`
- Config: one GPU for vLLM embeddings, CPU FAISS, `workers=4`.

Benchmark method:
- Mixed BrowseComp and local search requests against fresh helper jobs.
- `mixed_c24_n96`: 48 BrowseComp + 48 local, concurrency 24.
- `mixed_c64_n256`: 128 BrowseComp + 128 local, concurrency 64.

Results:

| helper | scenario | endpoint | ok/fail | wall s | p50 s | p95 s | endpoint req/s |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu_faiss_w1 | mixed_c24_n96 | browsecomp | 48/0 | 4.005 | 1.447 | 2.297 | 11.985 |
| cpu_faiss_w1 | mixed_c24_n96 | local | 48/0 | 4.005 | 0.164 | 1.083 | 11.985 |
| cpu_faiss_w1 | mixed_c64_n256 | browsecomp | 128/0 | 10.446 | 4.611 | 6.721 | 12.253 |
| cpu_faiss_w1 | mixed_c64_n256 | local | 128/0 | 10.446 | 0.268 | 2.545 | 12.253 |
| cpu_faiss_w4 | mixed_c24_n96 | browsecomp | 48/0 | 3.377 | 0.833 | 3.231 | 14.216 |
| cpu_faiss_w4 | mixed_c24_n96 | local | 48/0 | 3.377 | 0.072 | 0.182 | 14.216 |
| cpu_faiss_w4 | mixed_c64_n256 | browsecomp | 128/0 | 6.560 | 2.573 | 4.467 | 19.513 |
| cpu_faiss_w4 | mixed_c64_n256 | local | 128/0 | 6.560 | 0.120 | 1.608 | 19.513 |
| cpu_faiss_w8 | mixed_c24_n96 | browsecomp | 48/0 | 3.253 | 1.356 | 2.341 | 14.754 |
| cpu_faiss_w8 | mixed_c24_n96 | local | 48/0 | 3.253 | 0.103 | 0.384 | 14.754 |
| cpu_faiss_w8 | mixed_c64_n256 | browsecomp | 128/0 | 5.681 | 2.252 | 3.432 | 22.529 |
| cpu_faiss_w8 | mixed_c64_n256 | local | 128/0 | 5.681 | 0.290 | 0.909 | 22.529 |

Decision:
- `workers=8` is fastest but logged helper child-process deaths/restarts during stress.
- `workers=4` had zero request failures and clean logs in the benchmark, so it is the recommended training helper configuration.

Raw artifacts:
- `privacy_helper_benchmark_20260506_224941.csv/jsonl`
- `privacy_helper_benchmark_20260506_225253.csv/jsonl`
