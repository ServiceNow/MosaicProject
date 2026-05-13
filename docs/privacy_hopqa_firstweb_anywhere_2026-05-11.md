# Privacy HopQA first-web-anywhere run notes, 2026-05-11

This note documents the current first-web-anywhere dataset, launch commands,
and known reward-signal concerns.

## Dataset

The first-web-anywhere dataset selects the first web hop anywhere in each chain,
not only chains whose first hop is web. This keeps the selected hop as a
single-hop training problem with previous-hop placeholders already filled.

```bash
cd /home/toolkit/my_code/MosaicProject

python3 scripts/build_privacy_hopqa_first_web_hop_dataset.py \
  --source-dir /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset \
  --source-basename privacy_hopqa_finalized_2026-05-11 \
  --output-dir /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset \
  --output-basename privacy_hopqa_first_web_hop_anywhere_2026-05-11 \
  --selection first_web_hop_anywhere
```

Current artifact paths:

```text
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix1.jsonl
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix1_train.jsonl
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix1_val.jsonl
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix1_test.jsonl
```

Current training runs should use the question-audited version:

```text
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_train.jsonl
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_val.jsonl
/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_test.jsonl
```

Current split counts are `444 / 75 / 275` for train, val, and test. The test
split holds out MediConn.


## Variantfix1 artifacts

The `variantfix1` dataset version adds the 2026-05-11 evidence-audit answer
variants. It does not change questions, gold answers, document ids, or
supporting spans.

```text
Full dataset: /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix1.jsonl
First-web-anywhere: /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix1.jsonl
Applied log: /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_firstweb_proposals_2026-05-11/applied_answer_variant_proposals_batch1_2026-05-11.jsonl
QA/doc deep dive: /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_firstweb_proposals_2026-05-11/qa_doc_deep_dive_findings_2026-05-11.md
```

## Helper jobs

Launch separate helper jobs for the 4B and 8B runs so helper failures or
overload in one run do not affect the other run.

```bash
cd /home/toolkit/my_code/MosaicProject

ACCOUNT=snow.research.adea \
JOB_SPEC=/home/toolkit/my_code/MosaicProject/privacy_helper_2gpu.yaml \
JOB_NAME=privacy_helper_firstweb_anywhere_0511_4b \
DNS_NAME=phfwany4b0511 \
./run_privacy_helper.sh
```

```bash
cd /home/toolkit/my_code/MosaicProject

ACCOUNT=snow.research.adea \
JOB_SPEC=/home/toolkit/my_code/MosaicProject/privacy_helper_2gpu.yaml \
JOB_NAME=privacy_helper_firstweb_anywhere_0511_8b \
DNS_NAME=phfwany8b0511 \
./run_privacy_helper.sh
```

The helper wrapper prints the internal URL. Use that URL as
`PRIVACY_AGENT_HELPER_URL` for the matching training job.

Known active helper URLs from the 2026-05-11 run:

```text
4B: http://dns-24e3447c-506e-4b21-92df-156e18db5087-phfwany4b0511:8012
8B: http://dns-24e3447c-506e-4b21-92df-156e18db5087-phfwany8b0511:8012
```

## 4B first-web-anywhere training command

The 2026-05-11 large-parent setting keeps parent-document retrieval intact, but
does not read every window from very large retrieved parent documents in one
batch. Parent documents with at most 5 windows are fully read. Larger parents
read the 4 highest query-scoring windows plus one neighboring window on each
side first; the resolver can request more unread windows from the same parent.
This is controlled by:

```text
privacy_hopqa.large_parent_window_policy=top_with_neighbors
privacy_hopqa.read_all_windows_max_count=5
privacy_hopqa.large_parent_top_windows=4
privacy_hopqa.large_parent_neighbor_windows=1
privacy_hopqa.large_parent_max_read_windows=0
privacy_hopqa.reader_window_chars=20000
privacy_hopqa.reader_window_overlap_chars=2000
privacy_hopqa.max_parallel_doc_reads=6
```

Rollouts now log candidate windows, selected parent docs, selected/read window
counts, large-parent limiting, and whether the gold parent was retrieved/read
under `privacy_hopqa` metrics and in `report_metadata.retrieval_summary`.

```bash
cd /home/toolkit/my_code/MosaicProject

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_firstweb_anywhere_0511_qwen3_4b_alltext_${TS}"
FW_ROOT=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_firstweb_anywhere_qwen3_4b \
WANDB_GROUP=firstweb_anywhere_0511_alltext \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
CAPTURE_MODE=all_calls \
DATASET_SPLIT_ROOT="$FW_ROOT" \
FINAL_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl" \
TRAIN_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_train.jsonl" \
VAL_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_val.jsonl" \
TEST_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_test.jsonl" \
PRIVACY_AGENT_HELPER_URL="http://dns-24e3447c-506e-4b21-92df-156e18db5087-phfwany4b0511:8012" \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.reward_mode=all_hops_correct privacy_hopqa.answer_match_mode=exact_or_f1 privacy_hopqa.enable_local_search_bootstrap=false privacy_hopqa.iteration_budget_per_hop=3 privacy_hopqa.helper_timeout_s=180 privacy_hopqa.reader_window_chars=20000 privacy_hopqa.reader_window_overlap_chars=2000 privacy_hopqa.max_windows_per_parent=0 privacy_hopqa.large_parent_window_policy=top_with_neighbors privacy_hopqa.read_all_windows_max_count=5 privacy_hopqa.large_parent_top_windows=4 privacy_hopqa.large_parent_neighbor_windows=1 privacy_hopqa.max_parallel_doc_reads=6 actor.llm_max_rollouts=64" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

## 8B first-web-anywhere training command

```bash
cd /home/toolkit/my_code/MosaicProject

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_firstweb_anywhere_0511_qwen3_8b_alltext_${TS}"
FW_ROOT=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_firstweb_anywhere_qwen3_8b \
WANDB_GROUP=firstweb_anywhere_0511_alltext \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="/mnt/llmd/base_models/Qwen3-8B" \
CAPTURE_MODE=all_calls \
DATASET_SPLIT_ROOT="$FW_ROOT" \
FINAL_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl" \
TRAIN_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_train.jsonl" \
VAL_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_val.jsonl" \
TEST_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_test.jsonl" \
PRIVACY_AGENT_HELPER_URL="http://dns-24e3447c-506e-4b21-92df-156e18db5087-phfwany8b0511:8012" \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.reward_mode=all_hops_correct privacy_hopqa.answer_match_mode=exact_or_f1 privacy_hopqa.enable_local_search_bootstrap=false privacy_hopqa.iteration_budget_per_hop=3 privacy_hopqa.helper_timeout_s=180 privacy_hopqa.reader_window_chars=20000 privacy_hopqa.reader_window_overlap_chars=2000 privacy_hopqa.max_windows_per_parent=0 privacy_hopqa.large_parent_window_policy=top_with_neighbors privacy_hopqa.read_all_windows_max_count=5 privacy_hopqa.large_parent_top_windows=4 privacy_hopqa.large_parent_neighbor_windows=1 privacy_hopqa.max_parallel_doc_reads=6 actor.llm_max_rollouts=64" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

## Gold-evidence answerability audit

Use this audit before relaunching training when reward quality is suspect. It
tests whether `Qwen/Qwen3-4B-Instruct-2507` can answer each first-web-anywhere
question when given the gold document evidence.

```bash
cd /home/toolkit/my_code/MosaicProject

/home/toolkit/bin/micromamba run -p /home/toolkit/.conda/envs/pipeline-rl \
  python scripts/test_privacy_hopqa_firstweb_evidence_modes.py \
    --case-concurrency 8 \
    --concurrency 48 \
    --output-dir runtime_analysis/privacy_hopqa_firstweb_evidence_modes_2026-05-11_full
```

The audit runs two conditions per row:

```text
correct_window: read the 16k-character gold-document window containing the supporting span.
all_windows: read all 16k-character gold-document windows in parallel, then resolve from reader results.
```

Latest 2026-05-11 full-audit results:

```text
total: 794
correct_window_match: 620
any_window_match: 719
resolver_match: 696
correct_window_can_answer: 689
any_window_can_answer: 788
resolver_answered: 779
span_located: 507
```

Interpretation:

```text
correct_window_match < any_window_match means the supporting span/window metadata is often not enough by itself.
any_window_match > resolver_match means the resolver can lose a correct answer even after one reader found it.
any_window_miss rows are the highest-priority QA/doc/variant audit queue.
```

## JSON parse errors

JSON parse errors are actor-model output parsing failures inside the
`privacy_hopqa` harness. They are not helper response JSON errors.

The parser accepts raw JSON, fenced JSON, or the first valid JSON object or
array embedded in text. The common message:

```text
No valid JSON found in response: line 1 column 1 (char 0)
```

means the model response did not contain any valid JSON object or array that
the parser could extract.

Observed on the 2026-05-11 first-web-anywhere runs:

```text
4B sampled traces: 1 parse error in 2000 traces
8B sampled traces: 5130 parse errors in 2000 traces
8B sampled stage counts: doc_read 3005, doc_choose 1909, hop_plan 135, hop_resolve 92
```

Practical impact by stage:

```text
hop_plan parse error: the planner retries with explicit JSON guidance.
doc_choose parse error: the harness falls back to the top retrieved candidates.
doc_read parse error: that document is recorded as not answerable.
hop_resolve parse error: the harness falls back to reader results when possible.
```

The 8B run uses the base model path `/mnt/llmd/base_models/Qwen3-8B`, while the
4B run uses `Qwen/Qwen3-4B-Instruct-2507`. The base 8B model can still run, but
the current traces show it is much less reliable at the structured JSON stages
in this setup.

## Infrastructure errors

Infrastructure errors are helper transport, timeout, or service errors. The
actor treats these as infrastructure failures and retries the rollout instead
of converting the failed call into an ordinary low-reward answer.

They should not directly create unfair negative rewards when the retry
eventually succeeds. They can still matter indirectly:

```text
They reduce rollout throughput.
They can increase policy lag by delaying fresh samples.
They can drop samples if retries are exhausted.
They can confound dashboards if one model's helper is overloaded more often.
```

The relevant distinction is:

```text
LLM JSON parse error: model behavior, can affect reward and training text.
Helper infrastructure error: system failure, should be retried or dropped, not scored.
```

## Local search bootstrap

The upstream PipelineRL merge added a local-search bootstrap in the
`privacy_hopqa` agent. If a hop has not yet tried local search and the model
returns only web searches, the agent prepends one `local_document_search` using
the materialized current-hop question.

This is intended to prevent the full privacy-agent task from ignoring local
company files. For the first-web-anywhere debugging dataset, every selected hop
is a web hop, so this bootstrap is likely a confounder if it is active.

The 2026-05-11 first-web-anywhere jobs were launched before this merge in the
running Python processes. Relaunches after the merge should either disable this
bootstrap for the first-web-only experiment or explicitly document that the run
includes forced local searches.

## Reward-signal risks to audit

The highest-priority unfair reward checks are:

```text
Unfair negative: model found the right supporting document but answer variants missed its valid answer form.
Unfair negative: model found a plausible answer because the QA pair is underspecified, but the intended document requires extra hidden context.
Unfair negative: document chunking hid the answer although the parent document supports it.
Unfair negative: infrastructure failure exhausted retries and dropped otherwise-good samples.
Unfair positive: answer string matches but the model used the wrong source document.
Unfair positive: permissive matching accepts a partial or over-broad answer.
```

The PipelineRL branch now exposes an explicit answer-matching mode. For the
current default, use:

```text
privacy_hopqa.answer_match_mode=exact_or_f1
```

In `exact_or_f1` mode, the predicted answer is accepted if it either exactly
normalizes to one of the explicit accepted variants or reaches the configured
token-F1 threshold against one of those variants.

The stricter diagnostic option is:

```text
privacy_hopqa.answer_match_mode=accepted_exact
```

In `accepted_exact` mode, token F1 is still logged as `match_score`, but it does
not turn a mismatch into reward.

The deliberately permissive diagnostic option is:

```text
privacy_hopqa.answer_match_mode=f1_or_substring
```

That mode accepts substring containment in either direction or token F1 above
the threshold. It can reduce false negatives from formatting, but it can also
create unfair positives.

The local-search bootstrap is also explicit:

```text
privacy_hopqa.enable_local_search_bootstrap=false
```

Keep it disabled for first-web-only debugging runs. Enable it only when a run is
intended to force at least one local search before trusting web-only retrieval.

## Questionfix1 update, 2026-05-11

Use the `variantfix2_questionfix2` artifacts for the next first-web training/debug runs.

Final full-chain dataset:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.jsonl`

Final full-chain splits:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2_{train,val,test}.jsonl`

First-web-anywhere dataset:

`data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl`

First-web-anywhere splits:

`data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_{train,val,test}.jsonl`

Questionfix report:

`runtime_analysis/privacy_hopqa_firstweb_proposals_2026-05-11/questionfix1_report_2026-05-11.md`

Validation summary:

- Suspicious-row evidence audit: `9/9` resolver matches on the patched first-web rows.
- Correct-window reader matches: `8/9`; the remaining row is `4538351b`, where the reader justification contains the correct sentence but emits the wrong short answer, while the resolver returns `Belgium`.
- Structural check: full dataset `1001` rows; first-web-anywhere dataset `794` rows; no malformed materialized questions found.

## Launched questionfix1 4B run, 2026-05-11

Helper job:

- EAI job id: `1f0b1699-e765-47a4-b54b-39b38b6acd1d`
- Name: `privacy_helper_firstweb_qfix1_0511_4b`
- Internal URL: `http://dns-24e3447c-506e-4b21-92df-156e18db5087-phfwqfix14b0511:8012`
- Health checked before training launch.

Training job:

- EAI job id: `d7f202af-ccc8-463d-947b-5ac3030ecfc4`
- Replica group: `f1961032-8669-4639-b806-e97152f5f219`
- Output dir: `/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_firstweb_qfix1_0511_qwen3_4b_alltext_20260511145746`
- W&B project: `privacy_hopqa_firstweb_qfix1_qwen3_4b`
- W&B group: `firstweb_qfix1_0511_alltext`
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Replicas: `2`
- Capture mode: `all_calls`
- Dataset: `privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_{train,val,test}.jsonl`
- Key overrides: `privacy_hopqa.answer_match_mode=exact_or_f1`, `privacy_hopqa.enable_local_search_bootstrap=false`, `privacy_hopqa.helper_timeout_s=180`, `actor.llm_max_rollouts=64`.

## Full-parent-window rerun, 2026-05-11

The earlier `questionfix1` 4B run was stopped because the live harness still capped retrieved parent documents to four evidence windows and capped BrowseComp text at the helper/request layer. That could create unfair negatives when the right parent document was retrieved but the answer lived in a dropped or truncated window.

Code/config changes:

- `privacy_hopqa.max_windows_per_parent=0` now means no cap on evidence windows from a retrieved parent document.
- `privacy_hopqa.retrieval_source_chars=0` now requests unbounded BrowseComp hit text from the helper.
- `max_candidate_cards` is now interpreted as max parent documents. All windows from retained parents remain available for selection/reading.
- `privacy_helper_2gpu.yaml` now starts the helper with `--browsecomp-max-chars 0`.
- BrowseComp/helper truncation now treats `max_chars <= 0` as unbounded.

Stopped run:

- Training job id: `d7f202af-ccc8-463d-947b-5ac3030ecfc4`
- Reason: obsolete bounded-window/truncated-doc harness.

New helper:

- EAI job id: `b615edee-21c0-407a-9691-f825a401c9b9`
- Name: `privacy_helper_firstweb_qfix1_fullchunks_0511_4b`
- Internal URL: `http://dns-24e3447c-506e-4b21-92df-156e18db5087-phfwqfixfull4b0511:8012`
- Launch includes `--browsecomp-max-chars 0`.

New 4B training job:

- EAI job id: `8b7b7dce-d7d8-46e5-9a27-e12e53f6caf8`
- Replica group: `866887b7-a4f0-4a2d-b396-61800a2f0a98`
- Output dir: `/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_firstweb_qfix1_fullchunks_0511_qwen3_4b_alltext_20260511152005`
- W&B project: `privacy_hopqa_firstweb_qfix1_fullchunks_qwen3_4b`
- W&B group: `firstweb_qfix1_fullchunks_0511_alltext`
- Key overrides: `privacy_hopqa.retrieval_source_chars=0`, `privacy_hopqa.max_windows_per_parent=0`, `privacy_hopqa.max_candidate_cards=20`, `privacy_hopqa.answer_match_mode=exact_or_f1`, `privacy_hopqa.enable_local_search_bootstrap=false`, `actor.llm_max_rollouts=64`, `CAPTURE_MODE=all_calls`.

## Full-parent-window resolver fix, 2026-05-11

The first full-parent-window 4B run failed before optimizer steps because the resolver prompt re-concatenated every full document-read window. Several `hop_resolve` prompts reached 21k-24k tokens, above the 16,384-token Qwen context limit, causing vLLM 400 errors and then a server disconnect.

The retrieval change itself was correct: full BrowseComp text is requested, retained parent documents expose all evidence windows, and `max_candidate_cards` limits parent documents rather than flattened chunks. The additional fix is resolver-specific:

- Document-reader prompts still receive the selected evidence window.
- Resolver prompts now receive compact reader outputs: doc id, source/title/locator, `can_answer`, proposed answer, confidence, justification, and missing information.
- Resolver prompts receive compact unread candidate cards for `read_more`, not full unread document contents.
- This avoids context overflow without truncating the evidence window shown to the document-reader.

Failed run:

- Training job id: `8b7b7dce-d7d8-46e5-9a27-e12e53f6caf8`
- Failure: `hop_resolve` context overflow, then actor stopped on `PrivacyHopQALLMInfrastructureError`.

New resolver-compact 4B training job:

- EAI job id: `efd366bc-ea69-48cc-aca6-292f315b2220`
- Replica group: `952fc24d-9273-400c-ac5d-5022df219a2b`
- W&B project: `privacy_hopqa_firstweb_qfix1_fullchunks_resolvercompact_qwen3_4b`
- W&B group: `firstweb_qfix1_fullchunks_resolvercompact_0511_alltext`
- Dataset: `privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2_{train,val,test}.jsonl`
- Key overrides: unchanged from the full-parent-window rerun.

## Full-parent-window compact-card fix, 2026-05-11

The resolver-compact run reached rollouts, but some document-chooser prompts still overflowed because unbounded parent windows were represented with fixed 500-character previews. Examples:

- `doc_choose` prompt tokens 38,338 for `b1f740fc_first_web_hop`
- `doc_choose` prompt tokens 31,983 for `1f7e9978_first_web_hop`
- `doc_choose` prompt tokens 25,667 for `aa63bec9_first_web_hop`

Additional code change:

- Candidate pools remain unbounded by chunk within retained parent documents.
- Candidate cards shown to `doc_choose` and `hop_resolve` are now dynamically compacted based on candidate count.
- The compact card keeps IDs, parent/window metadata, score/rank, title/locator, short top queries, and a short snippet.
- The document-reader still receives the selected evidence window; compact cards are previews only.

Failed run:

- Training job id: `efd366bc-ea69-48cc-aca6-292f315b2220`
- Failure: `doc_choose` context overflow from too many fixed-size candidate previews, then actor stopped on LLM server disconnect.

New compact-card 4B training job:

- EAI job id: `a43c9526-4242-404a-a18f-462a24df6fb5`
- Replica group: `d88c0075-2f54-4ad2-a643-626693e6f991`
- W&B project: `privacy_hopqa_firstweb_qfix1_fullchunks_compactcards_qwen3_4b`
- W&B group: `firstweb_qfix1_fullchunks_compactcards_0511_alltext`
- Dataset and key overrides: unchanged from the full-parent-window rerun.

## Compact-history/context-guard fix, 2026-05-11

The compact-card run reached several dozen rollouts, but still produced a small number of oversized prompts:

- `doc_choose` prompt tokens up to 16,893
- `hop_resolve` prompt tokens up to 17,896
- Later `hop_plan` calls inherited bulky reader history from prior reads

Those oversized prompts caused vLLM 400s and eventually a server disconnect.

Additional code changes:

- Candidate-card snippets were compacted more aggressively while preserving all candidate IDs and parent/window metadata.
- Reader-result justifications included in resolver/planner prompts were shortened; the document-reader still sees the full selected evidence window.
- Hop planning now sees compact reader-history summaries instead of raw stored reader results.
- `PrivacyHopQALLMAdapter` now preflights context size and raises a local `PrivacyHopQAContextLengthError` when a prompt leaves no room for output, instead of sending a guaranteed-invalid request to vLLM.

Failed run:

- Training job id: `a43c9526-4242-404a-a18f-462a24df6fb5`
- Failure: residual oversized chooser/resolver prompts led to vLLM BadRequests and one shard disconnect.

New context-guard 4B training job:

- EAI job id: `4d57ab5a-d594-49d5-aa45-1ae632243038`
- Replica group: `5e2d4af0-fbe5-4d3d-864f-bcd0ddeb0a49`
- W&B project: `privacy_hopqa_firstweb_qfix1_fullchunks_contextguard_qwen3_4b`
- W&B group: `firstweb_qfix1_fullchunks_contextguard_0511_alltext`
- Dataset and key overrides: unchanged from the full-parent-window rerun.

### 2026-05-11 full-parent selection fix

The earlier `fullchunks`/`contextguard` attempts still let prompt-side limits act on individual evidence windows. That was not the intended semantics: once retrieval finds a parent document, `max_candidate_cards` should limit how many parent documents are considered, not how many windows/chunks from those parents are available.

Code changes in `PipelineRL/pipelinerl/domains/privacy_hopqa/agent.py` and `prompts.py`:

- `max_candidate_cards` is now interpreted as a parent-document cap in retrieval and prompting.
- Candidate cards shown to the document chooser and resolver are parent-document cards with `matched_window_count`, `total_window_count`, and a representative snippet.
- If the model selects a parent document, the harness expands that parent to every candidate evidence window for reading. We no longer slice selected chunks down to `choose_top_k`.
- Reader inputs no longer apply a second truncation to the candidate text; each selected evidence window is passed through as stored.
- `max_windows_per_parent=0` remains required for this experiment, so the helper/harness does not prune windows from a retrieved parent document.

Killed flawed run with oversized context behavior:

- `4d57ab5a-d594-49d5-aa45-1ae632243038` (`privacy_hopqa_firstweb_qfix1_fullchunks_contextguard_0511_qwen3_4b_alltext_20260511154747`)

Replacement run:

- Job: `894db861-ad80-418d-a4ed-35411be8f667`
- W&B project: `privacy_hopqa_firstweb_qfix1_fullparents_qwen3_4b`
- W&B group: `firstweb_qfix1_fullparents_0511_alltext`
- Dataset: `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl`
- Key overrides: `privacy_hopqa.retrieval_source_chars=0 privacy_hopqa.max_windows_per_parent=0 privacy_hopqa.max_candidate_cards=20 actor.llm_max_rollouts=64 privacy_hopqa.reward_mode=all_hops_correct privacy_hopqa.answer_match_mode=exact_or_f1 privacy_hopqa.enable_local_search_bootstrap=false`

Initial monitoring for replacement run `894db861-ad80-418d-a4ed-35411be8f667`:

- Job reached running state and started rollouts normally.
- First rollout monitoring showed no `doc_choose` or `hop_resolve` context-cap/skipped-generation warnings.
- Helper saw burst pressure during the heaviest phase, with transient `PrivacyHopQAHelperInfrastructureError, retry 1` warnings from the actor, but no observed retry-2/permanent failures in the monitored window.
- Helper health stayed `ok=true`; browse endpoint errors stayed at 0.
- Finetune completed step 1 at 2026-05-11 16:20:17 UTC with 1024 samples and `rl/reward=0.719`.

## 2026-05-11 gold-read accuracy fixes

- Patched `07df48bf` / `07df48bf_first_web_hop`: the Facebook MAU question now asks for "one decimal place" and uses canonical answer `3.1 billion`; `3.1`, `3.065 billion`, `3.065`, `3,065,000,000`, and `3065000000` are accepted variants.
- Synced full-chain top-level `global_answer_variants` / `global_alternate_valid_answers` from the final hop variants when `global_answer` equals the final hop answer. This prevents final-answer audits/viewers from missing variants that the hop-level reward accepts.
- Added safe alias variants found by gold-read audit: `AWS` <-> `Amazon Web Services`, `LSERS` <-> `Louisiana School Employees' Retirement System`, and `two years` <-> `2 years` / `2` / `two`.
- Added PipelineRL metrics for the key health check: `gold_parent_read_answer_accuracy`, `gold_parent_read_but_incorrect_hops`, `gold_parent_read_but_no_answer_hops`, `gold_reader_correct_hops`, and `gold_reader_correct_but_final_incorrect_hops`.
- Fixed a harness issue where a hop could read useful documents, then stall on a later no-plan iteration and finish with no answer. Prior reader results are now carried into later resolver calls, and no-plan iterations attempt to resolve from already-read evidence before stalling.
- Added generic table/list extraction guidance to the document reader and resolver prompts to reduce adjacent-cell mistakes, e.g. selecting production when the question asks for deliveries.

Offline sample from the live 20k-window run before these fixes: among recent traces where the gold parent was read, final answer accuracy was about 85.5%; the gold reader had already produced a reward-correct answer in about 93.7%. The gap is mostly resolver loss, no-plan-after-read, and answer-variant issues, so these metrics should be monitored on the next run.

## 2026-05-11 variantfix2/questionfix2

Current recommended dataset version:

```text
privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.jsonl
privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl
```

Changes from `variantfix1_questionfix1`:

```text
question rewrites: 14 hops
answer-variant additions: 5 hops
rows dropped: 0
```

The question rewrites were applied only when the gold document supported the
same answer but the old wording made adjacent table values, answer leakage, or
wrong quantity selection likely. Full-chain `question_templated` fields preserve
bridge placeholders. First-web rows remain standalone and contain filled-in
prior-hop answers.

Patch metadata:

```text
data_files/privacy_hopqa_final_dataset/proposed_question_changes_2026-05-11_questionfix2.jsonl
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.applied_log.jsonl
data_files/privacy_hopqa_final_dataset/manifest_privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.json
data_files/privacy_hopqa_first_web_hop_dataset/manifest_privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.json
```

### Full-hop gold-document answerability audit

The diagnostic script now supports full multi-hop datasets as well as the
first-web extraction. To audit every hop in the full dataset against its gold
document evidence:

```bash
cd /home/toolkit/my_code/MosaicProject

/home/toolkit/bin/micromamba run -p /home/toolkit/.conda/envs/pipeline-rl \
  python scripts/test_privacy_hopqa_firstweb_evidence_modes.py \
    --dataset data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.jsonl \
    --output-dir runtime_analysis/privacy_hopqa_full_gold_evidence_modes_2026-05-11_variantfix2_questionfix2 \
    --case-concurrency 8 \
    --concurrency 48
```

For a focused audit of changed rows, pass one or more IDs like:

```bash
--chain-id ac415bbb::hop3 --chain-id c4742644::hop1
```

This test is the systematic version of the manual check: if the model cannot
answer a hop when shown the gold document evidence, that row is a candidate for
question rewrite, answer-variant expansion, or document-assignment review.

### Question-template synchronization

After producing `variantfix2_questionfix2`, an integrity check found inherited
full-chain mismatches where `hop.question` contained a rounding qualifier but
`hop.question_templated` and therefore `numbered_questions` did not. These were
metadata inconsistencies: the gold answers and documents were unchanged, but the
agent prompt context was weaker than the per-hop question.

The version was updated in place with:

```text
full-chain realization mismatches before sync: 135
full-chain realization mismatches after sync: 0
first-web placeholder questions after sync: 0
```

Most fixes mechanically copied existing rounding qualifiers into the templated
question. One inherited wrong-placeholder row was corrected explicitly:
`16213b4e::hop4` now uses `(3)` for the year `2020` and `(2)` for `Basel III`.

### Shared document-reader audit refactor

`test_privacy_hopqa_firstweb_evidence_modes.py` no longer contains duplicated
`.docx/.pptx/.xlsx` parsing. It uses the existing StepFun answerability
extractor and, for local documents, first tries PipelineRL local-index
`documents.json` content when available.

Latest full-hop audit output:

```text
runtime_analysis/privacy_hopqa_full_gold_evidence_modes_2026-05-11_variantfix2_questionfix2_run3_shared_doc_reader/
```

Triage counts:

```text
ok: 3110
reader_answered_but_no_variant_match: 194
resolver_lost_reader_correct_answer: 43
no_gold_window_answer: 38
correct_window_answered_but_no_match: 18
```
