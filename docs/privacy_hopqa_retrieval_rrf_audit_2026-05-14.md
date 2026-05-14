# Privacy HopQA retrieval RRF audit, 2026-05-14

This note documents the retrieval-only audit and local Qwen replay used to decide whether to test reciprocal-rank fusion in the Privacy HopQA helper.

## What changed

The helper now supports explicit rank modes:

```bash
--browsecomp-rank-mode {hybrid,rrf,bm25}
--browsecomp-rrf-k 20
--local-rank-mode {dense,rrf,bm25}
--local-bm25-recall-k 100
--local-dense-recall-k 100
--local-rrf-k 20
```

The default helper behavior is unchanged unless these flags are passed. Use `privacy_helper_2gpu_rrf.yaml` to launch the RRF variant.
Use `privacy_helper_2gpu_bm25.yaml` to launch the BM25-only variant.

RRF score:

```text
score(doc) = dense_weight / (rrf_k + dense_rank(doc)) + lexical_weight / (rrf_k + lexical_rank(doc))
```

The implemented live mode uses equal weights and `rrf_k=20`.

## Current training-side document read default

PipelineRL and the runbook now use:

```text
privacy_hopqa.max_parallel_doc_reads=5
```

This limits concurrent document-reader calls. It does not change `choose_top_k=3`, `retrieval_top_k=5`, or `max_parallel_retrieval_actions=3`.

## Larger retrieval-only audit

Command:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/audit_privacy_hopqa_retrieval_methods.py \
  --sample-size 240 \
  --max-traces 5000 \
  --dense-k 50 \
  --max-model-queries-per-hop 20 \
  --workers 24 \
  --out-dir runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14_large240
```

Inputs:

```text
run_root=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122
dataset=data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl
sample_size=240
local_hops=122
web_hops=118
```

The audit uses actual model search queries from traces. `current` is the existing helper ranking for that source, `lexical` is BM25-style lexical retrieval against the same corpus, and `fusion` is equal-weight RRF.

### Overall actual-query gold parent rank

| method | top1 | top5 | top10 | top20 |
| --- | ---: | ---: | ---: | ---: |
| current | 18.8% | 48.3% | 68.8% | 79.2% |
| lexical | 75.4% | 84.6% | 88.3% | 90.0% |
| fusion | 64.2% | 85.8% | 90.8% | 94.2% |

### By source

| source | method | top5 | top10 | top20 |
| --- | --- | ---: | ---: | ---: |
| local | current | 46.7% | 65.6% | 82.0% |
| local | lexical | 89.3% | 91.8% | 91.8% |
| local | fusion | 83.6% | 89.3% | 91.8% |
| web | current | 50.0% | 72.0% | 76.3% |
| web | lexical | 79.7% | 84.7% | 88.1% |
| web | fusion | 88.1% | 92.4% | 96.6% |

### By source and difficulty

Values are top5/top10/top20.

| source | difficulty | current | lexical | fusion |
| --- | --- | ---: | ---: | ---: |
| local | easy | 81.2 / 96.9 / 100.0 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 |
| local | medium | 53.3 / 76.7 / 86.7 | 100.0 / 100.0 / 100.0 | 93.3 / 93.3 / 100.0 |
| local | hard-answer | 33.3 / 56.7 / 76.7 | 83.3 / 90.0 / 90.0 | 76.7 / 86.7 / 90.0 |
| local | hard-retrieval | 16.7 / 30.0 / 63.3 | 73.3 / 76.7 / 76.7 | 63.3 / 76.7 / 76.7 |
| web | easy | 86.7 / 93.3 / 93.3 | 73.3 / 73.3 / 76.7 | 93.3 / 96.7 / 100.0 |
| web | medium | 73.3 / 90.0 / 93.3 | 76.7 / 83.3 / 83.3 | 93.3 / 96.7 / 100.0 |
| web | hard-answer | 35.7 / 53.6 / 60.7 | 85.7 / 96.4 / 100.0 | 85.7 / 92.9 / 96.4 |
| web | hard-retrieval | 3.3 / 50.0 / 56.7 | 83.3 / 86.7 / 93.3 | 80.0 / 83.3 / 90.0 |

Interpretation:

- RRF is clearly better than the current web ranking on this sample.
- Lexical-only is very strong for local retrieval, but RRF is safer as a single helper mode because it preserves dense signal and is much better than current on web.
- The biggest current bottleneck is hard-retrieval, especially web hard-retrieval.

Viewer:

```text
runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14_large240/index.html
```

## Local Qwen replay

This replay asks local `Qwen/Qwen3-4B-Instruct-2507` to answer with the production doc-reader and resolver prompt builders using the top-5 documents from current ranking versus RRF ranking. It uses observed model queries and gold previous-hop answers to isolate retrieval/reader effects.

Start the local model:

```bash
source ~/initmamba.sh
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 32768
```

Run the replay:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/replay_privacy_hopqa_rrf_retrieval_qwen.py \
  --per-bucket 5 \
  --out-dir runtime_analysis/privacy_hopqa_rrf_replay_qwen4b_2026-05-14_large40
```

Results:

| bucket | n | current match | RRF match | current gold top5 | RRF gold top5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local easy | 5 | 5 | 5 | 5 | 5 |
| local medium | 5 | 5 | 3 | 5 | 5 |
| local hard-answer | 5 | 3 | 4 | 4 | 5 |
| local hard-retrieval | 5 | 0 | 3 | 2 | 5 |
| web easy | 5 | 5 | 5 | 5 | 5 |
| web medium | 5 | 3 | 3 | 5 | 5 |
| web hard-answer | 5 | 2 | 2 | 5 | 5 |
| web hard-retrieval | 5 | 0 | 4 | 1 | 5 |

Interpretation:

- RRF substantially helps when the current ranking does not surface the gold parent in the top 5.
- RRF is not a strict row-by-row dominance result; local medium had two cases where current answered and RRF did not despite both surfacing the gold parent. That points to reader/resolver variance and candidate-set composition, not just retrieval rank.
- The replay is a diagnostic, not a replacement for a live helper experiment.

Viewer:

```text
runtime_analysis/privacy_hopqa_rrf_replay_qwen4b_2026-05-14_large40/index.html
```

## Parallel-query chooser replay, current vs RRF

The direct top-5 replay above is useful but too optimistic about how evidence reaches the answer. The more realistic replay below uses the production chooser, reader, and resolver prompts:

- take up to three actual observed model queries for the hop, matching `max_parallel_retrieval_actions=3`
- retrieve top 5 per query under current ranking versus RRF ranking
- merge duplicate parent documents into one candidate-card pool
- add opposite-source distractor cards from a real trace when available, so local hops can include web distractors and web hops can include local distractors
- cap the chooser pool at 20 parent documents
- ask Qwen to select up to 3 parent documents
- read selected documents with `max_parallel_doc_reads=5`
- resolve the hop and score with both exact accepted-variant match and the current PipelineRL reward match, `exact_or_f1>=0.75`

This is still an isolated-hop counterfactual. Previous-hop answers are gold answers from the dataset, so the measurement asks: if the model made similar search queries, how would the ranking mode affect whether the current hop can be answered?

Command:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/replay_privacy_hopqa_parallel_rrf_qwen.py \
  --per-bucket 12 \
  --workers 8 \
  --out-dir runtime_analysis/privacy_hopqa_parallel_rrf_replay_qwen4b_2026-05-14_large96_mixed
```

Settings:

```text
model=Qwen/Qwen3-4B-Instruct-2507
n_cases=96
queries_per_hop=3
retrieval_top_k_per_query=5
max_candidate_cards=20
choose_top_k=3
max_parallel_doc_reads=5
rrf_k=20
include_trace_distractors=true
```

### Parallel replay results

| bucket | n | current gold in candidates | RRF gold in candidates | current gold selected | RRF gold selected | current reward match | RRF reward match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 96 | 41 | 81 | 34 | 61 | 26 | 54 |
| local | 48 | 18 | 41 | 14 | 33 | 11 | 31 |
| web | 48 | 23 | 40 | 20 | 28 | 15 | 23 |
| easy | 24 | 21 | 24 | 18 | 19 | 17 | 19 |
| medium | 24 | 12 | 22 | 9 | 19 | 6 | 17 |
| hard-answer | 24 | 5 | 17 | 5 | 9 | 3 | 7 |
| hard-retrieval | 24 | 3 | 18 | 2 | 14 | 0 | 11 |
| local easy | 12 | 10 | 12 | 8 | 10 | 8 | 11 |
| local medium | 12 | 4 | 11 | 3 | 11 | 2 | 10 |
| local hard-answer | 12 | 2 | 9 | 2 | 5 | 1 | 5 |
| local hard-retrieval | 12 | 2 | 9 | 1 | 7 | 0 | 5 |
| web easy | 12 | 11 | 12 | 10 | 9 | 9 | 8 |
| web medium | 12 | 8 | 11 | 6 | 8 | 4 | 7 |
| web hard-answer | 12 | 3 | 8 | 3 | 4 | 2 | 2 |
| web hard-retrieval | 12 | 1 | 9 | 1 | 7 | 0 | 6 |

Case-level deltas:

| outcome | count |
| --- | ---: |
| RRF correct, current wrong | 31 |
| current correct, RRF wrong | 3 |
| both correct | 23 |
| neither correct | 39 |

Interpretation:

- RRF has a large positive expected effect under the realistic chooser/reader path: reward matches improve from 26/96 to 54/96.
- The gain is mostly retrieval-side. Gold parent documents reach the chooser pool in 81/96 RRF cases versus 41/96 current cases.
- The chooser can use the extra signal: gold-selected cases rise from 34/96 to 61/96 even with opposite-source distractors present.
- The largest gains are in medium and hard-retrieval buckets. Hard-retrieval goes from 0/24 current reward matches to 11/24 with RRF.
- Easy web is the only small regression slice in this sample, 9/12 current versus 8/12 RRF. The net case-level tradeoff is still strongly positive: 31 improvements versus 3 regressions.
- There are still 39/96 cases where neither mode answers correctly. Those are now better targets for question quality, reader behavior, or query-generation work rather than pure ranking.

Viewer:

```text
runtime_analysis/privacy_hopqa_parallel_rrf_replay_qwen4b_2026-05-14_large96_mixed/index.html
```

## Parallel-query chooser replay, current vs BM25-only vs RRF

After the RRF replay, I reran the same realistic setup with BM25-only included as a third mode. BM25-only means no dense reranking in the counterfactual candidate pool: for each actual model query, the top lexical/BM25 parent documents are merged and passed to the same production chooser, reader, and resolver prompts.

Command:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/replay_privacy_hopqa_parallel_rrf_qwen.py \
  --per-bucket 12 \
  --workers 8 \
  --out-dir runtime_analysis/privacy_hopqa_parallel_bm25_rrf_replay_qwen4b_2026-05-14_large96_mixed
```

Settings are the same as the RRF replay:

```text
model=Qwen/Qwen3-4B-Instruct-2507
n_cases=96
queries_per_hop=3
retrieval_top_k_per_query=5
max_candidate_cards=20
choose_top_k=3
max_parallel_doc_reads=5
include_trace_distractors=true
```

### BM25-inclusive replay results

| bucket | n | current gold in candidates | BM25 gold in candidates | RRF gold in candidates | current gold selected | BM25 gold selected | RRF gold selected | current reward match | BM25 reward match | RRF reward match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 96 | 41 | 76 | 81 | 33 | 60 | 62 | 25 | 54 | 54 |
| local | 48 | 18 | 43 | 41 | 14 | 32 | 33 | 11 | 30 | 30 |
| web | 48 | 23 | 33 | 40 | 19 | 28 | 29 | 14 | 24 | 24 |
| easy | 24 | 21 | 20 | 24 | 18 | 17 | 19 | 17 | 17 | 19 |
| medium | 24 | 12 | 19 | 22 | 9 | 17 | 19 | 6 | 15 | 16 |
| hard-answer | 24 | 5 | 18 | 17 | 4 | 11 | 10 | 2 | 10 | 8 |
| hard-retrieval | 24 | 3 | 19 | 18 | 2 | 15 | 14 | 0 | 12 | 11 |
| local easy | 12 | 10 | 12 | 12 | 8 | 10 | 10 | 8 | 11 | 11 |
| local medium | 12 | 4 | 12 | 11 | 3 | 10 | 10 | 2 | 9 | 8 |
| local hard-answer | 12 | 2 | 9 | 9 | 2 | 5 | 6 | 1 | 5 | 6 |
| local hard-retrieval | 12 | 2 | 10 | 9 | 1 | 7 | 7 | 0 | 5 | 5 |
| web easy | 12 | 11 | 8 | 12 | 10 | 7 | 9 | 9 | 6 | 8 |
| web medium | 12 | 8 | 7 | 11 | 6 | 7 | 9 | 4 | 6 | 8 |
| web hard-answer | 12 | 3 | 9 | 8 | 2 | 6 | 4 | 1 | 5 | 2 |
| web hard-retrieval | 12 | 1 | 9 | 9 | 1 | 8 | 7 | 0 | 7 | 6 |

Case-level deltas versus current:

| mode | improves | regresses | both correct | neither correct |
| --- | ---: | ---: | ---: | ---: |
| BM25-only | 35 | 6 | 19 | 36 |
| RRF | 33 | 4 | 21 | 38 |

BM25-only versus RRF:

| relationship | count |
| --- | ---: |
| RRF correct, BM25 wrong | 9 |
| BM25 correct, RRF wrong | 9 |
| both correct | 45 |
| neither correct | 33 |

Interpretation:

- BM25-only is a serious baseline. In this replay it ties RRF on final reward matches, 54/96.
- RRF still has slightly better gold-parent coverage, 81/96 versus 76/96, and slightly better gold-selected coverage, 62/96 versus 60/96.
- BM25-only is stronger in this sample on hard-answer and hard-retrieval reward counts, especially web hard-answer and web hard-retrieval.
- BM25-only has a clear web-easy regression: 6/12 versus current 9/12 and RRF 8/12. This is the main reason I would not switch directly to BM25-only without a live A/B run.
- RRF is the safer default retrieval experiment because it preserves dense signal while capturing most BM25 gains. BM25-only is worth launching as a parallel ablation because it may be better for the genuinely hard retrieval failures.

Viewer:

```text
runtime_analysis/privacy_hopqa_parallel_bm25_rrf_replay_qwen4b_2026-05-14_large96_mixed/index.html
```

## Privacy-aware paraphrased web-query audit

I also audited the specific behavior where the model uses web queries that look privacy-aware: generalized/public/industry phrasing, while omitting the company name and the exact accepted answer value. This covers cases like:

```text
retail grocery chain disaster recovery investment food safety inventory system
```

instead of directly searching:

```text
Lee's Market disaster recovery investment approximately 250 high-risk food products
```

Command:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/audit_privacy_aware_web_queries_bm25_dense.py \
  --max-traces 2500 \
  --max-queries 80 \
  --max-per-hop 3 \
  --out-dir runtime_analysis/privacy_hopqa_privacy_aware_web_query_bm25_dense_2026-05-14_balanced80
```

Selection rule:

- actual `web_search` calls from traces
- query/action text contains general/public/industry-style phrasing
- query omits the company name
- query omits the exact accepted answer value
- at most 3 queries per hop, to avoid one chain dominating the analysis

Retrieval comparison:

- `current`: live helper `/browsecomp/search`, i.e. the current web retrieval path
- `BM25`: local lexical BM25 over `/home/toolkit/full_docs` plus the curated web chunks used by the helper

Results:

| group | n | mean top5 overlap | different top1 | current web-gold top5 | BM25 web-gold top5 | current web-gold top20 | BM25 web-gold top20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 74 | 0.085 | 65 | 16/23 | 19/23 | 18/23 | 19/23 |
| local-hop privacy paraphrases | 51 | 0.068 | 51 | n/a | n/a | n/a | n/a |
| web-hop privacy paraphrases | 23 | 0.122 | 14 | 16/23 | 19/23 | 18/23 | 19/23 |

Interpretation:

- BM25 and current helper results are usually very different for these paraphrased privacy-aware web queries.
- For web hops where a gold web document exists, BM25 has higher gold-document recall: 19/23 top5 versus 16/23 for current.
- For local hops, there is no web gold document by design. The important signal is that top5 overlap is only 0.068 and top1 differs in 51/51 cases, so ranking mode heavily changes what public context a privacy-aware agent sees.
- These queries are often lexical entity/title searches after removing private details. Dense/hybrid retrieval can drift to semantically related curated documents, while BM25 tends to preserve exact rare names and public-program terms.

Concrete web-hop example where BM25 wins:

```text
Question: In what year did Eepi Chaad complete a residency with the Contemporary Art Museum Houston?
Privacy-aware query: Eepi Chaad Contemporary Art Museum Houston residency program timeline
Gold web doc: web/48175
Current helper rank: not in top 20
BM25 rank: 1
```

Concrete local-hop privacy-paraphrase example:

```text
Question: What was the average order value increase at Lee's Market due to personalized product recommendations?
Privacy-aware query: case study grocery chain personalized product recommendations average order value increase
Current top result: BCG personalization article from curated DRBench URLs
BM25 top result: web/94142 from BrowseComp full docs
```

Conclusion:

- If we expect a more privacy-aware planner to use generalized web queries, BM25 becomes more important, not less.
- This strengthens the case for either BM25-only as a live ablation or an RRF/BM25-heavy web ranker.
- A pure dense-reranked web path is risky for paraphrased privacy-aware queries because it can miss exact public entities even when the query contains their names.

Viewer:

```text
runtime_analysis/privacy_hopqa_privacy_aware_web_query_bm25_dense_2026-05-14_balanced80/index.html
```

## Recommended live experiment

Launch one RRF helper:

```bash
cd /home/toolkit/my_code/MosaicProject
ACCOUNT=snow.research.adea JOB_SPEC=privacy_helper_2gpu_rrf.yaml ./run_privacy_helper.sh
```

Launch a BM25-only helper for ablation:

```bash
cd /home/toolkit/my_code/MosaicProject
ACCOUNT=snow.research.adea JOB_SPEC=privacy_helper_2gpu_bm25.yaml ./run_privacy_helper.sh
```

Run a comparable 4B planning-only trainer against that helper:

```bash
cd /home/toolkit/my_code/MosaicProject
export PRIVACY_AGENT_HELPER_URL="http://<rrf-helper-internal-dns>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_rrf_prefix_step_${TS}"

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_q30_retrieval_rrf \
WANDB_GROUP=q30_full_rrf \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=5 actor.llm_max_rollouts=64 privacy_hopqa.training_reward_mode=prefix_progress privacy_hopqa.stop_after_incorrect_hop=true finetune.rl.step_reward_advantages=true finetune.rl.pad_step_rewards_for_advantage=true" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

Monitor:

- gold parent retrieved/read rates
- reward by pattern and chain length
- parse errors
- helper timeout/error rate
- whether hard-retrieval chains start receiving non-zero advantage

## Query privacy levels and consistent local/web RRF default

The earlier phrase `privacy-aware query` was too vague. In this audit, query privacy and retrieval ranking are separated:

- Query privacy describes what the planner put in the search string.
- Retrieval rank mode describes how the helper orders candidate documents after receiving that search string.

### Query privacy levels

| level | meaning | desired use |
| --- | --- | --- |
| P0 raw or bridge leaky | Query includes the company name, accepted answer, or previous-hop answer. | Useful for diagnosing current behavior, not privacy preserving. |
| P1 private details remain | Company is omitted, but private numeric/date/entity details or internal operational terms remain. | Less raw than P0, still leaky. |
| P2 private-topic paraphrase | Company and exact answers are omitted, but the query still targets an internal company topic. | A privacy-improved but still sensitive planning behavior. |
| P3 public-targeted analogue | Query asks for a general/public analogue without direct company or answer leakage. | A reasonable privacy-aware planner target for local-hop uncertainty. |
| P4 independent public hop | Web-hop query aimed at external public evidence without obvious company, answer, or internal-topic leakage. | Desired behavior for genuine web hops. |

These labels are heuristic and intentionally conservative. Generic public words that also appear in internal contexts are not enough to label a query private; the classifier only treats non-public-overlap terms as private-topic terms.

### Query privacy audit command

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/audit_privacy_hopqa_query_privacy_levels.py \
  --out-dir runtime_analysis/privacy_hopqa_query_privacy_levels_2026-05-14
```

Inputs:

```text
run_root=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122
dataset=data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl
retrieval_audit_rows=runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14_large240/retrieval_method_rows.jsonl
```

Viewer:

```text
runtime_analysis/privacy_hopqa_query_privacy_levels_2026-05-14/index.html
```

### Observed query privacy distribution

This scans unique observed `web_search` calls from the trace root. The source prefix below is the true hop type being solved when the web query was made.

| group | queries | share |
| --- | ---: | ---: |
| overall | 19,066 | 100.0% |
| P0 raw or bridge leaky | 15,597 | 81.8% |
| P1 private details remain | 1,263 | 6.6% |
| P2 private-topic paraphrase | 236 | 1.2% |
| P3 public-targeted analogue | 898 | 4.7% |
| P4 independent public hop | 1,072 | 5.6% |
| local-hop web queries, P0 | 14,507 | 76.1% of all queries |
| web-hop queries, P4 | 1,072 | 5.6% of all queries |

Interpretation:

- Most observed web queries are leaky because the current planner often includes company names or bridge answers, especially while solving local hops.
- That is a planner/privacy issue, not a reason to avoid RRF. RRF only changes candidate ordering after a query exists.
- The query privacy viewer is meant to help debug whether future training shifts queries from P0/P1 toward P2/P3 for local hops and toward P4 for web hops.

### Retrieval mode recommendation

Use RRF for both web and local as the consistent default:

```text
--browsecomp-rank-mode rrf
--browsecomp-rrf-k 20
--local-rank-mode rrf
--local-bm25-recall-k 100
--local-dense-recall-k 100
--local-rrf-k 20
```

Keep BM25-only as a controlled ablation, not the default:

```text
--browsecomp-rank-mode bm25
--local-rank-mode bm25
```

Rationale:

- BM25-only is strong in several samples, including local retrieval.
- RRF is the better default because it keeps lexical gains while preserving dense signal, and it performed best on gold-candidate coverage in the realistic replay.
- Using RRF for both sources makes the helper behavior easier to reason about: local and web are both ranked by the same fusion principle, with source-specific corpora and recall pools.
