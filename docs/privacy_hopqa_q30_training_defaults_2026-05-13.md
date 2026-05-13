# Privacy HopQA q30 training defaults

Current default full dataset:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.jsonl`

Current shared helper artifact bundle:

`/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513`

This bundle is the helper-facing source for fixed local indices, BrowseComp
embedding shards/BM25 cache, the long-document blacklist, and curated web
overlay files. The older local paths are kept as comments in the configs for
debugging old runs, but new helpers should use the shared bundle.

Canonical splits:

| split | rows | company mix |
| --- | ---: | --- |
| train | 559 | Elexion Automotive 311, Lee's Market 248 |
| val | 98 | Elexion Automotive 56, Lee's Market 42 |
| test | 344 | MediConn Solutions 344 |

The train/val/test chain IDs are disjoint and their union is the full 1,001-chain dataset. The held-out test split is the full MediConn Solutions company slice.

PipelineRL defaults updated:

- `conf/domain_params/privacy_hopqa.yaml` now points `final_dataset_path`, `final_train_dataset_path`, `final_val_dataset_path`, and `final_test_dataset_path` at the q30 residual-review files.
- `pipelinerl.domains.privacy_hopqa.dataset.load_problems` now also supports `final_train`, `final_val`, and `final_test` aliases.
- The Privacy HopQA local-index fallback points at `/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/local_indices_fixeddeps_20260415` unless `PRIVACY_AGENT_LOCAL_INDEX_ROOT` is set.
- The BrowseComp index fallback points at `/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/browsecomp_indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl` unless `PRIVACY_AGENT_BROWSECOMP_INDEX_GLOB` is set.

Training metrics added:

- `reward_by_pattern/<pattern>` and `success_by_pattern/<pattern>`, for example `reward_by_pattern/LLW`.
- `reward_by_chain_length/<n>` and `success_by_chain_length/<n>`, for example `reward_by_chain_length/3`.

These are sparse per-rollout metrics. The actor flattens them into ordinary numeric stats, so W&B and stats viewers should show mean values under the usual actor/dataset namespaces, for example `.../reward_by_pattern/LLW_mean`.

Interpretation:

- `reward_by_pattern/LLW_mean` is the mean final reward over rollout groups whose gold hop pattern is LLW.
- `success_by_pattern/LLW_mean` is the fraction of LLW rollout groups with all hops correct and no rollout error.
- `reward_by_chain_length/3_mean` is the mean final reward over 3-hop chains.
- `success_by_chain_length/3_mean` is the all-hops-correct success rate for 3-hop chains.
