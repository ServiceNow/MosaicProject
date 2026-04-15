# Runtime Analysis for `privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_v1`

## Notes
- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.
- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.
- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.

## Headline Stats
- Completed rollouts analyzed: `7832`
- Total completed tool actions analyzed: `40430`
- Total LLM request duration samples: `52339`
- Total LLM call summaries: `0`
- Evaluated reports with rewards: `7832`
- Median LLM request latency: `13.46s`
- Median LLM response tokens: n/a
- Median LLM total tokens: n/a
- Mean rollout reward: `0.042`
- Median rollout reward: `0.000`
- Median completed rollout duration: `56.22s`
- Median tool execution time: `0.48s`
- Median Latest Findings share of logged prompts: `98.3%`

## Top LLM Call Types

## Top Action Types
- `web_search`: `25591`
- `local_document_search`: `6570`
- `context_synthesis`: `3221`
- `data_analysis`: `2321`
- `local_file_analysis`: `1746`
- `file_download`: `770`
- `enterprise_api`: `114`
- `url_fetch`: `60`
