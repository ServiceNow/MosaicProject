# Runtime Analysis for `privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_retrieveonly_v2`

## Notes
- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.
- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.
- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.

## Headline Stats
- Completed rollouts analyzed: `8844`
- Total completed tool actions analyzed: `59793`
- Total LLM request duration samples: `65574`
- Total LLM call summaries: `65563`
- Evaluated reports with rewards: `8844`
- Median LLM request latency: `13.18s`
- Median LLM response tokens: `850`
- Median LLM total tokens: `2564`
- Mean rollout reward: `0.031`
- Median rollout reward: `0.000`
- Median completed rollout duration: `63.00s`
- Median tool execution time: `0.47s`
- Median Latest Findings share of logged prompts: `98.2%`

## Top LLM Call Types
- `analysis_final_synthesis`: `13837`
- `adaptive_iter0`: `8942`
- `action_plan_fallback`: `8933`
- `report_qa`: `8836`
- `adaptive_iter1`: `6940`
- `adaptive_iter2`: `4859`
- `adaptive_iter3`: `3603`
- `adaptive_iter4`: `2688`

## Top Action Types
- `web_search`: `33365`
- `data_analysis`: `13821`
- `local_document_search`: `12607`
