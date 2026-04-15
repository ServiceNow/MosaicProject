# Runtime Analysis for `privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_retrieveonly_v2`

## Notes
- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.
- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.
- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.

## Headline Stats
- Completed rollouts analyzed: `9812`
- Total completed tool actions analyzed: `66995`
- Total LLM request duration samples: `73385`
- Total LLM call summaries: `73371`
- Evaluated reports with rewards: `9812`
- Median LLM request latency: `13.21s`
- Median LLM response tokens: `852`
- Median LLM total tokens: `2573`
- Mean rollout reward: `0.031`
- Median rollout reward: `0.000`
- Median completed rollout duration: `63.61s`
- Median tool execution time: `0.47s`
- Median Latest Findings share of logged prompts: `98.2%`

## Top LLM Call Types
- `analysis_final_synthesis`: `15562`
- `adaptive_iter0`: `9925`
- `action_plan_fallback`: `9915`
- `report_qa`: `9811`
- `adaptive_iter1`: `7740`
- `adaptive_iter2`: `5425`
- `adaptive_iter3`: `4029`
- `adaptive_iter4`: `3036`

## Top Action Types
- `web_search`: `37317`
- `data_analysis`: `15533`
- `local_document_search`: `14145`
