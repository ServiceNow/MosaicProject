# Runtime Analysis for `privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_retrieveonly_v2`

## Notes
- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.
- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.
- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.

## Headline Stats
- Completed rollouts analyzed: `14071`
- Total completed tool actions analyzed: `106338`
- Total LLM request duration samples: `114309`
- Total LLM call summaries: `114289`
- Evaluated reports with rewards: `14071`
- Median LLM request latency: `13.41s`
- Median LLM response tokens: `864`
- Median LLM total tokens: `2622`
- Mean rollout reward: `0.033`
- Median rollout reward: `0.000`
- Median completed rollout duration: `70.50s`
- Median tool execution time: `0.46s`
- Median Latest Findings share of logged prompts: `98.2%`

## Top LLM Call Types
- `analysis_final_synthesis`: `25800`
- `adaptive_iter0`: `14229`
- `action_plan_fallback`: `14198`
- `report_qa`: `14061`
- `adaptive_iter1`: `11285`
- `adaptive_iter2`: `8228`
- `adaptive_iter3`: `6430`
- `adaptive_iter4`: `5082`

## Top Action Types
- `web_search`: `57278`
- `data_analysis`: `25736`
- `local_document_search`: `23324`
