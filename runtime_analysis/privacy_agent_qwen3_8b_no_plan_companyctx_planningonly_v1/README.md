# Runtime Analysis for `privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_v1`

## Notes
- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.
- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.
- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.

## Headline Stats
- Completed rollouts analyzed: `17069`
- Total completed tool actions analyzed: `95098`
- Total LLM request duration samples: `124462`
- Total LLM call summaries: `0`
- Evaluated reports with rewards: `17069`
- Median LLM request latency: `13.73s`
- Median LLM response tokens: n/a
- Median LLM total tokens: n/a
- Mean rollout reward: `0.044`
- Median rollout reward: `0.000`
- Median completed rollout duration: `63.50s`
- Median tool execution time: `0.53s`
- Median Latest Findings share of logged prompts: `98.3%`

## Top LLM Call Types

## Top Action Types
- `web_search`: `57395`
- `local_document_search`: `16191`
- `context_synthesis`: `7521`
- `data_analysis`: `6690`
- `local_file_analysis`: `4761`
- `file_download`: `1825`
- `enterprise_api`: `410`
- `url_fetch`: `157`
