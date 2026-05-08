# Runtime Analysis for `privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_v1`

## Notes
- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.
- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.
- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.

## Headline Stats
- Completed rollouts analyzed: `20372`
- Total completed tool actions analyzed: `124227`
- Total LLM request duration samples: `165111`
- Total LLM call summaries: `0`
- Evaluated reports with rewards: `20372`
- Median LLM request latency: `13.85s`
- Median LLM response tokens: n/a
- Median LLM total tokens: n/a
- Mean rollout reward: `0.047`
- Median rollout reward: `0.000`
- Median completed rollout duration: `71.18s`
- Median tool execution time: `0.56s`
- Median Latest Findings share of logged prompts: `98.2%`

## Top LLM Call Types

## Top Action Types
- `web_search`: `70457`
- `local_document_search`: `22036`
- `data_analysis`: `10399`
- `context_synthesis`: `9416`
- `local_file_analysis`: `7591`
- `file_download`: `2784`
- `enterprise_api`: `980`
- `mcp_query`: `321`
