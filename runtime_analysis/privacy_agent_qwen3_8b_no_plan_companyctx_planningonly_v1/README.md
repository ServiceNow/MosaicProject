# Runtime Analysis for `privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_v1`

## Notes
- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.
- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.
- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.

## Headline Stats
- Completed rollouts analyzed: `2935`
- Total completed tool actions analyzed: `14936`
- Total LLM request duration samples: `19661`
- Median LLM request latency: `13.29s`
- Median completed rollout duration: `52.73s`
- Median tool execution time: `0.48s`
- Median Latest Findings share of logged prompts: `98.3%`

## Top Action Types
- `web_search`: `9400`
- `local_document_search`: `2498`
- `context_synthesis`: `1091`
- `data_analysis`: `870`
- `local_file_analysis`: `657`
- `file_download`: `336`
- `enterprise_api`: `56`
- `url_fetch`: `20`
