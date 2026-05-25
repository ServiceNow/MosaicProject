# Step100 token usage summary

Agent token counts come from rollout `prompt_tokens` and `output_tokens`.
Adversary + judging token counts sum the StepFun usage objects in each privacy-leakage judgment once, covering the full privacy evaluation suite for that rollout.

| Condition | Agent rollouts | Agent input / rollout | Agent output / rollout | Agent total / rollout | Agent total | Privacy judgments | Adv+judge input / judgment | Adv+judge output / judgment | Adv+judge total / judgment | Adv+judge total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 1032 | 139.9k | 6.5k | 146.4k | 151.1M | 1032 | 6.6k | 7.0k | 13.6k | 14.0M |
| Base + privacy prompt | 1032 | 125.3k | 6.0k | 131.2k | 135.4M | 1032 | 6.1k | 5.8k | 11.9k | 12.2M |
| Task s100 | 1032 | 174.1k | 9.5k | 183.5k | 189.4M | 1032 | 7.5k | 7.9k | 15.4k | 15.9M |
| Task s100 + privacy prompt | 1032 | 174.3k | 9.5k | 183.8k | 189.7M | 1032 | 7.5k | 8.5k | 15.9k | 16.4M |
| PA-DR s100 | 1032 | 169.8k | 8.8k | 178.6k | 184.3M | 1032 | 5.4k | 5.4k | 10.8k | 11.1M |
| PA-DR s100 + privacy prompt | 1032 | 166.6k | 8.7k | 175.3k | 180.9M | 1032 | 5.2k | 4.7k | 10.0k | 10.3M |
| All six step100 conditions | 6192 | 158.3k | 8.2k | 166.5k | 1030.9M | 6192 | 6.4k | 6.5k | 12.9k | 80.0M |

The adversary + judging columns are per privacy judgment. The total columns aggregate the complete six-condition step100 slice shown here.
