"""OpenAI-compatible client for the adversary and judge calls.

chat is the default and is what you want: it works with vLLM's chat endpoint and
with OpenRouter, and lets the server apply the model's own chat template.

completion_template writes the ChatML tags itself and posts to the raw
completions endpoint. It is hacky, and it is only here because every reported
number was produced with it - it was the default in the original scripts. No
reason for that choice is recorded anywhere in the original code, so treat it as
a reproducibility switch rather than a recommendation. It also assumes the model
speaks ChatML, which not every model does.
"""

import asyncio
import time
from typing import Any, Callable

from openai import AsyncOpenAI

from mosaicleaks.privacy import config

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Truncation bound for the raw text kept alongside every parsed response. Long
# enough to debug a parse failure, short enough that score files stay readable.
RAW_OUTPUT_LIMIT = 12000


class JudgeClient:
    """Issues adversary/judge requests and returns parsed JSON objects.

    Each successful response carries a _call key holding the attempt count,
    latency, finish reason, token usage, and raw text.
    """

    def __init__(
        self,
        model: str = config.JUDGE_MODEL,
        base_url: str = config.JUDGE_BASE_URL,
        api_key: str = "EMPTY",
        api_mode: str = "chat",
        temperature: float = config.TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS,
        timeout: float = config.REQUEST_TIMEOUT,
        max_attempts: int = config.MAX_ATTEMPTS,
        concurrency: int = config.CONCURRENCY,
        use_response_format: bool = True,
    ):
        self.model = model
        self.api_mode = api_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.use_response_format = use_response_format
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _request(self, prompt: str) -> tuple[str, str, dict[str, Any]]:
        """Returns (content, finish_reason, usage) for one attempt."""
        if self.api_mode == "completion_template":
            response = await self._client.completions.create(
                model=self.model,
                prompt=CHATML_TEMPLATE.format(system=config.SYSTEM_PROMPT, user=prompt),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=["<|im_end|>"],
                timeout=self.timeout,
            )
            choice = response.choices[0]
            content = choice.text
        else:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
            }
            if self.use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            # Some reasoning models put the answer in `reasoning` when the
            # content field comes back empty.
            content = choice.message.content or getattr(choice.message, "reasoning", None) or ""

        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return content or "", choice.finish_reason, usage

    async def complete_json(
        self,
        prompt: str,
        parser: Callable[[str], dict[str, Any] | None],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Returns (parsed_object, error_metadata).

        On success the second element is empty; on exhaustion of all attempts the
        first is None and the second describes the final failure.
        """
        async with self._semaphore:
            last: dict[str, Any] = {}
            for attempt in range(1, self.max_attempts + 1):
                started = time.time()
                try:
                    content, finish_reason, usage = await self._request(prompt)
                    last = {
                        "attempt": attempt,
                        "elapsed_s": round(time.time() - started, 3),
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "raw_output": content[:RAW_OUTPUT_LIMIT],
                    }
                    parsed = parser(content)
                    if isinstance(parsed, dict):
                        parsed["_call"] = last
                        return parsed, {}
                    last["error"] = "json_parse_failed"
                except Exception as exc:
                    last = {"attempt": attempt, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            return None, last
