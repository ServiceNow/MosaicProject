"""Thin LLM client wrapper over any OpenAI-compatible API."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        min_max_tokens: int | None = None,
    ) -> None:
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or ""
        if not api_key:
            if base_url is None:
                raise RuntimeError("No API key and no base_url provided.")
            api_key = "DUMMY"

        timeout = 600.0 if min_max_tokens and min_max_tokens >= 8000 else 120.0
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.min_max_tokens = min_max_tokens

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> str:
        """Send chat completion, return assistant text."""
        effective_max = max(max_tokens, self.min_max_tokens) if self.min_max_tokens else max_tokens
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        # top_k is vLLM-specific; OpenAI API doesn't support it, so we skip it
        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        usage = resp.usage
        tok_info = f"prompt={usage.prompt_tokens} completion={usage.completion_tokens}" if usage else ""
        content = (resp.choices[0].message.content or "").strip()
        preview = content[:80].replace("\n", " ") if content else "(empty)"
        logger.info("LLM call %.1fs  max_tok=%d  %s  → %s", elapsed, effective_max, tok_info, preview)
        return content
