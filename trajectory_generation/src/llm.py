"""Thin LLM client wrapper over any OpenAI-compatible API."""

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
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
        self._OpenAI = OpenAI
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._local = threading.local()
        self.model = model
        self.min_max_tokens = min_max_tokens

    def _get_client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
            self._local.client = client
        return client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
        top_p: float | None = None,
        top_k: int | None = None,
        use_min_max_tokens: bool = True,
    ) -> str:
        """Send chat completion, return assistant text."""
        effective_max = max_tokens
        # Build/filter/reword attach a long-form floor through min_max_tokens.
        # Short ranking/checking calls pass use_min_max_tokens=False so the
        # visible per-call limit is the real one.
        if use_min_max_tokens and self.min_max_tokens:
            effective_max = max(max_tokens, self.min_max_tokens)
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
        resp = self._get_client().chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        usage = resp.usage
        tok_info = f"prompt={usage.prompt_tokens} completion={usage.completion_tokens}" if usage else ""
        content = (resp.choices[0].message.content or "").strip()
        preview = content[:80].replace("\n", " ") if content else "(empty)"
        logger.info("LLM call %.1fs  max_tok=%d  %s  → %s", elapsed, effective_max, tok_info, preview)
        return content

    def chat_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
        top_p: float | None = None,
        top_k: int | None = None,
        batch_size: int = 16,
        use_min_max_tokens: bool = True,
    ) -> list[str]:
        """Send multiple chat completions concurrently in fixed-size batches."""
        if not messages_batch:
            return []

        outputs: list[str] = []
        for start in range(0, len(messages_batch), batch_size):
            chunk = messages_batch[start:start + batch_size]
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=min(batch_size, len(chunk))) as pool:
                futures = [
                    pool.submit(
                        self.chat,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        use_min_max_tokens=use_min_max_tokens,
                    )
                    for messages in chunk
                ]
                chunk_outputs = [future.result() for future in futures]
            elapsed = time.time() - t0
            logger.info(
                "LLM batch %.1fs  size=%d  batch_size=%d",
                elapsed,
                len(chunk),
                batch_size,
            )
            outputs.extend(chunk_outputs)
        return outputs
