"""Serve the privacy reward model, backed by vLLM.

This is the shape the RL loop needs. The reward is queried on every planning step
of every rollout across parallel actors, so throughput is the whole game: vLLM
does continuous batching and holds the weights, and this process is a thin shim
that renders prompts and reads back one number.

Scoring never generates text. vLLM is asked for a single token constrained to the
two label ids, with the top two logprobs, and prob_yes is the softmax over those:

    max_tokens=1, logprobs=2, allowed_token_ids=[yes_id, no_id]

Constraining the ids is what makes that equivalent to reading the label logits
directly - nothing else can come back, so the top two logprobs are the whole
distribution.

Start vLLM with the adapter, then this:

    vllm serve BASE_MODEL --enable-lora --lora-modules privacy_rm=runs/rm/adapter
    mosaicleaks-reward-serve --model privacy_rm --base-model BASE_MODEL \\
        --vllm-base-url http://127.0.0.1:8000 --port 8020
"""

import argparse
import math
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from mosaicleaks.reward.dataset import SYSTEM_PROMPT
from mosaicleaks.reward.prompts import SERVABLE_VARIANTS, build_prompt, compact


class Example(BaseModel):
    id: str = ""
    web_queries: list[str] = []
    company: str = ""
    private_qas: list[dict] = []
    local_hop_facts: list[dict] = []
    variant: str = "private_fact_context"


class ScoreRequest(BaseModel):
    examples: list[Example]


class Score(BaseModel):
    id: str
    prob_yes: float
    leaked: bool
    n_web_queries: int


class ScoreResponse(BaseModel):
    scores: list[Score]


class RewardProxy:
    def __init__(self, model: str, base_model: str, vllm_base_url: str, batch_size: int, timeout_s: float):
        self.model = model
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.batch_size = batch_size
        self.client = httpx.AsyncClient(timeout=timeout_s)

        # The tokenizer is only here to resolve the two label ids, and to render
        # the chat template the model was trained with.
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.yes_id, self.no_id = self._label_token_ids()
        self.yes_token = self.tokenizer.decode([self.yes_id])
        self.no_token = self.tokenizer.decode([self.no_id])

    def _label_token_ids(self) -> tuple[int, int]:
        yes = self.tokenizer.encode("Yes", add_special_tokens=False)
        no = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError(f"labels must be single tokens for this tokenizer, got Yes={yes} No={no}")
        return yes[0], no[0]

    def render(self, example: Example) -> str:
        if example.variant not in SERVABLE_VARIANTS:
            raise ValueError(f"variant {example.variant} needs fields the server is not given")
        prompt = build_prompt(
            example.variant,
            example.web_queries,
            company=example.company,
            private_qas=example.private_qas,
            local_hop_facts=example.local_hop_facts,
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _prob_yes(self, choice: dict[str, Any]) -> float:
        logprobs = choice.get("logprobs") or {}
        top = logprobs.get("top_logprobs") or []
        if not top or not isinstance(top[0], dict):
            raise ValueError("vLLM response carried no first-token top_logprobs")

        def lookup(token: str) -> float:
            entry = top[0]
            if token in entry:
                return float(entry[token])
            # Tokenizers differ on leading whitespace; match on the stripped form.
            for key, value in entry.items():
                if key.strip() == token.strip():
                    return float(value)
            raise ValueError(f"vLLM top_logprobs did not include {token!r}: {sorted(entry)}")

        yes, no = lookup(self.yes_token), lookup(self.no_token)
        ceiling = max(yes, no)
        return math.exp(yes - ceiling) / (math.exp(yes - ceiling) + math.exp(no - ceiling))

    async def score(self, examples: list[Example]) -> list[Score]:
        scores: list[Score] = []
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start:start + self.batch_size]
            response = await self.client.post(
                f"{self.vllm_base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": [self.render(example) for example in batch],
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": 2,
                    "allowed_token_ids": [self.yes_id, self.no_id],
                },
            )
            response.raise_for_status()
            choices = response.json()["choices"]
            if len(choices) != len(batch):
                raise ValueError(f"vLLM returned {len(choices)} choices for {len(batch)} prompts")
            for example, choice in zip(batch, choices):
                prob_yes = self._prob_yes(choice)
                scores.append(Score(
                    id=example.id,
                    prob_yes=prob_yes,
                    leaked=prob_yes >= 0.5,
                    n_web_queries=len([q for q in example.web_queries if compact(q)]),
                ))
        return scores


def create_app(proxy: RewardProxy) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await proxy.client.aclose()

    app = FastAPI(title="mosaicleaks-privacy-reward-model", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        upstream = await proxy.client.get(f"{proxy.vllm_base_url}/health")
        return {"status": "ok" if upstream.status_code == 200 else "vllm_unhealthy",
                "model": proxy.model, "variants": list(SERVABLE_VARIANTS)}

    @app.post("/score", response_model=ScoreResponse)
    async def score(request: ScoreRequest) -> ScoreResponse:
        try:
            return ScoreResponse(scores=await proxy.score(request.examples))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def main() -> int:
    p = argparse.ArgumentParser(description="Serve the privacy reward model over vLLM.")
    p.add_argument("--model", required=True, help="LoRA module name registered with vLLM.")
    p.add_argument("--base-model", required=True, help="Base model, for the tokenizer only.")
    p.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--timeout-s", type=float, default=120.0)
    args = p.parse_args()

    proxy = RewardProxy(args.model, args.base_model, args.vllm_base_url, args.batch_size, args.timeout_s)
    uvicorn.run(create_app(proxy), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
