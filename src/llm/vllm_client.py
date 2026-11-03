"""
Thin OpenAI-compatible client for the local vLLM server (Gemma-4-31B on :8020).

Same generate() interface as OllamaClient/CloudClient so it plugs into every
pipeline step, baseline and attacker. Thread-safe (the OpenAI client is), retries
with backoff, strips a leaked <think> block if the model emits one.
"""
from __future__ import annotations

import re
import time

from openai import OpenAI

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8020/v1",
        model: str = "local/active-model",
        max_tokens: int = 700,
        timeout: float = 600.0,
        disable_thinking: bool = True,
        max_retries: int = 3,
    ):
        self.model = model
        self._default_max_tokens = max_tokens
        self._client = OpenAI(api_key="local", base_url=base_url, timeout=timeout, max_retries=0)
        self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}
        self._max_retries = max_retries
        self._async_client = None  # keeps ParallelS0Generator on the sync path

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0,
                 max_tokens: int | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(
            model=self.model, messages=messages, temperature=temperature,
            max_tokens=max_tokens or self._default_max_tokens,
        )
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return THINK_RE.sub("", text).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"vLLM request failed after {self._max_retries} attempts: {last_err}")
