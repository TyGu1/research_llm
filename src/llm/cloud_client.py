from openai import AsyncOpenAI, AzureOpenAI, OpenAI


class CloudClient:
    """Unified cloud LLM client supporting OpenAI, Azure OpenAI, and Anthropic (via OpenAI-compat)."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: str = "",
                 api_version: str = ""):
        self.provider = provider
        self.model = model

        timeout = 120.0  # seconds per request

        if provider == "openai":
            # Azure AI OSS models need api-key header instead of Bearer token
            extra_headers = {"api-key": api_key} if "azure" in (base_url or "") else {}
            self._client = OpenAI(
                api_key=api_key, base_url=base_url or None,
                timeout=timeout, default_headers=extra_headers,
            )
            self._async_client = AsyncOpenAI(
                api_key=api_key, base_url=base_url or None,
                timeout=timeout, default_headers=extra_headers,
            )
        elif provider == "azure":
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version or "2024-12-01-preview",
                timeout=timeout,
            )
            self._async_client = None
        elif provider == "anthropic":
            # Anthropic's OpenAI-compatible endpoint
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.anthropic.com/v1",
                timeout=timeout,
            )
            self._async_client = None
        elif provider == "mock":
            self._client = None
            self._async_client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0,
                 max_tokens: int | None = None) -> str:
        if self._client is None:
            return f"[mock cloud answer for: {prompt[:80]}...]"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(model=self.model, messages=messages, temperature=temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        import time
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""

    async def agenerate(self, prompt: str, system: str = "", temperature: float = 0.0,
                        max_tokens: int | None = None) -> str:
        """Async version of generate() for parallel calls."""
        if self._async_client is None:
            return self.generate(prompt, system, temperature, max_tokens)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(model=self.model, messages=messages, temperature=temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        import asyncio
        for attempt in range(3):
            try:
                resp = await self._async_client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        return ""

    def warmup(self):
        """Send a minimal request to establish TLS connection."""
        if self._client is None:
            return
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
        except Exception:
            pass
