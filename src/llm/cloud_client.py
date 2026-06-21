from openai import AzureOpenAI, OpenAI


class CloudClient:
    """Unified cloud LLM client supporting OpenAI, Azure OpenAI, and Anthropic (via OpenAI-compat)."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: str = "",
                 api_version: str = ""):
        self.provider = provider
        self.model = model

        timeout = 120.0  # seconds per request

        if provider == "openai":
            self._client = OpenAI(api_key=api_key, timeout=timeout)
        elif provider == "azure":
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version or "2024-12-01-preview",
                timeout=timeout,
            )
        elif provider == "anthropic":
            # Anthropic's OpenAI-compatible endpoint
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.anthropic.com/v1",
                timeout=timeout,
            )
        elif provider == "mock":
            self._client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        if self._client is None:
            return f"[mock cloud answer for: {prompt[:80]}...]"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        import time
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""
