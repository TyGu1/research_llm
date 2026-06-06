from openai import OpenAI


class CloudClient:
    """Unified cloud LLM client supporting OpenAI and Anthropic (via OpenAI-compat)."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: str = ""):
        self.provider = provider
        self.model = model

        if provider == "openai":
            self._client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            # Anthropic's OpenAI-compatible endpoint
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.anthropic.com/v1",
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

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content
