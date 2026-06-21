import httpx
import json


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def embed(self, text: str, model: str | None = None) -> list[float]:
        payload = {
            "model": model or self.model,
            "input": text,
        }
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"embeddings": [[...]]} for /api/embed
        embeddings = data.get("embeddings") or data.get("embedding")
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings
