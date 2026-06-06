import hashlib
import json


class MockOllamaClient:
    """Deterministic mock for unit tests. No Ollama required."""

    def __init__(self, base_url: str = "", model: str = "mock"):
        self.model = model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        # Return well-structured responses based on prompt content
        if "json" in prompt.lower() or "topics" in prompt.lower():
            return json.dumps({
                "topics": {"contract_law": 0.5, "business_agreements": 0.3, "legal_obligations": 0.2}
            })
        if "questions" in prompt.lower() or "generate" in prompt.lower():
            return "\n".join([
                f"1. What are the key obligations under this type of agreement?",
                f"2. How are disputes typically resolved in similar contracts?",
                f"3. What termination conditions are standard for this clause type?",
                f"4. What liability limitations are commonly included?",
                f"5. How is notice typically defined in these agreements?",
                f"6. What are common renewal terms in similar arrangements?",
                f"7. What representations and warranties are standard?",
                f"8. How is confidentiality typically handled?",
            ])
        if "select" in prompt.lower() or "closest" in prompt.lower():
            return "1, 2, 3"
        if "synthesize" in prompt.lower() or "answer" in prompt.lower():
            return f"Based on standard legal frameworks, the contract provisions relate to [{h}] typical obligations and rights."
        return f"Mock response [{h}]: This is a generic response about legal contract terms and obligations."

    def embed(self, text: str, model: str | None = None) -> list[float]:
        # Deterministic pseudo-embedding based on text hash
        h = hashlib.md5(text.encode()).digest()
        floats = [((b / 255.0) - 0.5) * 2 for b in h]
        # Pad to 384 dimensions (all-MiniLM-L6-v2 size)
        return (floats * (384 // len(floats) + 1))[:384]
