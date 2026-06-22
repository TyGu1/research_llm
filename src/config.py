from pydantic_settings import BaseSettings
from pydantic import Field


class PipelineConfig(BaseSettings):
    # Local LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "gpt-oss-120b"  # Azure AI OSS model; switch to gpt-oss-20b when available
    local_model_heavy: str = "gemma4:26b"
    embed_model: str = "nomic-embed-text"

    # Local LLM (cloud-as-local for steps 1-4, 6)
    local_provider: str = "openai"      # "openai" | "azure" | "ollama"
    local_api_key: str = ""
    local_base_url: str = ""            # OpenAI-compatible endpoint for local model

    # Cloud LLM (step 5: cloud answering)
    cloud_provider: str = "mock"        # "openai" | "azure" | "anthropic" | "mock"
    cloud_model: str = "gpt-4o"
    cloud_api_key: str = ""
    cloud_base_url: str = ""            # override for Anthropic or Azure endpoint
    cloud_api_version: str = ""         # Azure API version

    # Pipeline parameters
    num_proxy_questions: int = 10       # Step 3: generate N proxy questions
    num_selected_questions: int = 3     # Step 4: select K closest
    s0_text_min_words: int = 200
    s0_text_max_words: int = 300
    use_vocab_exclusion: bool = False   # toggle hardening variant
    question_selector: str = "embedding"  # "embedding" | "llm"
    temperature: float = 0.0           # 0.0 for eval runs, higher for exploration

    # Context handling
    local_model_max_tokens: int = 0     # 0 = no truncation (gpt-4o-mini has 128K ctx)
    heavy_model_max_tokens: int = 0     # 0 = no truncation (gemma4:26b 128k ctx)

    # CUAD sampling
    cuad_max_samples: int = 50
    cuad_answerable_only: bool = True
    cuad_seed: int = 42
    cuad_split: str = "dev"  # "dev" | "test" | "all"

    # Evaluation
    similarity_model: str = "all-MiniLM-L6-v2"
    bertscore_model: str = "roberta-large"

    # Output
    results_dir: str = "results"

    class Config:
        env_file = ".env"
        env_prefix = "PIPELINE_"
