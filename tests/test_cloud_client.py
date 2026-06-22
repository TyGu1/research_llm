"""Tests for CloudClient: auth headers, async support, warmup, max_tokens."""
from unittest.mock import patch, MagicMock


class TestCloudClientAzureOSSAuth:
    def test_azure_url_gets_api_key_header(self):
        """When base_url contains 'azure', the api-key header should be set."""
        with patch("src.llm.cloud_client.OpenAI") as mock_openai:
            from src.llm.cloud_client import CloudClient
            CloudClient(
                provider="openai",
                model="gpt-oss-120b",
                api_key="test-key",
                base_url="https://example.services.ai.azure.com/openai/v1/",
            )
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["default_headers"]["api-key"] == "test-key"
            assert call_kwargs["base_url"] == "https://example.services.ai.azure.com/openai/v1/"

    def test_non_azure_url_no_extra_header(self):
        """When base_url is not Azure, no api-key header should be added."""
        with patch("src.llm.cloud_client.OpenAI") as mock_openai:
            from src.llm.cloud_client import CloudClient
            CloudClient(
                provider="openai",
                model="gpt-4o-mini",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["default_headers"] == {}

    def test_no_base_url_no_extra_header(self):
        """When no base_url, no api-key header should be added."""
        with patch("src.llm.cloud_client.OpenAI") as mock_openai:
            from src.llm.cloud_client import CloudClient
            CloudClient(provider="openai", model="gpt-4o", api_key="test-key")
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["default_headers"] == {}


class TestCloudClientAsyncAndWarmup:
    @patch("src.llm.cloud_client.AsyncOpenAI")
    @patch("src.llm.cloud_client.OpenAI")
    def test_openai_provider_creates_async_client(self, mock_openai, mock_async):
        from src.llm.cloud_client import CloudClient
        c = CloudClient(provider="openai", model="test", api_key="k")
        assert c._async_client is not None
        mock_async.assert_called_once()

    @patch("src.llm.cloud_client.AzureOpenAI")
    def test_azure_provider_no_async_client(self, mock_azure):
        from src.llm.cloud_client import CloudClient
        c = CloudClient(provider="azure", model="test", api_key="k", base_url="https://x.com")
        assert c._async_client is None

    def test_mock_provider_no_async(self):
        from src.llm.cloud_client import CloudClient
        c = CloudClient(provider="mock", model="test", api_key="k")
        assert c._async_client is None

    def test_generate_passes_max_tokens(self):
        from src.llm.cloud_client import CloudClient
        c = CloudClient(provider="mock", model="test", api_key="k")
        result = c.generate("hello", max_tokens=100)
        assert "[mock" in result

    def test_warmup_no_crash_on_mock(self):
        from src.llm.cloud_client import CloudClient
        c = CloudClient(provider="mock", model="test", api_key="k")
        c.warmup()  # should not raise
