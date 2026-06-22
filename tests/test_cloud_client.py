"""Tests for CloudClient Azure AI OSS auth header logic."""
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
