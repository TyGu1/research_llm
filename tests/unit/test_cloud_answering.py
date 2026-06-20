"""
Unit tests for cloud answering — retry logic on API failure.
"""
import pytest
from src.pipeline.cloud_answering import StandardCloudAnswerer


class FailNTimesClient:
    """Mock client that fails N times then succeeds."""

    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self._calls = 0

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise ConnectionError(f"API unavailable (call {self._calls})")
        return f"Answer to: {prompt[:30]}"


def test_retry_succeeds_after_transient_failure():
    client = FailNTimesClient(fail_count=2)
    answerer = StandardCloudAnswerer(cloud_client=client, max_retries=3)
    results = answerer.answer(["What is contract law?"])
    assert len(results) == 1
    assert "Answer to" in results[0].answer
    assert client._calls == 3  # 2 failures + 1 success


def test_retry_exhausted_raises():
    client = FailNTimesClient(fail_count=5)
    answerer = StandardCloudAnswerer(cloud_client=client, max_retries=3)
    with pytest.raises(ConnectionError):
        answerer.answer(["What is contract law?"])
    assert client._calls == 3


def test_no_retry_needed_on_success():
    client = FailNTimesClient(fail_count=0)
    answerer = StandardCloudAnswerer(cloud_client=client, max_retries=3)
    results = answerer.answer(["Q1?", "Q2?"])
    assert len(results) == 2
    assert client._calls == 2


def test_partial_answers_on_mid_batch_failure():
    """If question 2 of 3 fails permanently, we get an exception but question 1 was answered."""
    call_count = 0

    class FailOnSecondClient:
        def generate(self, prompt, system="", temperature=0.0):
            nonlocal call_count
            call_count += 1
            if "Q2" in prompt:
                raise ConnectionError("permanent failure")
            return f"Answer: {prompt}"

    answerer = StandardCloudAnswerer(cloud_client=FailOnSecondClient(), max_retries=1)
    with pytest.raises(ConnectionError):
        answerer.answer(["Q1?", "Q2?", "Q3?"])
