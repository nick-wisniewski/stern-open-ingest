# SPDX-License-Identifier: Apache-2.0
"""Tests for model_provider_utils.py — client construction and config helpers."""

import asyncio
from unittest.mock import MagicMock, patch

from tensorlake_docai.providers.model_provider_utils import (
    _get_api_semaphore,
    get_openai_client_and_model,
    get_openai_sync_client_and_model,
)

# ---------------------------------------------------------------------------
# get_openai_client_and_model (async client)
# ---------------------------------------------------------------------------


def test_get_openai_client_regular(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_client = MagicMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client, model = get_openai_client_and_model(api_key="test-key")
        assert model is not None
        assert isinstance(model, str)


def test_get_openai_sync_client_regular(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_client = MagicMock()
    with patch("openai.OpenAI", return_value=mock_client):
        client, model = get_openai_sync_client_and_model(api_key="test-key")
        assert isinstance(model, str)


# ---------------------------------------------------------------------------
# _get_api_semaphore
# ---------------------------------------------------------------------------


def test_get_api_semaphore_outside_loop_returns_none():
    # Called outside an async context should return None gracefully
    result = _get_api_semaphore()
    assert result is None


def test_get_api_semaphore_inside_loop_returns_semaphore():
    async def _inner():
        sem = _get_api_semaphore()
        assert isinstance(sem, asyncio.Semaphore)
        # Second call returns the same semaphore (cached on loop)
        sem2 = _get_api_semaphore()
        assert sem is sem2

    asyncio.run(_inner())


def test_get_api_semaphore_limit_is_10():
    async def _inner():
        sem = _get_api_semaphore()
        assert sem._value == 10

    asyncio.run(_inner())
