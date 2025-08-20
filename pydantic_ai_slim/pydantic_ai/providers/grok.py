from __future__ import annotations as _annotations

import os
from functools import lru_cache

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Grok provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class GrokProvider(Provider[AsyncOpenAI]):
    """Provider for Grok API."""

    @property
    def name(self) -> str:
        return 'grok'

    @property
    def base_url(self) -> str:
        return 'https://api.x.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = grok_model_profile(model_name)
        # Use cached base profile and only call update; minimizes redundant work
        return self._base_openai_profile().update(profile)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('GROK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
                'to use the Grok provider.'
            )
        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='grok')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('GROK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
                'to use the Grok provider.'
            )
        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='grok')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('GROK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
                'to use the Grok provider.'
            )
        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='grok')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('GROK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
                'to use the Grok provider.'
            )
        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='grok')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('GROK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
                'to use the Grok provider.'
            )
        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='grok')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    @staticmethod
    @lru_cache(maxsize=1)
    def _base_openai_profile():
        # lru_cache to ensure we only construct OpenAIModelProfile once per GrokProvider lifetime
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer, openai_supports_strict_tool_definition=False
        )
