from __future__ import annotations as _annotations

import os

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Together AI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class TogetherProvider(Provider[AsyncOpenAI]):
    """Provider for Together AI API."""

    @property
    def name(self) -> str:
        return 'together'

    @property
    def base_url(self) -> str:
        return 'https://api.together.xyz/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()
        try:
            provider, model_short_name = model_name.split('/', 1)
        except ValueError:
            return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)
        if provider in _PROVIDER_TO_PROFILE:
            profile = _PROVIDER_TO_PROFILE[provider](model_short_name)
        elif provider in ('deepseek-ai', 'mistralai'):
            profile = None
        else:
            profile = None
        # As the Together API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        key = api_key or os.getenv('TOGETHER_API_KEY')
        if not key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                'to use the Together AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            if http_client is None:
                http_client = cached_async_http_client(provider='together')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        key = api_key or os.getenv('TOGETHER_API_KEY')
        if not key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                'to use the Together AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            if http_client is None:
                http_client = cached_async_http_client(provider='together')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        key = api_key or os.getenv('TOGETHER_API_KEY')
        if not key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                'to use the Together AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            if http_client is None:
                http_client = cached_async_http_client(provider='together')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        key = api_key or os.getenv('TOGETHER_API_KEY')
        if not key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                'to use the Together AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            if http_client is None:
                http_client = cached_async_http_client(provider='together')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        key = api_key or os.getenv('TOGETHER_API_KEY')
        if not key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                'to use the Together AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            if http_client is None:
                http_client = cached_async_http_client(provider='together')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=key, http_client=http_client)


# Move profile functions here for efficiency:
def _google_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Google model."""
    # GoogleJsonSchemaTransformer is likely available by import in the real codebase, but omitted here for reference.
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


def _qwen_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Qwen model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


def _meta_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


_PROVIDER_TO_PROFILE = {
    'google': _google_model_profile,
    'qwen': _qwen_model_profile,
    'meta-llama': _meta_model_profile,
    # The following providers return None so can skip function call and just skip/None
    # 'deepseek-ai': lambda _: None,
    # 'mistralai': lambda _: None,
}
