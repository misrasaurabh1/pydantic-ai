from __future__ import annotations as _annotations

import os

from groq import AsyncGroq
from httpx import AsyncClient as AsyncHTTPClient

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.providers import Provider

try:
    from groq import AsyncGroq
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `groq` package to use the Groq provider, '
        'you can use the `groq` optional group — `pip install "pydantic-ai-slim[groq]"`'
    ) from _import_error


class GroqProvider(Provider[AsyncGroq]):
    """Provider for Groq API."""

    @property
    def name(self) -> str:
        return 'groq'

    @property
    def base_url(self) -> str:
        return os.environ.get('GROQ_BASE_URL', 'https://api.groq.com')

    @property
    def client(self) -> AsyncGroq:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        # Optimized: static prefix list, .lower() once, no per-call dict allocation
        name = model_name.lower()
        for prefix, profile_func in _GROQ_MODEL_PREFIXES:
            if name.startswith(prefix):
                # Only strip prefix if it ends with '/' (original code intent)
                if prefix.endswith('/'):
                    name_without_prefix = name[len(prefix) :]
                else:
                    name_without_prefix = name
                return profile_func(name_without_prefix)
        return None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
        """
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            self._client = groq_client
        else:
            api_key = api_key or os.environ.get('GROQ_API_KEY')

            if not api_key:
                raise UserError(
                    'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                    'to use the Groq provider.'
                )
            elif http_client is not None:
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='groq')
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
        """
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            self._client = groq_client
        else:
            api_key = api_key or os.environ.get('GROQ_API_KEY')

            if not api_key:
                raise UserError(
                    'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                    'to use the Groq provider.'
                )
            elif http_client is not None:
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='groq')
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
        """
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            self._client = groq_client
        else:
            api_key = api_key or os.environ.get('GROQ_API_KEY')

            if not api_key:
                raise UserError(
                    'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                    'to use the Groq provider.'
                )
            elif http_client is not None:
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='groq')
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)


_GROQ_MODEL_PREFIXES = (
    ('meta-llama/', lambda _: ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)),
    ('llama', lambda _: ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)),
    (
        'gemma',
        lambda _: ModelProfile(
            json_schema_transformer=GoogleJsonSchemaTransformer,
            supports_json_schema_output=True,
            supports_json_object_output=True,
        ),
    ),
    ('qwen', lambda _: ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)),
    ('deepseek', lambda _: None),
    ('mistral', lambda _: None),
)
