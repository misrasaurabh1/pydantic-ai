from __future__ import annotations as _annotations

import os

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncAzureOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Azure provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class AzureProvider(Provider[AsyncOpenAI]):
    """Provider for Azure OpenAI API.

    See <https://azure.microsoft.com/en-us/products/ai-foundry> for more information.
    """

    @property
    def name(self) -> str:
        return 'azure'

    @property
    def base_url(self) -> str:
        assert self._base_url is not None
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        model_name_lower = model_name.lower()
        prefix_table = [
            ('llama', meta_model_profile, 5, False),
            ('meta-', meta_model_profile, 5, True),
            ('deepseek', deepseek_model_profile, 8, False),
            ('mistralai-', mistral_model_profile, 10, True),
            ('mistral', mistral_model_profile, 7, False),
            ('cohere-', cohere_model_profile, 7, True),
            ('grok', grok_model_profile, 4, False),
        ]
        # Check prefix matches (order matters)
        for prefix, profunc, cutpoint, drop_suffix in prefix_table:
            if model_name_lower.startswith(prefix):
                matched_name = model_name[cutpoint:] if drop_suffix else model_name
                profile = profunc(matched_name)
                # Always update OpenAIModelProfile to preserve old behavior unless json_schema_transformer is set
                return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)
        # Default to openai model profile
        return openai_model_profile(model_name)

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncAzureOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._base_url = str(openai_client.base_url)
            self._client = openai_client
            return

        azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_endpoint:
            raise UserError(
                'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
            )

        api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise UserError(
                'Must provide one of the `api_key` argument or the `AZURE_OPENAI_API_KEY` environment variable'
            )

        api_version = api_version or os.environ.get('OPENAI_API_VERSION')
        if not api_version:
            raise UserError(
                'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
            )

        client = http_client or cached_async_http_client(provider='azure')
        self._client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            http_client=client,
        )
        self._base_url = str(self._client.base_url)

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncAzureOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._base_url = str(openai_client.base_url)
            self._client = openai_client
            return

        azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_endpoint:
            raise UserError(
                'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
            )

        api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise UserError(
                'Must provide one of the `api_key` argument or the `AZURE_OPENAI_API_KEY` environment variable'
            )

        api_version = api_version or os.environ.get('OPENAI_API_VERSION')
        if not api_version:
            raise UserError(
                'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
            )

        client = http_client or cached_async_http_client(provider='azure')
        self._client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            http_client=client,
        )
        self._base_url = str(self._client.base_url)

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncAzureOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._base_url = str(openai_client.base_url)
            self._client = openai_client
            return

        azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_endpoint:
            raise UserError(
                'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
            )

        api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise UserError(
                'Must provide one of the `api_key` argument or the `AZURE_OPENAI_API_KEY` environment variable'
            )

        api_version = api_version or os.environ.get('OPENAI_API_VERSION')
        if not api_version:
            raise UserError(
                'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
            )

        client = http_client or cached_async_http_client(provider='azure')
        self._client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            http_client=client,
        )
        self._base_url = str(self._client.base_url)
