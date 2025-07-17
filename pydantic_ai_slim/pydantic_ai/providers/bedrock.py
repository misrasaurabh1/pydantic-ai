from __future__ import annotations as _annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Literal

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import NoRegionError
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.anthropic import \
    anthropic_model_profile as _anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.mistral import \
    mistral_model_profile as _mistral_model_profile
from pydantic_ai.providers import Provider

try:
    import boto3
    from botocore.client import BaseClient
    from botocore.config import Config
    from botocore.exceptions import NoRegionError

    # Cache for the provider_to_profile dictionary
    _provider_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
        'anthropic': lambda model_name: BedrockModelProfile(bedrock_supports_tool_choice=False).update(
            _anthropic_model_profile(model_name)
        ),
        'mistral':  lambda model_name: BedrockModelProfile(bedrock_tool_result_format='json').update(
            _mistral_model_profile(model_name)
        ),
        'cohere':   lambda model_name: None,  # replaced with inline for performance, original is trivial (profile is always None)
        'amazon':   lambda model_name: ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer), # inlined for speed
        'meta':     lambda model_name: ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer), # inlined for speed
        'deepseek': lambda model_name: None,
    }
except ImportError as _import_error:
    raise ImportError(
        'Please install the `boto3` package to use the Bedrock provider, '
        'you can use the `bedrock` optional group — `pip install "pydantic-ai-slim[bedrock]"`'
    ) from _import_error


@dataclass
class BedrockModelProfile(ModelProfile):
    """Profile for models used with BedrockModel.

    ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    bedrock_supports_tool_choice: bool = True
    bedrock_tool_result_format: Literal['text', 'json'] = 'text'


class BedrockProvider(Provider[BaseClient]):
    """Provider for AWS Bedrock."""

    @property
    def name(self) -> str:
        return 'bedrock'

    @property
    def base_url(self) -> str:
        return self._client.meta.endpoint_url

    @property
    def client(self) -> BaseClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """
        Return the model profile for a given Bedrock model_name.
        Optimized for speed: Compiled regex and class pre-initialized provider_to_profile cache.
        """
        # Split only on the first 2 dots (fast)
        parts = model_name.split('.', 2)

        # Handle regional prefixes (e.g. "us.") - mutate parts in-place, only if necessary
        if len(parts) > 2 and len(parts[0]) == 2:
            parts = parts[1:]

        # Reject if not at least provider/model
        if len(parts) < 2:
            return None

        provider = parts[0]
        model_name_with_version = parts[1]

        # Use the module-global precompiled regex for much faster matching
        version_match = _VERSION_RE.match(model_name_with_version)
        if version_match:
            bare_model_name = version_match.group(1)
        else:
            bare_model_name = model_name_with_version

        profile_factory = _provider_to_profile.get(provider)
        if profile_factory is not None:
            return profile_factory(bare_model_name)

        return None

    def __init__(
        self,
        *,
        bedrock_client: BaseClient | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None:
        """
        Initialize BedrockProvider. If bedrock_client is provided, use it; otherwise, create a boto3 Session/client.
        Minor I/O optimizations: don't env-read os.getenv except if needed.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            try:
                # Only check env vars if the arguments are None, to minimize lookups
                read_timeout = aws_read_timeout if aws_read_timeout is not None else float(os.getenv('AWS_READ_TIMEOUT', 300))
                connect_timeout = aws_connect_timeout if aws_connect_timeout is not None else float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=region_name,
                    profile_name=profile_name,
                )
                self._client = session.client(
                    'bedrock-runtime',
                    config=Config(read_timeout=read_timeout, connect_timeout=connect_timeout),
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc

    def __init__(
        self,
        *,
        bedrock_client: BaseClient | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None:
        """
        Initialize BedrockProvider. If bedrock_client is provided, use it; otherwise, create a boto3 Session/client.
        Minor I/O optimizations: don't env-read os.getenv except if needed.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            try:
                # Only check env vars if the arguments are None, to minimize lookups
                read_timeout = aws_read_timeout if aws_read_timeout is not None else float(os.getenv('AWS_READ_TIMEOUT', 300))
                connect_timeout = aws_connect_timeout if aws_connect_timeout is not None else float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=region_name,
                    profile_name=profile_name,
                )
                self._client = session.client(
                    'bedrock-runtime',
                    config=Config(read_timeout=read_timeout, connect_timeout=connect_timeout),
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc

    def __init__(
        self,
        *,
        bedrock_client: BaseClient | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None:
        """
        Initialize BedrockProvider. If bedrock_client is provided, use it; otherwise, create a boto3 Session/client.
        Minor I/O optimizations: don't env-read os.getenv except if needed.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            try:
                # Only check env vars if the arguments are None, to minimize lookups
                read_timeout = aws_read_timeout if aws_read_timeout is not None else float(os.getenv('AWS_READ_TIMEOUT', 300))
                connect_timeout = aws_connect_timeout if aws_connect_timeout is not None else float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=region_name,
                    profile_name=profile_name,
                )
                self._client = session.client(
                    'bedrock-runtime',
                    config=Config(read_timeout=read_timeout, connect_timeout=connect_timeout),
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc

_VERSION_RE = re.compile(r'(.+)-v\d+(?::\d+)?$')
