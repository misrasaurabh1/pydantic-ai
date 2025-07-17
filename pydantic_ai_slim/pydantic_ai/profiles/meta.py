from __future__ import annotations as _annotations

import re
from typing import Callable

from pydantic_ai.profiles.anthropic import \
    anthropic_model_profile as _anthropic_model_profile
from pydantic_ai.profiles.mistral import \
    mistral_model_profile as _mistral_model_profile

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer

# Cache for the provider_to_profile dictionary
_provider_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
    'anthropic': lambda model_name: BedrockModelProfile(bedrock_supports_tool_choice=False).update(
        _anthropic_model_profile(model_name)
    ),
    'mistral': lambda model_name: BedrockModelProfile(bedrock_tool_result_format='json').update(
        _mistral_model_profile(model_name)
    ),
    'cohere': lambda model_name: None,  # replaced with inline for performance, original is trivial (profile is always None)
    'amazon': lambda model_name: ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer
    ),  # inlined for speed
    'meta': lambda model_name: ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer
    ),  # inlined for speed
    'deepseek': lambda model_name: None,
}


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


_VERSION_RE = re.compile(r'(.+)-v\d+(?::\d+)?$')
