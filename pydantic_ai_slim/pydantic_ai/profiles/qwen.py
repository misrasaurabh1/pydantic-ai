from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


_META_PREFIXES = ('llama', 'meta-llama/')

_GOOGLE_PREFIXES = ('gemma',)

_QWEN_PREFIXES = ('qwen',)

_DEEPSEEK_PREFIXES = ('deepseek',)

_MISTRAL_PREFIXES = ('mistral',)
