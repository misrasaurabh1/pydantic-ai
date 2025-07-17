from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


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
