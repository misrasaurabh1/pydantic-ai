from __future__ import annotations as _annotations

from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer

from . import ModelProfile


def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    return None


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
