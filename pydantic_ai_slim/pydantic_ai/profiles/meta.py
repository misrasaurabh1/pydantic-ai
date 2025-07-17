from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


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
