from __future__ import annotations as _annotations

from dataclasses import dataclass, fields, replace
from textwrap import dedent
from typing import Callable, Union

from typing_extensions import Self

from ..output import StructuredOutputMode
from ._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer


@dataclass
class ModelProfile:
    """Describes how requests to a specific model or family of models need to be constructed to get the best results, independent of the model and provider classes used."""

    supports_tools: bool = True
    """Whether the model supports tools."""
    supports_json_schema_output: bool = False
    """Whether the model supports JSON schema output."""
    supports_json_object_output: bool = False
    """Whether the model supports JSON object output."""
    default_structured_output_mode: StructuredOutputMode = 'tool'
    """The default structured output mode to use for the model."""
    prompted_output_template: str = dedent(
        """
        Always respond with a JSON object that's compatible with this schema:

        {schema}

        Don't include any text or Markdown fencing before or after.
        """
    )
    """The instructions template to use for prompted structured output. The '{schema}' placeholder will be replaced with the JSON schema for the output."""
    json_schema_transformer: type[JsonSchemaTransformer] | None = None
    """The transformer to use to make JSON schemas for tools and structured output compatible with the model."""

    @classmethod
    def from_profile(cls, profile: ModelProfile | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        if isinstance(profile, cls):
            return profile
        return cls().update(profile)

    def update(self, profile: ModelProfile | None) -> Self:
        """Update this ModelProfile (subclass) instance with the non-default values from another ModelProfile instance."""
        if not profile:
            return self
        non_default_attrs = {}
        for f in fields(self):
            try:
                value = getattr(profile, f.name)
                if value != f.default:
                    non_default_attrs[f.name] = value
            except AttributeError:
                continue
        return replace(self, **non_default_attrs)


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


ModelProfileSpec = Union[ModelProfile, Callable[[str], Union[ModelProfile, None]]]

DEFAULT_PROFILE = ModelProfile()

_PROVIDER_TO_PROFILE = {
    'google': _google_model_profile,
    'qwen': _qwen_model_profile,
    'meta-llama': _meta_model_profile,
    # The following providers return None so can skip function call and just skip/None
    # 'deepseek-ai': lambda _: None,
    # 'mistralai': lambda _: None,
}
