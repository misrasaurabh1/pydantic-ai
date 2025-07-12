from __future__ import annotations as _annotations

from dataclasses import Field, dataclass, fields, replace
from textwrap import dedent
from typing import Callable, Union

from typing_extensions import Self

from ..output import StructuredOutputMode
from ._json_schema import JsonSchemaTransformer


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

        # Optimization: Only compute defaults and fields needed once
        # Build a map of field name to default for fast lookup, and avoid repeated getattr
        # This avoids calling getattr(profile, ...) and f.default in the condition for every field.
        self_fields = {f.name for f in fields(self)}
        profile_fields = fields(profile)
        profile_values = profile.__dict__
        defaults = {f.name: f.default for f in profile_fields if isinstance(f, Field) or f.default is not None}
        # Use __dataclass_fields__ to handle defaults for all data class fields robustly
        defaults = {}
        for f in profile_fields:
            if f.default_factory is not dataclass._MISSING_TYPE:  # type: ignore
                defaults[f.name] = f.default_factory()
            elif f.default is not dataclass._MISSING_TYPE:
                defaults[f.name] = f.default
        # Only override values that are present in both and are not the default
        non_default_attrs = {
            fname: value
            for fname, value in profile_values.items()
            if fname in self_fields and fname in defaults and value != defaults[fname]
        }
        return replace(self, **non_default_attrs)


ModelProfileSpec = Union[ModelProfile, Callable[[str], Union[ModelProfile, None]]]

DEFAULT_PROFILE = ModelProfile()
