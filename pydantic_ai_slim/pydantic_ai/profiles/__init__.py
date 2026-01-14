from __future__ import annotations as _annotations

from dataclasses import dataclass, fields, replace
from functools import lru_cache
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
        self_fields, self_field_names, self_defaults = _fields_and_defaults(type(self))
        profile_fields, _, profile_defaults = _fields_and_defaults(type(profile))
        # Only include profile attributes that are in this class and not default
        non_default_attrs = {
            f.name: getattr(profile, f.name)
            for f in profile_fields
            if (f.name in self_field_names and getattr(profile, f.name) != profile_defaults.get(f.name, None))
        }
        return replace(self, **non_default_attrs)


# --------------------------------------------------------------------------
# Efficient field name and object default lookups for update()


@lru_cache(maxsize=32)
def _fields_and_defaults(cls):
    flds = fields(cls)
    return (tuple(flds), {f.name for f in flds}, {f.name: f.default for f in flds})


ModelProfileSpec = Union[ModelProfile, Callable[[str], Union[ModelProfile, None]]]

DEFAULT_PROFILE = ModelProfile()
