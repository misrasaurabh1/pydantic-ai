from __future__ import annotations as _annotations

from dataclasses import MISSING, dataclass, fields, replace
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
        if profile is None:
            return self
        # OPTIMIZED: Avoid repeated getattr, fields, set call; batch in one loop using zip.
        # Also, do not call fields(profile) as it is always same type as self, so cache fields().
        model_fields = fields(self)
        # Use MISSING for dataclass unset
        nd_attrs = {}
        self_get = self.__getattribute__
        profile_get = profile.__getattribute__
        for f in model_fields:
            v = profile_get(f.name)
            # f.default_factory is only set if default is not present, otherwise default is set
            default = (
                f.default
                if f.default is not MISSING
                else (f.default_factory() if f.default_factory is not MISSING else MISSING)
            )
            # Only override if v differs from default and from self (skip if same as self)
            if v != default and v != self_get(f.name):
                nd_attrs[f.name] = v
        # Early return if nothing to update.
        if not nd_attrs:
            return self
        return replace(self, **nd_attrs)


ModelProfileSpec = Union[ModelProfile, Callable[[str], Union[ModelProfile, None]]]

DEFAULT_PROFILE = ModelProfile()
