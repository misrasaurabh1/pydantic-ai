from __future__ import annotations as _annotations

from dataclasses import dataclass, fields, replace
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
        if not profile or self is profile:
            return self

        # Use a class-level cache for ModelProfile defaults
        cls = type(self)
        try:
            _model_fields = cls._modelprofile_fields
            _model_defaults = cls._modelprofile_defaults
        except AttributeError:
            _model_fields = tuple(fields(cls))
            _model_defaults = {f.name: f.default for f in _model_fields}
            cls._modelprofile_fields = _model_fields
            cls._modelprofile_defaults = _model_defaults

        # Build only what is needed and avoid unnecessary getattr
        non_default_attrs = {}
        for f in _model_fields:
            fname = f.name
            val = getattr(profile, fname, _model_defaults[fname])
            if fname != 'default_structured_output_mode' and val == _model_defaults[fname]:
                continue
            if getattr(self, fname) != val:
                non_default_attrs[fname] = val

        if not non_default_attrs:
            return self
        return replace(self, **non_default_attrs)


ModelProfileSpec = Union[ModelProfile, Callable[[str], Union[ModelProfile, None]]]

DEFAULT_PROFILE = ModelProfile()
