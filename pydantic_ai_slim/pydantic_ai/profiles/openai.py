from __future__ import annotations as _annotations

import re
from dataclasses import dataclass

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


@dataclass
class OpenAIModelProfile(ModelProfile):
    """Profile for models used with OpenAIModel.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_supports_strict_tool_definition: bool = True
    """This can be set by a provider or user if the OpenAI-"compatible" API doesn't support strict tool definitions."""

    openai_supports_sampling_settings: bool = True
    """Turn off to don't send sampling settings like `temperature` and `top_p` to models that don't support them, like OpenAI's o-series reasoning models."""


def openai_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for an OpenAI model."""
    is_reasoning_model = model_name.startswith('o')
    # Structured Outputs (output mode 'native') is only supported with the gpt-4o-mini, gpt-4o-mini-2024-07-18, and gpt-4o-2024-08-06 model snapshots and later.
    # We leave it in here for all models because the `default_structured_output_mode` is `'tool'`, so `native` is only used
    # when the user specifically uses the `NativeOutput` marker, so an error from the API is acceptable.
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        openai_supports_sampling_settings=not is_reasoning_model,
    )


_STRICT_INCOMPATIBLE_KEYS = [
    'minLength',
    'maxLength',
    'pattern',
    'format',
    'minimum',
    'maximum',
    'multipleOf',
    'patternProperties',
    'unevaluatedProperties',
    'propertyNames',
    'minProperties',
    'maxProperties',
    'unevaluatedItems',
    'contains',
    'minContains',
    'maxContains',
    'minItems',
    'maxItems',
    'uniqueItems',
]

_sentinel = object()


@dataclass
class OpenAIJsonSchemaTransformer(JsonSchemaTransformer):
    """Recursively handle the schema to make it compatible with OpenAI strict mode.

    See https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode for more details,
    but this basically just requires:
    * `additionalProperties` must be set to false for each object in the parameters
    * all fields in properties must be marked as required
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict)
        self.root_ref = schema.get('$ref')

    def walk(self) -> JsonSchema:
        # Note: OpenAI does not support anyOf at the root in strict mode
        # However, we don't need to check for it here because we ensure in pydantic_ai._utils.check_object_json_schema
        # that the root schema either has type 'object' or is recursive.
        result = super().walk()

        # For recursive models, we need to tweak the schema to make it compatible with strict mode.
        # Because the following should never change the semantics of the schema we apply it unconditionally.
        if self.root_ref is not None:
            result.pop('$ref', None)  # We replace references to the self.root_ref with just '#' in the transform method
            root_key = re.sub(r'^#/\$defs/', '', self.root_ref)
            result.update(self.defs.get(root_key) or {})

        return result

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove known irrelevant keys quickly
        for k in ('title', '$schema', 'discriminator'):
            schema.pop(k, None)

        # Handle "default" key smartly
        default = schema.get('default', _sentinel)
        if default is not _sentinel:
            if self.strict is True:
                schema.pop('default', None)
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        # Optimize $ref transformation
        schema_ref = schema.get('$ref')
        if schema_ref is not None:
            if schema_ref == self.root_ref:
                schema['$ref'] = '#'
            if len(schema) > 1:
                # Move $ref to anyOf as needed
                schema['anyOf'] = [{'$ref': schema.pop('$ref')}]

        # Bulk collect incompatible keys in a single pass
        present_keys = _STRICT_INCOMPATIBLE_KEYS_SET & schema.keys()
        if present_keys:
            incompatible_values = {k: schema[k] for k in present_keys}
        else:
            incompatible_values = {}

        # Handle incompatible keys
        if incompatible_values:
            description = schema.get('description')
            if self.strict is True:
                notes = [f'{key}={value}' for key, value in incompatible_values.items()]
                for key in incompatible_values:
                    schema.pop(key, None)
                notes_string = ', '.join(notes)
                # Faster description update
                if description:
                    schema['description'] = f'{description} ({notes_string})'
                else:
                    schema['description'] = notes_string
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        # Optimize openai/strict mode for oneOf
        if 'oneOf' in schema:
            if self.strict is True:
                schema['anyOf'] = schema.pop('oneOf')
            else:
                self.is_strict_compatible = False

        schema_type = schema.get('type')
        if schema_type == 'object':
            if self.strict is True:
                # additionalProperties are disallowed
                schema['additionalProperties'] = False
                # all properties are required
                if 'properties' not in schema:
                    schema['properties'] = {}
                schema['required'] = list(schema['properties'].keys())
            elif self.strict is None:
                # Avoid multiple lookups
                additional_props = schema.get('additionalProperties')
                properties = schema.get('properties')
                required = schema.get('required')
                if additional_props is not False or properties is None or required is None:
                    self.is_strict_compatible = False
                else:
                    # Only check missing keys quickly
                    required_set = set(required)
                    for k in properties:
                        if k not in required_set:
                            self.is_strict_compatible = False
                            break
        return schema


_STRICT_INCOMPATIBLE_KEYS_SET = set(_STRICT_INCOMPATIBLE_KEYS)
