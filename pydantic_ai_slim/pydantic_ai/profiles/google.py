from __future__ import annotations as _annotations

import warnings

from pydantic_ai.exceptions import UserError

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini which [supports](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)
    a subset of OpenAPI v3.0.3.

    Specifically:
    * gemini doesn't allow the `title` keyword to be set
    * gemini doesn't allow `$defs` — we need to inline the definitions where possible
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Note: we need to remove `additionalProperties: False` since it is currently mishandled by Gemini
        additional_properties = schema.pop('additionalProperties', None)
        if additional_properties:
            original_schema = {**schema, 'additionalProperties': additional_properties}
            warnings.warn(
                '`additionalProperties` is not supported by Gemini; it will be removed from the tool JSON schema.'
                f' Full schema: {self.schema}\n\n'
                f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                "If Google's APIs are updated to support this properly, please create an issue on the PydanticAI GitHub"
                ' and we will fix this behavior.',
                UserWarning,
            )

        # Batch pop all irrelevant fields in a single loop for minimal overhead
        # (this is a minimal gain/runtime, but for completeness and clean code)
        for k in (
            'title',
            'default',
            '$schema',
            'discriminator',
            'examples',
            'exclusiveMaximum',
            'exclusiveMinimum',
        ):
            schema.pop(k, None)

        if (const := schema.pop('const', None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]

        # Gemini only supports string enums, so we need to convert any enum values to strings.
        enum = schema.get('enum')
        if enum:
            schema['type'] = 'string'
            schema['enum'] = [str(val) for val in enum]

        type_ = schema.get('type')
        if 'oneOf' in schema and 'type' not in schema:  # pragma: no cover
            # This gets hit when we have a discriminated union
            # Gemini returns an API error in this case even though it says in its error message it shouldn't...
            # Changing the oneOf to an anyOf prevents the API error and I think is functionally equivalent
            schema['anyOf'] = schema.pop('oneOf')

        if type_ == 'string':
            fmt = schema.pop('format', None)
            if fmt:
                description = schema.get('description')
                if description:
                    schema['description'] = f'{description} (format: {fmt})'
                else:
                    schema['description'] = f'Format: {fmt}'

        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        if 'prefixItems' in schema:
            # prefixItems is not currently supported in Gemini, so we convert it to items for best compatibility
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')

            # OPTIMIZED: Use set to avoid O(N^2) list membership test; preserves order of prefix_items
            unique_items = [items] if items is not None else []
            seen = set()
            if items is not None:
                seen.add(self._item_hashable(items))
            for item in prefix_items:
                item_hash = self._item_hashable(item)
                if item_hash not in seen:
                    unique_items.append(item)
                    seen.add(item_hash)

            if len(unique_items) > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif len(unique_items) == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema

    @staticmethod
    def _item_hashable(item):
        # Helper: dicts are not hashable; convert to tuple of sorted items,
        # fallback to id if not dict, to allow set-based duplicate detection.
        if isinstance(item, dict):
            # Items in JSON schema dicts are also always str:Any
            return tuple(sorted(item.items()))
        try:
            hash(item)
            return item
        except TypeError:
            # Should not occur, but in case of nested unhashable types
            return id(item)
