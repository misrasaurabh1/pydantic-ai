from __future__ import annotations as _annotations

import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.exceptions import UserError

JsonSchema = dict[str, Any]


@dataclass(init=False)
class JsonSchemaTransformer(ABC):
    """Walks a JSON schema, applying transformations to it at each level.

    Note: We may eventually want to rework tools to build the JSON schema from the type directly, using a subclass of
    pydantic.json_schema.GenerateJsonSchema, rather than making use of this machinery.
    """

    def __init__(
        self,
        schema: JsonSchema,
        *,
        strict: bool | None = None,
        prefer_inlined_defs: bool = False,
        simplify_nullable_unions: bool = False,
    ):
        self.schema = schema
        self.strict = strict
        self.is_strict_compatible = True

        self.prefer_inlined_defs = prefer_inlined_defs
        self.simplify_nullable_unions = simplify_nullable_unions

        self.defs: dict[str, JsonSchema] = self.schema.get('$defs', {})
        self.refs_stack: list[str] = []
        self.recursive_refs = set[str]()

    @abstractmethod
    def transform(self, schema: JsonSchema) -> JsonSchema:
        """Make changes to the schema."""
        return schema

    def walk(self) -> JsonSchema:
        schema = deepcopy(self.schema)

        # First, handle everything but $defs:
        schema.pop('$defs', None)
        handled = self._handle(schema)

        if not self.prefer_inlined_defs and self.defs:
            handled['$defs'] = {k: self._handle(v) for k, v in self.defs.items()}

        elif self.recursive_refs:  # pragma: no cover
            # If we are preferring inlined defs and there are recursive refs, we _have_ to use a $defs+$ref structure
            # We try to use whatever the original root key was, but if it is already in use,
            # we modify it to avoid collisions.
            defs = {key: self.defs[key] for key in self.recursive_refs}
            root_ref = self.schema.get('$ref')
            root_key = None if root_ref is None else re.sub(r'^#/\$defs/', '', root_ref)
            if root_key is None:
                root_key = self.schema.get('title', 'root')
                while root_key in defs:
                    # Modify the root key until it is not already in use
                    root_key = f'{root_key}_root'

            defs[root_key] = handled
            return {'$defs': defs, '$ref': f'#/$defs/{root_key}'}

        return handled

    def _handle(self, schema: JsonSchema) -> JsonSchema:
        # Avoid regex, use efficient prefix-check and slicing for $ref handling
        nested_refs = 0
        if self.prefer_inlined_defs:
            ref = schema.get('$ref')
            while ref is not None:
                prefix = '#/$defs/'
                if ref.startswith(prefix):
                    key = ref[len(prefix) :]
                else:
                    # Fallback to regex only if prefix not found, which is rare
                    key = re.sub(r'^#/\$defs/', '', ref)
                # Use a set for faster "already recursing" checks
                if key in self.refs_stack:
                    self.recursive_refs.add(key)
                    break  # recursive ref can't be unpacked
                self.refs_stack.append(key)
                nested_refs += 1

                def_schema = self.defs.get(key)
                if def_schema is None:  # pragma: no cover
                    raise UserError(f'Could not find $ref definition for {key}')
                schema = def_schema
                ref = schema.get('$ref')

        type_ = schema.get('type')
        if type_ == 'object':
            schema = self._handle_object(schema)
        elif type_ == 'array':
            schema = self._handle_array(schema)
        elif type_ is None:
            schema = self._handle_union(schema, 'anyOf')
            schema = self._handle_union(schema, 'oneOf')

        schema = self.transform(schema)

        if nested_refs > 0:
            del self.refs_stack[-nested_refs:]

        return schema

    def _handle_object(self, schema: JsonSchema) -> JsonSchema:
        # Properties
        properties = schema.get('properties')
        if properties:
            handled_properties = None
            # Only create new dict if at least one property changes
            for key, value in properties.items():
                handled_value = self._handle(value)
                if handled_value is not value:
                    if handled_properties is None:
                        handled_properties = properties.copy()
                    handled_properties[key] = handled_value
            if handled_properties is not None:
                schema['properties'] = handled_properties

        # AdditionalProperties
        additional_properties = schema.get('additionalProperties')
        if additional_properties is not None and not isinstance(additional_properties, bool):
            handled = self._handle(additional_properties)
            if handled is not additional_properties:
                schema['additionalProperties'] = handled

        # PatternProperties
        pattern_properties = schema.get('patternProperties')
        if pattern_properties is not None:
            handled_pattern_properties = None
            for key, value in pattern_properties.items():
                handled_value = self._handle(value)
                if handled_value is not value:
                    if handled_pattern_properties is None:
                        handled_pattern_properties = pattern_properties.copy()
                    handled_pattern_properties[key] = handled_value
            if handled_pattern_properties is not None:
                schema['patternProperties'] = handled_pattern_properties

        return schema

    def _handle_array(self, schema: JsonSchema) -> JsonSchema:
        prefix_items = schema.get('prefixItems')
        if prefix_items:
            updated_items = None
            for i, item in enumerate(prefix_items):
                handled_item = self._handle(item)
                if handled_item is not item:
                    if updated_items is None:
                        updated_items = list(prefix_items)
                    updated_items[i] = handled_item
            if updated_items is not None:
                schema['prefixItems'] = updated_items

        items = schema.get('items')
        if items:
            handled_items = self._handle(items)
            if handled_items is not items:
                schema['items'] = handled_items

        return schema

    def _handle_union(self, schema: JsonSchema, union_kind: Literal['anyOf', 'oneOf']) -> JsonSchema:
        members = schema.get(union_kind)
        if not members:
            return schema

        # No need to copy schema unless something changes
        handled = None
        for i, member in enumerate(members):
            handled_member = self._handle(member)
            if handled_member is not member:
                if handled is None:
                    handled = list(members)
                handled[i] = handled_member
        final_handled = handled if handled is not None else members

        # convert nullable unions to nullable types
        if self.simplify_nullable_unions:
            simplified_handled = self._simplify_nullable_union(final_handled)
            if simplified_handled is not final_handled:
                final_handled = simplified_handled

        if len(final_handled) == 1:
            return final_handled[0]

        # Only copy schema if any change occurred
        if handled is not None or (self.simplify_nullable_unions and simplified_handled is not handled):
            schema = schema.copy()
            schema[union_kind] = final_handled

        return schema

    @staticmethod
    def _simplify_nullable_union(cases: list[JsonSchema]) -> list[JsonSchema]:
        # TODO: Should we move this to relevant subclasses? Or is it worth keeping here to make reuse easier?
        if len(cases) == 2 and {'type': 'null'} in cases:
            # Find the non-null schema
            non_null_schema = next(
                (item for item in cases if item != {'type': 'null'}),
                None,
            )
            if non_null_schema:
                # Create a new schema based on the non-null part, mark as nullable
                new_schema = deepcopy(non_null_schema)
                new_schema['nullable'] = True
                return [new_schema]
            else:  # pragma: no cover
                # they are both null, so just return one of them
                return [cases[0]]

        return cases


class InlineDefsJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema to inline $defs."""

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema
