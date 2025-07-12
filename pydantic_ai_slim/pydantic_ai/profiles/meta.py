from __future__ import annotations as _annotations

from dataclasses import fields
from functools import lru_cache

from ._json_schema import InlineDefsJsonSchemaTransformer


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


# --------------------------------------------------------------------------
# Efficient field name and object default lookups for update()


@lru_cache(maxsize=32)
def _fields_and_defaults(cls):
    flds = fields(cls)
    return (tuple(flds), {f.name for f in flds}, {f.name: f.default for f in flds})
