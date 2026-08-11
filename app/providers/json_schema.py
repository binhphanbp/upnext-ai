"""Compatibility helpers for the internal structured-output contract.

The backend's existing Gemini-oriented contract uses upper-case ``type`` values
and ``nullable``.  The Google Gen AI SDK's ``response_json_schema`` parameter,
however, requires standard JSON Schema.  Keep this adaptation at the provider
edge so internal callers do not become coupled to one SDK's representation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_TYPE_ALIASES = {
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "OBJECT": "object",
    "ARRAY": "array",
    "NULL": "null",
}

_SCHEMA_MAP_KEYS = {
    "$defs",
    "definitions",
    "properties",
    "patternProperties",
    "dependentSchemas",
}
_SCHEMA_VALUE_KEYS = {
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_SCHEMA_LIST_KEYS = {"allOf", "anyOf", "oneOf", "prefixItems"}


def normalize_response_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a standard JSON Schema copy suitable for ``response_json_schema``.

    The input is never mutated.  Unsupported type names are rejected before a
    provider request is made, making an internal contract mistake diagnosable
    instead of surfacing as an opaque Gemini HTTP 400.
    """

    return _normalize_node(schema)


def _normalize_node(node: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    nullable = node.get("nullable", False)
    if not isinstance(nullable, bool):
        raise ValueError("JSON Schema nullable must be a boolean")

    for key, value in node.items():
        if key == "nullable":
            continue
        if key == "type":
            normalized[key] = _normalize_type(value)
        elif key in _SCHEMA_MAP_KEYS:
            if not isinstance(value, dict):
                raise ValueError(f"JSON Schema {key} must be an object")
            normalized[key] = {
                name: _normalize_node(child)
                if isinstance(child, dict)
                else deepcopy(child)
                for name, child in value.items()
            }
        elif key in _SCHEMA_VALUE_KEYS and isinstance(value, dict):
            normalized[key] = _normalize_node(value)
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            normalized[key] = [
                _normalize_node(item) if isinstance(item, dict) else deepcopy(item)
                for item in value
            ]
        else:
            normalized[key] = deepcopy(value)

    if nullable:
        _make_nullable(normalized)
    return normalized


def _normalize_type(value: Any) -> str | list[str]:
    if isinstance(value, str):
        return _normalize_type_name(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [_normalize_type_name(item) for item in value]
    raise ValueError("JSON Schema type must be a string or an array of strings")


def _normalize_type_name(value: str) -> str:
    normalized = _TYPE_ALIASES.get(value, value.lower())
    if normalized not in {"string", "number", "integer", "boolean", "object", "array", "null"}:
        raise ValueError(f"Unsupported JSON Schema type: {value}")
    return normalized


def _make_nullable(schema: dict[str, Any]) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema["type"] = [schema_type, "null"] if schema_type != "null" else "null"
        return
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            schema["type"] = [*schema_type, "null"]
        return

    # ``nullable`` without a type is uncommon, but converting it to a standard
    # union retains its intended semantics rather than silently dropping it.
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        schema["anyOf"] = [*any_of, {"type": "null"}]
    else:
        schema["anyOf"] = [{"type": "null"}]
