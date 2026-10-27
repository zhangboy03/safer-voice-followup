from __future__ import annotations

STRICT_JSON_SCHEMA_MODELS = frozenset(
    {
        "qwen3.5-plus",
        "qwen3.5-plus-latest",
        "qwen3.7-plus",
        "qwen3.7-plus-latest",
        "qwen3.7-max",
        "qwen3.7-max-latest",
        "qwen3.6-plus",
        "qwen3.6-plus-latest",
        "qwen3.6-plus-2026-04-02",
        "qwen3.6-flash",
        "qwen3.6-flash-latest",
        "qwen3.6-flash-2026-04-16",
        "qwen-plus-2025-12-01",
    }
)


def build_response_format(*, model: str, schema_name: str, schema: dict) -> dict:
    if model in STRICT_JSON_SCHEMA_MODELS:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    return {"type": "json_object"}
