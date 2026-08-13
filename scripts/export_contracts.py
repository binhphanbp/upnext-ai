from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.contracts.embedding import EmbeddingRequest, EmbeddingResponse
from app.contracts.llm import (
    StructuredRequest,
    StructuredResponse,
    TextChunk,
    TextStreamRequest,
    UsageChunk,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "contracts" / "generated"


def documents() -> dict[str, dict[str, object]]:
    return {
        "embedding-request.schema.json": EmbeddingRequest.model_json_schema(by_alias=True),
        "embedding-response.schema.json": EmbeddingResponse.model_json_schema(by_alias=True),
        "llm-structured-request.schema.json": StructuredRequest.model_json_schema(by_alias=True),
        "llm-structured-response.schema.json": StructuredResponse.model_json_schema(by_alias=True),
        "llm-text-stream-request.schema.json": TextStreamRequest.model_json_schema(by_alias=True),
        "llm-text-chunk.schema.json": TextChunk.model_json_schema(by_alias=True),
        "llm-usage-chunk.schema.json": UsageChunk.model_json_schema(by_alias=True),
    }


def render(document: dict[str, object]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatch = False
    for filename, document in documents().items():
        path = OUTPUT / filename
        expected = render(document)
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual != expected:
            mismatch = True
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected, encoding="utf-8")
                print(f"generated {path.relative_to(ROOT)}")
            else:
                print(f"outdated {path.relative_to(ROOT)}")
    return 1 if mismatch and args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
