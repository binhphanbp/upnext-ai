from __future__ import annotations

from pydantic import Field

from app.contracts.llm import StrictModel


class GroundedRequest(StrictModel):
    """Narrow internal contract for an answer that must cite live web sources."""

    system_instruction: str = Field(min_length=1, max_length=30_000, alias="systemInstruction")
    prompt: str = Field(min_length=1, max_length=50_000)
    temperature: float | None = Field(default=None, ge=0, le=2)


class GroundedSource(StrictModel):
    title: str
    url: str


class GroundedResponse(StrictModel):
    """Raw answer plus the evidence the provider actually consulted.

    The text is returned unparsed on purpose. Structured output cannot be
    combined with the search tool -- asking for a response schema makes the
    provider return empty grounding metadata, and the citations are the whole
    point of this capability. The caller owns the shape it asked for in the
    prompt, and owns deciding whether the evidence is strong enough to use.
    """

    text: str
    sources: list[GroundedSource]
    search_queries: list[str] = Field(alias="searchQueries")
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    model: str
