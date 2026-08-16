from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_grounded_provider
from app.api.routes.internal_llm import _provider_exception
from app.contracts.research import GroundedRequest, GroundedResponse, GroundedSource
from app.core.security import InternalPrincipal, require_grounded_principal
from app.providers.base import GroundedProvider, ProviderError

router = APIRouter(prefix="/internal/v1/research", tags=["internal-research"])


@router.post("/grounded", response_model=GroundedResponse, response_model_by_alias=True)
async def generate_grounded(
    request: GroundedRequest,
    _: InternalPrincipal = Depends(require_grounded_principal),
    provider: GroundedProvider = Depends(get_grounded_provider),
) -> GroundedResponse:
    """Answer a question against live web search and return what was consulted.

    Held apart from the structured routes and given its own scope because the
    cost and blast radius are different in kind: each call fans out into several
    external searches on a premium model, so a token minted for cheap in-context
    extraction must not be able to trigger it.

    The answer is returned verbatim with its citations rather than parsed here.
    Deciding whether the evidence is strong enough -- how many distinct sources
    are required, how confidence degrades when they are thin -- is a business
    rule belonging to the caller, and this service stays stateless about it.
    """

    try:
        answer = await provider.generate_grounded(
            system_instruction=request.system_instruction,
            prompt=request.prompt,
            temperature=request.temperature,
        )
    except ProviderError as error:
        raise _provider_exception(error) from error
    return GroundedResponse(
        text=answer.text,
        sources=[GroundedSource(title=title, url=url) for title, url in answer.sources],
        searchQueries=list(answer.search_queries),
        inputTokens=answer.input_tokens,
        outputTokens=answer.output_tokens,
        model=provider.grounded_model,
    )
