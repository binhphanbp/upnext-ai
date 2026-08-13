from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_llm_provider
from app.api.routes.internal_llm import _provider_exception
from app.contracts.job_post import JobPostExtractionRequest
from app.contracts.llm import StructuredResponse
from app.core.security import InternalPrincipal, require_job_post_extraction_principal
from app.providers.base import LlmProvider, ProviderError

router = APIRouter(prefix="/internal/v1/job-posts", tags=["internal-job-posts"])


@router.post("/extract", response_model=StructuredResponse, response_model_by_alias=True)
async def extract_job_post(
    request: JobPostExtractionRequest,
    _: InternalPrincipal = Depends(require_job_post_extraction_principal),
    provider: LlmProvider = Depends(get_llm_provider),
) -> StructuredResponse:
    """Extract a structured JD from text or a supported private source file."""

    try:
        value, input_tokens, output_tokens = await provider.generate_structured_with_file(
            system_instruction=request.system_instruction,
            prompt=request.prompt,
            response_schema=request.response_schema,
            file=(request.file.mime_type, request.file.content()) if request.file else None,
            temperature=request.temperature,
            model_tier=request.model_tier,
            execution_profile=request.execution_profile,
        )
    except ProviderError as error:
        raise _provider_exception(error) from error
    return StructuredResponse(
        value=value,
        inputTokens=input_tokens,
        outputTokens=output_tokens,
        model=provider.structured_model_for(request.model_tier),
    )
