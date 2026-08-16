from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_llm_provider
from app.api.routes.internal_llm import _provider_exception
from app.contracts.company import CompanyLicenseExtractionRequest
from app.contracts.llm import StructuredResponse
from app.core.security import (
    InternalPrincipal,
    require_company_license_extraction_principal,
)
from app.providers.base import LlmProvider, ProviderError

router = APIRouter(prefix="/internal/v1/companies", tags=["internal-companies"])


@router.post("/license-extract", response_model=StructuredResponse, response_model_by_alias=True)
async def extract_company_license(
    request: CompanyLicenseExtractionRequest,
    _: InternalPrincipal = Depends(require_company_license_extraction_principal),
    provider: LlmProvider = Depends(get_llm_provider),
) -> StructuredResponse:
    """Read registration fields from a company's business licence document.

    Kept on its own route and scope rather than folded into job-post extraction:
    a token minted to read a JD must not also be able to read company
    registration documents, and the two capabilities should be able to be
    rolled out, rate limited and rolled back independently.
    """

    try:
        value, input_tokens, output_tokens = await provider.generate_structured_with_file(
            system_instruction=request.system_instruction,
            prompt=request.prompt,
            response_schema=request.response_schema,
            file=(request.file.mime_type, request.file.content()),
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
