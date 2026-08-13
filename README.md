# upnext-ai

Private AI inference service for UpNext. It is **not** a public API and must only be
reachable by `upnext-be` on the private Docker network.

## Service boundary

`upnext-be` remains the source of truth for authentication, RBAC, subscriptions,
candidate/job/application data, conversations, audit records, and all business writes.
This service owns provider access, prompt/model execution, structured output validation,
streaming and private embeddings. Retrieval and evaluation remain later migration slices.

The embedding endpoint is deliberately narrow: `POST /internal/v1/embeddings`
requires an internal JWT with `embedding:invoke` scope and returns the frozen
`gemini-embedding-001:768:l2-v1` vector contract used by UpNext's pgvector indexes.
Changing that model, dimension, normalization or cache key requires a versioned
contract and an explicit re-index plan.

The frontend never calls this service directly. AI-proposed actions are only executed by
`upnext-be` after authorization and explicit user confirmation.

AI capabilities move here incrementally rather than through a big-bang rewrite. See the
[capability migration ADR](docs/adr/0002-capability-migration.md) for the current ownership
matrix, model-tier policy and rollout requirements.

## Local development

1. Copy `.env.example` to `.env` and set a unique `AI_INTERNAL_JWT_SECRET`.
2. Add `GEMINI_API_KEY` only when testing a real model call.
3. Create a Python 3.12 environment and install `.[dev]`.
4. Run `fastapi dev app/main.py`.

Useful commands:

```bash
python -m pytest
ruff check .
ruff format --check .
pyright
python scripts/export_contracts.py --check
```

`/health/live` proves the process is running. `/health/ready` additionally requires a
configured provider. Do not expose either internal LLM endpoint through Nginx.

## Internal protocol

Every `/internal/v1/*` request requires an HS256 JWT issued by `upnext-be`, with:

- `iss=upnext-be`, `aud=upnext-ai`, and the least-privilege scope required by the route:
  `llm:invoke`, `embedding:invoke`, or `job-post:extract`;
- a non-empty `sub`, `jti`, `iat` and `exp`;
- a maximum lifetime of 90 seconds by default;
- an `environment` claim matching the target deployment.

The contract is generated from Pydantic models in `contracts/generated/`. The BE adapter
has matching integration tests; this avoids maintaining unverified TypeScript/Python DTOs.

`POST /internal/v1/job-posts/extract` is the narrow multimodal capability for recruiter JD
imports. It accepts structured instructions, a response schema, and at most one private PDF
or image source (8 MiB). It does not store, log, or expose the source file.

## Staging rollout

Follow the [staging rollout runbook](docs/runbooks/staging-rollout.md). Deploy the service
privately first while the backend remains on its direct Gemini adapter; only then enable the
backend feature flag. This keeps a one-change rollback path available at all times.
