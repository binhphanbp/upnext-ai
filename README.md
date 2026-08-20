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
  `llm:invoke`, `embedding:invoke`, `job-post:extract`, `job-post:generate`,
  `company-license:extract`, or `research:grounded`;
- a non-empty `sub`, `jti`, `iat` and `exp`;
- a maximum lifetime of 90 seconds by default;
- an `environment` claim matching the target deployment.

The contract is generated from Pydantic models in `contracts/generated/`. The BE adapter
has matching integration tests; this avoids maintaining unverified TypeScript/Python DTOs.

`POST /internal/v1/job-posts/extract` is the narrow multimodal capability for recruiter JD
imports. It accepts structured instructions, a response schema, and at most one private PDF
or image source (8 MiB). It does not store, log, or expose the source file.

`POST /internal/v1/job-posts/generate` is the separate structured capability for generating
a recruiter JD from backend-prepared facts. It intentionally accepts one bounded prompt, not
conversation history or browser input, and requires the dedicated `job-post:generate` scope.

`POST /internal/v1/companies/license-extract` reads registration fields from a company's
business licence document. It has its own scope so that a token minted to read a JD cannot
also read company registration documents.

`POST /internal/v1/research/grounded` answers a question against live web search and returns
the answer together with the sources and search queries the model actually used. It is the
only route that reaches outside the model's own knowledge, and each call fans out into
several searches on a premium model, so it requires the dedicated `research:grounded` scope.
The answer text is returned unparsed: a response schema cannot be combined with the search
tool without emptying the grounding metadata, so the caller pins the shape in its prompt and
decides for itself whether the cited evidence is strong enough to use.

## Staging rollout

Follow the [staging rollout runbook](docs/runbooks/staging-rollout.md). Deploy the service
privately first while the backend remains on its direct Gemini adapter; only then enable the
backend feature flag. This keeps a one-change rollback path available at all times.

## Agent Tooling

This repo declares the [Superpowers](https://github.com/obra/superpowers) Claude
Code plugin in `.claude/settings.json`, but that file only records intent —
Claude Code does not auto-install a plugin just because a repo declares it. After
cloning, run once per machine:

```bash
claude plugin marketplace add obra/superpowers-marketplace
claude plugin install superpowers@superpowers-marketplace --scope project
```

Skip this and `claude plugin list` will show the plugin as `failed to load` inside
this repo. Not using Claude Code, or don't want the plugin? Nothing to do — it has
no effect on the build or runtime.
