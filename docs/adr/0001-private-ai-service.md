# ADR-0001: Private AI service behind the UpNext backend

## Status

Accepted on 2026-08-10.

## Decision

`upnext-ai` is a private FastAPI service. `upnext-be` remains the public gateway and
the authoritative owner of identities, permissions, subscriptions, recruitment data,
conversations and all business writes.

The frontend must never call this service. Internal requests require a short-lived,
environment-bound service JWT issued by the backend. AI output is untrusted until the
backend validates it; no model tool may directly write to the core database.

## Migration rule

Adopt a strangler migration. The backend keeps its Gemini adapter as a feature-flagged
fallback. First migrate only the provider port (`structured` and `stream`), then run
shadow/canary traffic. Retrieval, embeddings, screening and interviews move only after
their contract, evaluation data and rollback path exist.

## Consequences

- A temporary second deployable and CI pipeline are intentional costs.
- No direct shared core-database access is permitted for this service.
- Long-running work moves through the existing backend outbox before a separate broker
  is introduced.
