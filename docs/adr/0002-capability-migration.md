# ADR-0002: Migrate AI capabilities in reversible slices

## Status

Accepted on 2026-08-13.

## Context

UpNext already has AI-assisted features in `upnext-be`. Moving every provider call to a
new service in one release would combine contract, model-quality, multimodal and rollout
risk. It would also make a provider outage indistinguishable from a business-logic bug.

## Decision

Migrate one capability at a time through the existing backend provider port. The backend
continues to own authorization, source data, validation, persistence, quotas and audit
records. `upnext-ai` owns provider credentials, model selection and inference only.

The internal structured endpoint accepts a controlled model tier, not a provider model
name:

- `fast`: routing and low-latency structured tasks;
- `quality`: user-facing generation where output quality is more important than latency.

The service maps each tier to an environment-controlled model. This prevents callers from
bypassing cost and quality policy while allowing models to change without a BE release.

## Capability matrix

| Capability | Current path after this change | Next migration condition |
| --- | --- | --- |
| Candidate Copilot | BE gateway to `upnext-ai` | Add evaluations and canary telemetry |
| Generate or optimize a JD from form data | BE gateway to `upnext-ai`, `quality` tier | Monitor quality, latency and fallback rate |
| Import a JD from PDF/DOCX | Transitional direct BE multimodal call | Add an authenticated binary/multimodal contract |
| Salary research | Transitional direct BE call | Define citation, freshness and cache policy |
| CV screening and embeddings | Transitional direct BE call | Version embeddings, evaluation set and re-index plan |
| Company-license extraction | Transitional direct BE multimodal call | Add document safety and extraction evaluation |

“Transitional direct” is intentional and observable; it is not considered migrated.

## Rollout and rollback

The additive `upnext-ai` contract must be deployed before the backend begins sending
`modelTier`. Existing requests remain compatible because the omitted tier defaults to
`fast`. During rollback, return the backend to its direct provider first, then roll back
`upnext-ai` only after traffic has drained.

Each later capability requires its own contract tests, representative evaluation data,
feature flag or fallback, and an explicit rollback path before traffic is moved.
