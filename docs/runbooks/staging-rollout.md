# Staging rollout: private AI service

## Scope and ownership

This runbook deploys `upnext-ai` as an **internal** service. It does not expose a public
route, change the frontend, or grant the AI service access to the core database. `upnext-be`
continues to authenticate users, enforce subscriptions and quotas, store chat history, and
perform every business write.

Do not run this against production until the staging acceptance checks below have passed.

## 1. Create and store the internal secret

Generate one random secret. Use the same value in the backend and AI service staging env
files; never commit it, reuse `JWT_ACCESS_SECRET`, or paste it into tickets/logs.

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set it in the deployment host only:

```dotenv
# env/ai.staging.env
AI_ENVIRONMENT=staging
AI_INTERNAL_JWT_SECRET=<generated-value>
GEMINI_API_KEY=<server-side-provider-key>

# env/backend.staging.env
AI_INTERNAL_JWT_SECRET=<same-generated-value>
AI_LLM_PROVIDER=gemini
AI_SERVICE_URL=http://ai-staging:8000
AI_SERVICE_TIMEOUT_MS=25000
AI_SERVICE_FALLBACK_TO_GEMINI=true
```

Keep `AI_LLM_PROVIDER=gemini` for the first deploy. This proves that introducing the
container cannot interrupt existing candidate chat.

Embedding migration is an independent switch. Deploy this service and its embedding
contract first, then deploy the compatible backend while keeping:

```dotenv
AI_EMBEDDING_PROVIDER=gemini
AI_EMBEDDING_FALLBACK_TO_GEMINI=true
```

After readiness and the internal contract smoke test pass, canary with
`AI_EMBEDDING_PROVIDER=upnext-ai`. The endpoint preserves
`gemini-embedding-001:768:l2-v1`, so valid cached vectors remain reusable and rollback
does not require re-indexing. Do not change the model, dimension, normalization or
cache key during this rollout.

## 2. Deploy the image privately

The infrastructure compose file must use the GHCR image and the `upnext-staging` network.
`ai-staging` must have `expose: 8000` only—never a host `ports` mapping and never an Nginx
location. Confirm GHCR credentials can pull the private package before the deployment.

After the normal image pull/deploy process, verify from the host:

```bash
docker compose -f compose/docker-compose.staging.yml ps ai-staging
docker inspect --format '{{.State.Health.Status}}' upnext-ai-staging
docker compose -f compose/docker-compose.staging.yml logs --tail=100 ai-staging
```

Expected result: the container is `healthy`; logs contain request metadata only and no prompt,
answer, bearer token, or API key.

## 3. Enable the backend gateway

Once the private health check is green, change only the backend environment setting:

```dotenv
AI_LLM_PROVIDER=upnext-ai
```

Restart the backend through the normal deployment workflow. Keep
`AI_SERVICE_FALLBACK_TO_GEMINI=true` during staging so a network timeout, 429 or 5xx from the
private service does not interrupt a candidate conversation. The backend must not fall back
for malformed structured output: that is a contract defect that needs investigation.

## 4. Acceptance checks

Perform each check with a non-production test account:

1. Existing AI chat responds through the normal frontend; browser network traffic targets only
   `upnext-be`, never `upnext-ai`.
2. A structured action (for example, job-search intent extraction) returns schema-valid output.
3. A streaming reply can complete and can be cancelled without keeping the request open.
4. An expired, wrong-audience, or wrong-environment internal token receives `401` from the AI
   service.
5. `docker compose ... config` contains no published AI port and no secret literal.
6. Backend and AI logs contain correlation IDs and status/latency, but no candidate CV content,
   prompts, model replies, credentials, or authorization headers.

Record the image digest, backend version, model names, and check time in the deployment log.

## Structured model tiers

The structured endpoint accepts only the controlled tiers `fast` and `quality`; callers never send
provider model names. `fast` is the backward-compatible default used by routing and classification.
`quality` is reserved for user-facing authoring workloads such as recruiter JD generation and maps
to `AI_QUALITY_STRUCTURED_MODEL` inside this service.

When rolling out a backend that sends `modelTier`, deploy and verify the compatible `upnext-ai`
image first, then deploy the backend. During rollback, roll the backend back before the AI service.
This order prevents an older strict Pydantic contract from rejecting a newer backend request.

Batch structured workloads use `executionProfile=batch` and
`AI_BATCH_STRUCTURED_TIMEOUT_SECONDS` (default 60 seconds). Deploy this additive AI contract
before enabling CV-screening traffic in the backend. Keep the interactive timeout short; do not
raise it globally to accommodate batch work.

## 5. Rollback and incident handling

For an AI-service incident, first set the backend back to the established direct provider and
restart it:

```dotenv
AI_LLM_PROVIDER=gemini
```

This is the primary rollback; do not change frontend code or expose the AI service to bypass
it. Preserve the failed request correlation ID and timestamps for investigation. Rotate
`AI_INTERNAL_JWT_SECRET` only by updating **both** services atomically, then restarting both;
otherwise every internal call will correctly fail closed.

If a secret or provider key may have been exposed, revoke it at the provider, replace it in the
deployment secret store, restart the affected service, and audit deployment logs/access.
