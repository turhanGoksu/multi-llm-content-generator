# Multi-LLM Content Generator

A FastAPI service that generates structured ad copy (titles, descriptions, keywords) for a small business. For each request it calls **Google Gemini** and **Groq** at the same time, validates each provider's output on its own, stores both outcomes in PostgreSQL, and lets you compare them side by side.

It rebuilds, from scratch, the kind of LLM ad-copy generator I worked on during an internship. It is designed around specific bugs I found when I later reviewed that old codebase. No code or content from that project is reused. The [last section](#lessons-from-a-past-project) maps each of those bugs to the design decision here that prevents it.

Only free-tier APIs are used.

## Example

```http
POST /generate
{"description": "Kadıköy'de ekşi mayalı ekmek yapan küçük bir mahalle fırını.",
 "category": "bakery", "tone": "friendly"}
```

Abridged response from a real run. The copy comes back in the language of the description:

```json
{
  "id": "…",
  "results": {
    "gemini": {
      "model": "gemini-3.5-flash-lite", "status": "success", "latency_ms": 2242,
      "content": {
        "titles": ["Kadıköy'ün Sıcak Mahalle Fırını", "…"],
        "descriptions": ["Katkısız ve doğal ekşi mayalı ekmeklerimizle kahvaltılarınıza lezzet katıyoruz. …"],
        "keywords": ["kadıköy fırın", "ekşi mayalı ekmek", "…"]
      },
      "warnings": []
    },
    "groq": {
      "model": "openai/gpt-oss-20b", "status": "success", "latency_ms": 821,
      "content": { "…": "…" }
    }
  }
}
```

`GET /compare/{id}` returns the same record, read back from the database.

## Architecture

```mermaid
flowchart LR
    A[POST /generate] --> B[build_prompt<br/>config/categories.yaml]
    B --> C{asyncio.gather}
    C --> D[GeminiClient.generate]
    C --> E[GroqClient.generate]
    D --> F[validate_output]
    E --> G[validate_output]
    F --> H[(PostgreSQL<br/>one row per provider)]
    G --> H
    H --> I[GET /compare/id]
```

```
app/
  main.py            FastAPI app, lifespan, endpoints
  config.py          Typed settings from environment variables
  prompts.py         Prompt builder driven by config/categories.yaml
  schemas.py         Output schema, status model, per-provider validation
  db.py              SQLAlchemy models and persistence
  providers/
    base.py          LLMProvider: the shared generate() and ProviderResult
    gemini_client.py Gemini request/response/429 format
    groq_client.py   Groq request/response/429 format
    rate_limit.py    Per-provider sliding-window limiter + 429 cooldown
config/categories.yaml  Business categories and tones (data, not code)
tests/                  49 tests; providers are always mocked
```

**Providers share one interface.** `LLMProvider.generate(prompt) -> ProviderResult` is written once in the base class. Each provider only defines how to build its request, where the text sits in its response, and how it reports a retry delay. Adding a provider means adding one subclass and one entry in the provider list. The endpoint and the database schema stay the same.

## Why `async def` here

Most of a `/generate` request is spent **waiting on two remote HTTP APIs**. That is I/O-bound work. While one `await` waits for Gemini, the event loop can wait for Groq and serve other requests. `asyncio.gather` runs both calls concurrently, so a request takes about as long as the **slowest** provider, not the sum of both. In the run above that was ~2.2 s instead of ~3.1 s. The gap grows with each provider added.

In an earlier project, a model-inference `/predict` endpoint was deliberately a plain `def`. Model inference is **CPU-bound**. Inside `async def` it would block the single event-loop thread, and every other request, including health checks, would stall until it finished. FastAPI runs plain `def` endpoints in a threadpool instead. The rule I follow: use `async def` when the endpoint waits on I/O, and plain `def` (or a worker) when it computes.

The database layer is async too (SQLAlchemy + asyncpg), for the same reason. A synchronous driver called inside `async def` would block the loop just like CPU-bound work.

## Status model

Every provider gets exactly one status per request. Failures are recorded, never dropped.

| Status | Meaning |
|---|---|
| `success` | Valid JSON matching the schema; every field has at least one item |
| `partial` | Some content, but at least one field has no valid items (e.g. all descriptions exceeded the length limit). `warnings` says which. |
| `empty` | The call succeeded, but the model returned no content |
| `invalid_output` | Malformed JSON or a schema mismatch; `raw_text` is kept for debugging |
| `rate_limited` | A 429 from the provider (`rate_limit_source: provider`) or refused by our limiter (`local`), with `retry_after_seconds` when known |
| `error` | Timeout, network error, non-2xx response, unexpected response shape, or a bug in provider code |

`POST /generate` returns `201` whenever the request itself is valid, even if every provider failed. The outcome is in each provider's `status`.

## Rate limiting

Free-tier limits differ per provider and per model, and they change over time. So they are configured in `.env` (`GEMINI_RPM`, `GROQ_RPM`), next to the model names.

- **Before sending:** each provider has its own sliding-window limiter. If its per-minute budget is used up, the request is **not sent** and the provider is reported as `rate_limited` / `local` with the wait time. The limiter never sleeps. A blind `time.sleep` inside `async def` would freeze the whole server, and one shared delay can't fit two providers with different limits.
- **On a 429:** each client reads its provider's own retry hint. Groq sends a `Retry-After` header. Gemini sends none; its delay is in the error body (`google.rpc.RetryInfo.retryDelay`). That provider then enters a cooldown, so later requests skip it without spending quota, while the other provider keeps working.
- **No automatic retry inside a request.** If Groq says "retry in 40 s", waiting would hold a finished Gemini result hostage and risk proxy timeouts. The response comes back immediately with Groq's `retry_after_seconds`, and the client decides.

## Running it

Requires Docker. Get free API keys from [Google AI Studio](https://aistudio.google.com/apikey) and [Groq](https://console.groq.com/keys).

```bash
cp .env.example .env      # then fill in your keys, limits and a DB password
docker compose up -d --build
open http://localhost:8000/docs
```

`db` must pass its healthcheck before `api` starts, because the app creates its tables on startup. Postgres data lives in the `pgdata` volume. `docker compose down -v` deletes it.

### Tests

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .

# Unit tests only (API tests skip without a database):
pytest -ra

# Full suite against a throwaway Postgres:
docker run -d --rm --name test-db -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=test_db postgres:17-alpine
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/test_db pytest -ra
```

Every provider call in the tests goes through `httpx.MockTransport`, so tests need no API keys. They are also deterministic: a real API can't be made to return a 429 or malformed JSON on demand, and real LLM output changes from run to run. CI (GitHub Actions) runs lint, the full test suite against a Postgres service container, and a Docker build. In CI, a missing test database **fails** the run instead of quietly skipping the API tests.

## Known limitations

- The local limiter tracks **requests per minute** only. Daily and token limits are handled when the provider reports them, through the 429 path.
- Limiter state is **per process**. With several workers, each one has its own budget, so set limits accordingly or share state (e.g. in Redis).
- For **daily** quota exhaustion, Gemini's `retryDelay` can be much shorter than the real reset time. The value is reported as given, and not relied on for automatic retries.
- Tables are created with `create_all`, which **never alters existing tables**. A schema change needs a fresh database or a migration tool such as Alembic.
- Item **counts** (e.g. 5 titles) are requested in the prompt but not enforced; four good titles are still usable output. **Lengths** are enforced.
- LLMs count characters poorly. The prompt asks for 80% of each limit to leave a margin, and items still over the limit are dropped with a warning.

## Lessons from a past project

I built an LLM-based ad copy generator during an internship. Reviewing that code later, I found the bugs below. This project was designed so that none of them can recur. Each row gives the failure pattern, not the original code.

| # | Old behavior | How this project avoids it |
|---|---|---|
| 1 | **Positional pairing.** `zip(id_list, name_list)` matched two lists by position. When one list was missing an item, everything after it shifted, and content was saved under the wrong ID with no error. `zip` also stops silently at the shorter list. | Related data is paired **by key, never by position**. Categories are a mapping keyed by ID ([`categories.yaml`](config/categories.yaml)). Provider results are keyed by the name each result carries (`{r.provider: … for r in results}`). DB rows are unique per `(generation_id, provider)`. |
| 2 | **Hardcoded "all items".** `range(1, 430)` defined the full set of categories, so anything added later was silently never processed. | "All X" always comes from the real source. Valid categories and tones come from the YAML, and error messages list them from there. Providers are iterated from a list. The DB stores **one row per provider**, so adding one needs no schema change. A test adds a category through config alone. |
| 3 | **One bad item lost the whole batch.** One malformed LLM response became `None`, the merge step over all 20 categories threw, and the 19 successful results were lost with it. | Provider failures are **returned as data, not raised as exceptions**, and **each provider's output is validated separately**, so one provider's bad output cannot affect another's. `asyncio.gather` always gets two results. An extra guard turns even an unexpected bug into an `error` result. The same idea works inside one output: an over-length title is dropped on its own instead of invalidating the response. |
| 4 | **Silent failure looked like success.** A rate-limited category was saved as `[]`, identical to "the model returned nothing". | Six explicit statuses. `empty`, `rate_limited` and `invalid_output` are distinct and stored with `http_status`, `error_message`, `latency_ms`, `retry_after_seconds` and `rate_limit_source`. Missing content is SQL `NULL` (not JSON `null`), so `WHERE content IS NULL` finds it. In CI, a missing test database fails the run instead of skipping tests. |
| 5 | **Hardcoded dated model name**, which later became unavailable. | Model names live only in `.env`, with no default in code. This happened again while building: the model initially planned (`gemini-2.0-flash`) had been shut down, and Llama models had left Groq's free plan. Both were fixed by changing `.env`. Model-specific options (`GROQ_REASONING_EFFORT`) are optional, so switching models can't break requests. |
| 6 | **Docstring didn't match behavior.** A parameter was documented as a character limit but actually limited tokens. | Limits are named and documented by their real unit. `MAX_TITLE_CHARS` is checked with `len()` (characters, tested with multi-byte text). `GROQ_MAX_COMPLETION_TOKENS` is documented as tokens covering the model's reasoning plus its answer, a distinction that came from a real failure. The Gemini retry-delay caveat is written down, not glossed over. |
| 7 | **Duplicate implementations** of the same function in two files, plus dead code. | `generate()` exists once, in the base class; providers supply only what differs. When PostgreSQL replaced the temporary in-memory store, the store was deleted, not left beside it. |
| 8 | **Order-losing dedup** with `set()`. | `dedupe_preserving_order()` keeps the first occurrence in order, which matters because models list the most relevant keywords first. Keywords are compared case-insensitively. |
| 9 | **The same constant in two places** (config and prompt text), free to drift apart. | Length limits are defined once in [`schemas.py`](app/schemas.py). The prompt's targets are **derived** from them. DB credentials are defined once, and compose builds `DATABASE_URL` from them. CI takes tool versions from `requirements-dev.txt`. |
