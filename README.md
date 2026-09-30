# DOTMappers Ticket AI

End-to-end AI system for the DOTMappers IT Pvt. Ltd. AI Intern Technical Assessment.

The assessment asks for a Python-only system that ingests a 500-row support-ticket CSV, answers natural-language questions with an LLM, detects anomalies, exposes a REST API, and provides a minimal UI. It also specifies zero-cost execution through a free-tier or locally run model. This repository implements all of those pieces with an in-memory SQLite data layer, Pandas, FastAPI, and Streamlit. The supplied assessment explicitly calls for NL questions such as ticket counts, agent performance, unresolved Critical tickets, average ratings, and resolution-time anomalies.

## What is included

```text
dotmappers-ticket-ai/
├── data/
│   └── support_tickets.csv
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── anomaly_detector.py
│   ├── llm_engine.py
│   └── api.py
├── api.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── run.sh
└── README.md
```

`api.py` at the repository root is a thin launcher so both `uvicorn api:app` and `uvicorn src.api:app` are convenient. The implementation lives in `src/api.py`.

## 1. Architecture

```text
                         ┌───────────────────────────┐
                         │ support_tickets.csv       │
                         │ 500 rows / 10 columns     │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │ database.py               │
                         │ validation + Pandas       │
                         │ typed in-memory SQLite    │
                         └─────────────┬─────────────┘
                                       │
                   ┌───────────────────┴──────────────────┐
                   │                                      │
                   ▼                                      ▼
        ┌──────────────────────┐               ┌──────────────────────┐
        │ llm_engine.py        │               │ anomaly_detector.py  │
        │ NL → safe SQL        │               │ deterministic rules  │
        │ Groq / Ollama / Mock │               │ IQR / stale / rating │
        └──────────┬───────────┘               └──────────┬───────────┘
                   │                                      │
                   └───────────────────┬──────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │ FastAPI                   │
                         │ /health /query /anomalies│
                         └─────────────┬─────────────┘
                                       │
                          server-side HTTP requests
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │ Streamlit UI               │
                         │ Query / Anomalies / Raw    │
                         └───────────────────────────┘
```

### Design choices

**Pandas + SQLite:** Pandas handles input normalization and statistical calculations; SQLite provides a compact relational query layer so the LLM can generate standard SQL. The database is intentionally in-memory because the assessment requests it and the dataset is small.

**LLM only for language-to-query translation:** The LLM converts an English question into a constrained SQL statement. The application then executes that SQL locally and builds the final response summary deterministically. This reduces token usage and avoids asking the model to hallucinate numeric answers.

**Read-only enforcement at two levels:**

1. SQL text is validated to allow one `SELECT`/`WITH` statement and reject DML/DDL keywords and comments.
2. SQLite's authorizer is configured so only reads from the `tickets` table and normal function calls are allowed. Database writes, PRAGMA calls, attached databases, transactions, and reads from other tables are denied by the engine itself.

**Provider fallback:** `auto` uses Groq when `GROQ_API_KEY` is present, then Ollama when it is reachable, then a deterministic mock translator. The mock supports the assessment's sample-style questions, so the project still runs without an API key.

## 2. Setup

### Linux / macOS / WSL / Git Bash

```bash
git clone <your-repository-url>
cd dotmappers-ticket-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add `GROQ_API_KEY` to `.env` for Groq. You can leave it empty and run in Ollama or mock mode.

Start both services with one command:

```bash
bash run.sh
```

Then open:

- Streamlit: `http://127.0.0.1:8501`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### Windows PowerShell

The repository's `run.sh` is designed for Bash, WSL, or Git Bash. In native PowerShell, start the two services in separate terminals:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

and:

```powershell
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## 3. Environment variables

See `.env.example` for the complete list.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `auto`, `groq`, `ollama`, or `mock` | `auto` |
| `GROQ_API_KEY` | Groq authentication | empty |
| `GROQ_MODEL` | Groq model ID | `llama-3.3-70b-versatile` |
| `OLLAMA_BASE_URL` | Local Ollama host | `http://localhost:11434` |
| `OLLAMA_MODEL` | Local model name | `llama3.2:3b` |
| `MAX_QUERY_ROWS` | Max response rows returned | `200` |
| `TICKETS_CSV_PATH` | Optional CSV override | project dataset |
| `API_BASE_URL` | Streamlit backend URL | `http://127.0.0.1:8000` |
| `API_PORT` | FastAPI port | `8000` |
| `STREAMLIT_PORT` | Streamlit port | `8501` |
| `ANOMALY_REFERENCE_AT` | Optional explicit age reference | dataset max timestamp |

## 4. LLM model selection

The assessment's requested Groq target is `llama-3.3-70b-versatile`; the code keeps this as the default configurable model. Groq's documentation currently lists that model among its production model IDs, but its deprecation documentation also contains a 2026 deprecation notice. Because model availability can change, the application does not hard-wire a second paid provider: when Groq fails, it falls back to local Ollama and then mock mode. Change `GROQ_MODEL` in `.env` to any model available to your Groq project.

The prompt deliberately uses low temperature and a JSON contract so the LLM returns a small machine-readable object. The database remains the source of truth for actual numeric results.

## 5. Anomaly detection logic

### A. Statistical outliers in resolution time

Only non-null `resolution_time_hrs` values are considered. The detector computes:

```text
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
```

Tickets with resolution time greater than the upper bound are flagged.

### B. Open/Escalated High & Critical tickets older than 24h

A ticket is flagged when:

```text
status in {Open, Escalated}
AND priority in {High, Critical}
AND age_hours > 24
```

For the fixed assessment dataset, the default reference time is the latest `created_at` value in the data. That makes the anomaly result deterministic instead of depending on the evaluator's wall clock. `ANOMALY_REFERENCE_AT` can be set when live-age semantics are desired.

### C. Low-rating clusters

The detector groups rated tickets by **agent-week** and **category-week**. A cluster needs at least 3 rated tickets and is flagged when either:

- at least 50% of ratings are 1-2, or
- the average rating is <= 2.5.

Null ratings are excluded from these calculations.

## 6. REST API

### `GET /health`

Returns application status, row count, dataset reference timestamp, and the configured LLM fallback chain.

### `POST /query`

Request:

```json
{
  "question": "How many tickets are currently open?"
}
```

Response shape:

```json
{
  "success": true,
  "question": "How many tickets are currently open?",
  "sql": "SELECT COUNT(*) AS open_tickets FROM tickets WHERE status = 'Open'",
  "data": [
    {"open_tickets": 111}
  ],
  "row_count": 1,
  "summary": "Open tickets: 111.",
  "provider": "mock",
  "model": null,
  "rationale": "Counts tickets currently marked Open."
}
```

The SQL shown above is the deterministic mock-mode result for the supplied dataset; with an LLM provider enabled, the generated query may differ while producing the same data answer.

### `GET /anomalies`

Returns the three detector payloads, their counts, thresholds, and flagged records.

## 7. Sample queries and expected answers from the supplied CSV

The included `support_tickets.csv` contains 500 rows. The following values were computed from that exact file when this project was generated:

| Query | Expected result |
|---|---|
| How many tickets are currently open? | **111** |
| Which agent resolved the most tickets this month? | **AGT-01 with 16 resolved tickets** in March 2024, the dataset's latest month |
| What is the average customer rating for Technical category tickets? | **3.74** |
| Show me all Critical tickets not resolved within 12 hours. | **34 tickets** where status is not Resolved and resolution time is NULL or > 12h |
| Are there anomalies in resolution times? | The IQR detector uses an upper bound of about **48.15 hours** and flags **21** tickets |

These are examples, not hard-coded responses. The API recomputes the answers from SQLite every request.

## 8. Run curl examples

Open tickets:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How many tickets are currently open?"}'
```

Anomalies:

```bash
curl http://127.0.0.1:8000/anomalies
```

## 9. Error handling and security

- Missing CSV or invalid required columns stop startup with a clear validation error.
- Invalid dates, negative times, duplicate ticket IDs, and out-of-range customer ratings are rejected.
- `resolution_time_hrs` and `customer_rating` may safely be NULL and are preserved as SQL NULL.
- SQL comments and multi-statement requests are rejected.
- DML/DDL keywords are rejected before execution.
- SQLite authorizer enforcement blocks database writes even if a generated query bypasses the text checks.
- LLM providers can fail independently; provider errors trigger the next fallback provider.
- A failed SQL query gets one repair attempt from the same provider before the engine falls through to the next provider.
- The API limits returned rows to avoid accidentally rendering unbounded result sets.

## 10. Limitations

1. **In-memory SQLite is process-local.** Running multiple Uvicorn workers would create separate datasets. Use a file-backed SQLite database or PostgreSQL for multi-worker deployment.
2. **LLM SQL accuracy depends on the chosen model.** Prompt constraints and the SQLite authorizer reduce risk, but semantic misunderstandings are still possible.
3. **Mock mode has intentionally limited language coverage.** It is a zero-cost fallback, not a replacement for the LLM.
4. **The anomaly thresholds are intentionally simple.** Production support operations would likely use seasonality, SLA-specific thresholds, agent workload normalization, and alert suppression/deduplication.
5. **No authentication is included.** The assessment asks for a local technical prototype; a production deployment should add authentication, authorization, rate limiting, audit logging, and HTTPS.

## 11. Future scaling

A natural next step would be:

```text
SQLite memory
   ↓
PostgreSQL / warehouse
   ↓
Pydantic query planner + SQL AST validation
   ↓
LLM gateway with retries, tracing, cost controls
   ↓
Redis/cache for repeated questions
   ↓
Prometheus/OpenTelemetry metrics
   ↓
Containerized API + independently deployed Streamlit/React UI
```

For a larger dataset, separate the analytics layer from the transactional ticket store, pre-compute operational metrics, and move anomaly detection to scheduled jobs. LLM calls should remain narrowly scoped to intent understanding and query planning, with deterministic execution and authorization outside the model.

## 12. Walkthrough points for the technical interview

Be ready to explain these trade-offs:

- Why the LLM generates SQL instead of the answer itself.
- Why SQLite authorizer enforcement is stronger than a keyword blacklist alone.
- Why deterministic anomaly detection is separate from the LLM.
- Why dataset-relative time is used for a fixed historical CSV.
- How the project behaves with no API key.
- How you would move from 500 rows to millions of tickets.
- How you would add authentication, observability, caching, and model governance.

## 13. Assessment alignment

The assessment asks for a working system, REST endpoints, a UI, a README, a reproducible requirements file, and a single-command startup path. This repository provides `/health`, `/query`, `/anomalies`, a Streamlit UI with the requested three dashboard tabs, `requirements.txt`, and `run.sh`.

