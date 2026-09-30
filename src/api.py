"""FastAPI application for the DOTMappers Ticket AI assessment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .anomaly_detector import AnomalyDetector
from .database import DEFAULT_CSV_PATH, TicketDatabase
from .llm_engine import LLMEngine

load_dotenv()

app = FastAPI(
    title="DOTMappers Ticket AI",
    version="1.0.0",
    description="Natural-language analytics and anomaly detection over support tickets.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Natural-language query request."""

    question: str = Field(..., min_length=3, max_length=1000)


class QueryResponse(BaseModel):
    """Structured NL-to-SQL response."""

    success: bool
    question: str
    sql: str | None = None
    data: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    summary: str
    provider: str
    model: str | None = None
    rationale: str | None = None
    error: str | None = None


csv_path_env = os.getenv("TICKETS_CSV_PATH", "").strip()
CSV_PATH = Path(csv_path_env or DEFAULT_CSV_PATH).expanduser().resolve()
database = TicketDatabase(CSV_PATH)
engine = LLMEngine(database)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return application and dataset health information."""
    return {
        "status": "ok",
        "database": "sqlite-memory",
        "dataset": str(CSV_PATH),
        "ticket_count": database.row_count,
        "dataset_max_created_at": database.dataset_max_created_at,
        "llm_strategy": engine.configured_provider(),
    }


@app.post("/query", response_model=QueryResponse)
def query_tickets(payload: QueryRequest) -> QueryResponse:
    """Answer a natural-language support-ticket analytics question."""
    try:
        result = engine.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryResponse(**result)


@app.get("/anomalies")
def anomalies() -> dict[str, Any]:
    """Run all deterministic anomaly detectors."""
    # Re-read the in-memory table so the endpoint remains consistent with the DB.
    detector = AnomalyDetector(database.to_dataframe())
    reference = os.getenv("ANOMALY_REFERENCE_AT", "").strip() or None
    return detector.detect_all(reference_time=reference)


@app.on_event("shutdown")
def shutdown_event() -> None:
    """Release the in-memory SQLite connection."""
    database.close()
