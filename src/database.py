"""SQLite data layer for the support ticket dataset.

The assessment requires an in-memory SQLite database. This module owns the
single connection used by the API process, validates and normalizes the CSV,
and exposes a small read-only query surface to the rest of the application.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "support_tickets.csv"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
)

_STRING_COLUMNS = ("ticket_id", "category", "priority", "status", "agent_id", "issue_summary")
_FLOAT_COLUMNS = ("response_time_hrs", "resolution_time_hrs", "customer_rating")

_BANNED_SQL = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REPLACE|TRUNCATE|VACUUM|REINDEX|PRAGMA|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


class DatabaseError(RuntimeError):
    """Base exception for database failures."""


class DataValidationError(DatabaseError):
    """Raised when the input CSV does not satisfy the expected schema."""


class UnsafeQueryError(DatabaseError):
    """Raised when a query is not a safe, single read-only SQLite statement."""


class SQLExecutionError(DatabaseError):
    """Raised when a validated SQL statement fails during execution."""


class TicketDatabase:
    """Manage the normalized support-ticket dataset in an in-memory SQLite DB."""

    def __init__(self, csv_path: str | Path = DEFAULT_CSV_PATH) -> None:
        self.csv_path = Path(csv_path)
        self._lock = threading.RLock()
        try:
            self.connection = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
            )
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self._load_csv()
            self._install_read_only_authorizer()
        except Exception:
            try:
                self.connection.close()
            finally:
                raise

    def _install_read_only_authorizer(self) -> None:
        """Install a SQLite-level read-only guard in addition to SQL text checks."""

        allowed_actions = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
        }

        dangerous_functions = {
            "load_extension",
            "writefile",
            "edit",
        }

        def authorizer(
            action_code: int,
            arg1: str | None,
            arg2: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action_code not in allowed_actions:
                return sqlite3.SQLITE_DENY

            if action_code == sqlite3.SQLITE_READ and arg1 not in {"tickets"}:
                return sqlite3.SQLITE_DENY

            if action_code == sqlite3.SQLITE_FUNCTION:
                function_name = (arg2 or arg1 or "").lower()
                if function_name in dangerous_functions:
                    return sqlite3.SQLITE_DENY

            return sqlite3.SQLITE_OK

        self.connection.set_authorizer(authorizer)

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise DataValidationError(
                f"CSV is missing required columns: {', '.join(missing)}"
            )

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dates, strings, floats, and allowed null values."""
        df = df.loc[:, list(REQUIRED_COLUMNS)].copy()

        parsed_dates = pd.to_datetime(df["created_at"], errors="coerce")
        if parsed_dates.isna().any():
            bad_count = int(parsed_dates.isna().sum())
            raise DataValidationError(
                f"created_at contains {bad_count} invalid or missing timestamp values."
            )
        df["created_at"] = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S")

        for column in _STRING_COLUMNS:
            df[column] = df[column].astype("string").str.strip()

        for column in _FLOAT_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        if df["ticket_id"].isna().any() or df["ticket_id"].duplicated().any():
            raise DataValidationError("ticket_id values must be present and unique.")

        if df["response_time_hrs"].isna().any():
            raise DataValidationError("response_time_hrs contains missing or non-numeric values.")

        invalid_rating = df["customer_rating"].notna() & ~df["customer_rating"].between(1, 5)
        if invalid_rating.any():
            raise DataValidationError("customer_rating must be between 1 and 5 when present.")

        invalid_resolution = df["resolution_time_hrs"].notna() & (df["resolution_time_hrs"] < 0)
        invalid_response = df["response_time_hrs"] < 0
        if invalid_resolution.any() or invalid_response.any():
            raise DataValidationError("Time columns cannot contain negative values.")

        return df

    def _load_csv(self) -> None:
        if not self.csv_path.exists():
            raise DataValidationError(f"Dataset not found: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as exc:
            raise DataValidationError(f"Unable to read CSV: {exc}") from exc

        self._validate_columns(df)
        df = self._normalize_dataframe(df)

        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE tickets (
                    ticket_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time_hrs REAL NOT NULL,
                    resolution_time_hrs REAL,
                    agent_id TEXT NOT NULL,
                    customer_rating REAL,
                    issue_summary TEXT NOT NULL
                );

                CREATE INDEX idx_tickets_created_at ON tickets(created_at);
                CREATE INDEX idx_tickets_status ON tickets(status);
                CREATE INDEX idx_tickets_priority ON tickets(priority);
                CREATE INDEX idx_tickets_category ON tickets(category);
                CREATE INDEX idx_tickets_agent_id ON tickets(agent_id);
                """
            )

            records = [
                tuple(None if pd.isna(value) else value for value in row)
                for row in df.itertuples(index=False, name=None)
            ]
            self.connection.executemany(
                """
                INSERT INTO tickets (
                    ticket_id, created_at, category, priority, status,
                    response_time_hrs, resolution_time_hrs, agent_id,
                    customer_rating, issue_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            self.connection.commit()

    @property
    def row_count(self) -> int:
        """Return the number of loaded tickets."""
        with self._lock:
            return int(self.connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0])

    @property
    def dataset_max_created_at(self) -> str:
        """Return the latest ticket timestamp in ISO-like SQLite text format."""
        with self._lock:
            value = self.connection.execute("SELECT MAX(created_at) FROM tickets").fetchone()[0]
        return str(value)

    def schema_description(self) -> str:
        """Return a compact schema description suitable for an LLM prompt."""
        return """
Table: tickets
- ticket_id TEXT PRIMARY KEY: unique ticket ID
- created_at TEXT: timestamp stored as YYYY-MM-DD HH:MM:SS
- category TEXT: Billing | Technical | General
- priority TEXT: Low | Medium | High | Critical
- status TEXT: Open | Resolved | Escalated
- response_time_hrs REAL: hours from creation to first response
- resolution_time_hrs REAL: hours from creation to resolution; NULL when unresolved
- agent_id TEXT: assigned agent ID
- customer_rating REAL: 1-5 post-resolution rating; NULL when unresolved
- issue_summary TEXT: short free-text description

Null semantics:
- NULL resolution_time_hrs means the ticket is unresolved.
- NULL customer_rating means there is no post-resolution rating yet.
""".strip()

    def to_dataframe(self) -> pd.DataFrame:
        """Return the complete table as a pandas DataFrame."""
        with self._lock:
            rows = self.connection.execute("SELECT * FROM tickets ORDER BY created_at").fetchall()
        df = pd.DataFrame([dict(row) for row in rows], columns=REQUIRED_COLUMNS)
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df

    @staticmethod
    def validate_read_only_query(query: str) -> str:
        """Validate that *query* is a single SELECT/WITH statement."""
        if not isinstance(query, str) or not query.strip():
            raise UnsafeQueryError("SQL query is empty.")

        normalized = query.strip()
        if "--" in normalized or "/*" in normalized or "*/" in normalized:
            raise UnsafeQueryError("SQL comments are not permitted.")

        if normalized.endswith(";"):
            normalized = normalized[:-1].rstrip()

        if ";" in normalized:
            raise UnsafeQueryError("Only one SQL statement is allowed.")

        if not re.match(r"^(SELECT|WITH)\b", normalized, re.IGNORECASE):
            raise UnsafeQueryError("Only SELECT or WITH queries are permitted.")

        if _BANNED_SQL.search(normalized):
            raise UnsafeQueryError("Read-only SQL policy rejected the query.")

        if re.search(r"\bINTO\b", normalized, re.IGNORECASE):
            raise UnsafeQueryError("SELECT INTO is not permitted.")

        return normalized

    def execute_read_only(
        self,
        query: str,
        params: Sequence[Any] | Iterable[Any] = (),
        max_rows: int = 200,
    ) -> list[dict[str, Any]]:
        """Execute a validated query under the SQLite authorizer and return JSON-safe rows."""
        normalized = self.validate_read_only_query(query)
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")

        try:
            with self._lock:
                cursor = self.connection.execute(normalized, tuple(params))
                rows = cursor.fetchmany(max_rows)
                columns = [description[0] for description in cursor.description or ()]
        except sqlite3.Error as exc:
            raise SQLExecutionError(f"SQL execution failed: {exc}") from exc

        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self.connection.close()
