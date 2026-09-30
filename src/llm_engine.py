"""Natural-language-to-SQL engine with Groq, Ollama, and deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import requests
from dotenv import load_dotenv

from .database import SQLExecutionError, TicketDatabase, UnsafeQueryError

try:
    from groq import Groq
except ImportError:  # pragma: no cover - dependency is installed in requirements.txt
    Groq = None  # type: ignore[assignment]

load_dotenv()


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot generate a query."""


@dataclass(frozen=True)
class GeneratedQuery:
    """Normalized provider output."""

    sql: str
    rationale: str
    provider: str
    model: str | None = None


class QueryProvider(Protocol):
    """Protocol implemented by SQL-generating providers."""

    name: str

    def generate(self, question: str, prompt: str) -> GeneratedQuery:
        """Generate a safe candidate SQL query."""

    def repair(self, question: str, sql: str, error: str, prompt: str) -> GeneratedQuery:
        """Repair a failed candidate SQL query."""


class GroqProvider:
    """Groq chat-completions provider using the official Python SDK."""

    name = "groq"

    def __init__(self, api_key: str, model: str) -> None:
        if Groq is None:
            raise LLMProviderError("The groq package is not installed.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, question: str, prompt: str) -> GeneratedQuery:
        raw, model = self._call(prompt, question)
        return _parse_generated_query(raw, self.name, model)

    def repair(self, question: str, sql: str, error: str, prompt: str) -> GeneratedQuery:
        repair_prompt = (
            f"{prompt}\n\nA previous SQL query failed. Repair it without changing the user's intent.\n"
            f"Question: {question}\nFailed SQL: {sql}\nSQLite error: {error}\n"
            "Return the same JSON object shape with a corrected SELECT/WITH query."
        )
        raw, model = self._call(repair_prompt, question)
        return _parse_generated_query(raw, self.name, model)

    def _call(self, system_prompt: str, question: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # GPT-OSS models use the completion-token parameter and support
        # explicit reasoning control. Older Groq models use max_tokens instead.
        is_gpt_oss = self.model.startswith("openai/gpt-oss-")
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if is_gpt_oss:
            request_kwargs["reasoning_effort"] = "low"
            request_kwargs["max_completion_tokens"] = 2048
        else:
            request_kwargs["max_tokens"] = 1024

        # JSON mode is preferred for structured output. If the SDK/model rejects
        # that option, retry once without it so the application can still operate.
        try:
            completion = self.client.chat.completions.create(**request_kwargs)
        except Exception as first_exc:
            message = str(first_exc).lower()
            if "response_format" not in message and "json" not in message:
                raise LLMProviderError(f"Groq request failed: {first_exc}") from first_exc
            request_kwargs.pop("response_format", None)
            try:
                completion = self.client.chat.completions.create(**request_kwargs)
            except Exception as second_exc:
                raise LLMProviderError(f"Groq request failed: {second_exc}") from second_exc

        content = completion.choices[0].message.content
        if not content:
            raise LLMProviderError("Groq returned an empty response.")
        return content, self.model


class OllamaProvider:
    """Local Ollama provider using the HTTP API."""

    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            return response.ok
        except requests.RequestException:
            return False

    def generate(self, question: str, prompt: str) -> GeneratedQuery:
        raw = self._call(prompt, question)
        return _parse_generated_query(raw, self.name, self.model)

    def repair(self, question: str, sql: str, error: str, prompt: str) -> GeneratedQuery:
        repair_prompt = (
            f"{prompt}\n\nRepair the previous SQL.\nQuestion: {question}\n"
            f"Failed SQL: {sql}\nSQLite error: {error}\nReturn only the required JSON object."
        )
        raw = self._call(repair_prompt, question)
        return _parse_generated_query(raw, self.name, self.model)

    def _call(self, system_prompt: str, question: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc

        try:
            content = response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError("Ollama returned an unexpected response shape.") from exc
        if not content:
            raise LLMProviderError("Ollama returned an empty response.")
        return content


class MockProvider:
    """Deterministic heuristic provider used when no LLM service is configured."""

    name = "mock"

    def __init__(self, database: TicketDatabase) -> None:
        self.database = database

    def generate(self, question: str, prompt: str) -> GeneratedQuery:  # noqa: ARG002
        return self._translate(question)

    def repair(self, question: str, sql: str, error: str, prompt: str) -> GeneratedQuery:  # noqa: ARG002
        return self._translate(question)

    def _translate(self, question: str) -> GeneratedQuery:
        q = question.lower().strip()
        max_date = pd.to_datetime(self.database.dataset_max_created_at)
        month_start = max_date.replace(day=1).strftime("%Y-%m-%d 00:00:00")
        next_month = (max_date + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d 00:00:00")
        week_start = (max_date - pd.Timedelta(days=6)).strftime("%Y-%m-%d 00:00:00")
        max_day_end = (max_date + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")

        if "how many" in q and "open" in q:
            return self._result(
                "SELECT COUNT(*) AS open_tickets FROM tickets WHERE status = 'Open'",
                "Counts tickets currently marked Open.",
            )

        if "how many" in q and "critical" in q and ("unresolved" in q or "not resolved" in q):
            return self._result(
                "SELECT COUNT(*) AS critical_unresolved FROM tickets "
                "WHERE priority = 'Critical' AND status != 'Resolved'",
                "Counts Critical tickets whose status is not Resolved.",
            )

        if "resolved the most" in q and "this month" in q:
            sql = (
                "SELECT agent_id, COUNT(*) AS resolved_tickets "
                "FROM tickets WHERE status = 'Resolved' "
                f"AND created_at >= '{month_start}' AND created_at < '{next_month}' "
                "GROUP BY agent_id ORDER BY resolved_tickets DESC LIMIT 1"
            )
            return self._result(sql, "Finds the agent with the most resolved tickets in the dataset's latest month.")

        category_match = re.search(r"average customer rating for (billing|technical|general)", q)
        if category_match:
            category = category_match.group(1).title()
            sql = (
                "SELECT ROUND(AVG(customer_rating), 2) AS average_customer_rating "
                f"FROM tickets WHERE category = '{category}' AND customer_rating IS NOT NULL"
            )
            return self._result(sql, f"Averages non-null customer ratings for the {category} category.")

        if ("lowest average customer rating" in q) or ("lowest" in q and "average" in q and "rating" in q):
            return self._result(
                "SELECT agent_id, ROUND(AVG(customer_rating), 2) AS average_rating "
                "FROM tickets WHERE customer_rating IS NOT NULL "
                "GROUP BY agent_id ORDER BY average_rating ASC LIMIT 1",
                "Ranks agents by average non-null customer rating and returns the lowest average.",
            )

        if "critical tickets" in q and "12 hours" in q:
            return self._result(
                "SELECT ticket_id, created_at, priority, status, resolution_time_hrs, agent_id, issue_summary "
                "FROM tickets WHERE priority = 'Critical' "
                "AND (resolution_time_hrs IS NULL OR resolution_time_hrs > 12) "
                "ORDER BY created_at DESC LIMIT 200",
                "Shows Critical tickets unresolved or taking more than 12 hours to resolve.",
            )

        if "this week" in q and ("open" in q or "ticket" in q) and "how many" in q:
            return self._result(
                "SELECT COUNT(*) AS tickets_this_week FROM tickets "
                f"WHERE created_at >= '{week_start}' AND created_at < '{max_day_end}'",
                "Counts tickets in the latest seven-day window in the dataset.",
            )

        if "resolution" in q and "anomal" in q:
            iqr = self._resolution_iqr_upper_bound()
            sql = (
                "SELECT ticket_id, created_at, resolution_time_hrs, priority, status, agent_id, issue_summary "
                f"FROM tickets WHERE resolution_time_hrs > {iqr:.4f} "
                "ORDER BY resolution_time_hrs DESC LIMIT 200"
            )
            return self._result(sql, "Uses the dataset's 1.5*IQR upper bound to list unusually long resolutions.")

        for status in ("open", "escalated", "resolved"):
            if "how many" in q and status in q:
                return self._result(
                    f"SELECT COUNT(*) AS {status}_tickets FROM tickets WHERE status = '{status.title()}'",
                    f"Counts tickets with status {status.title()}.",
                )

        for category in ("billing", "technical", "general"):
            if "how many" in q and category in q:
                return self._result(
                    f"SELECT COUNT(*) AS {category}_tickets FROM tickets WHERE category = '{category.title()}'",
                    f"Counts tickets in the {category.title()} category.",
                )

        if "average response time" in q:
            return self._result(
                "SELECT ROUND(AVG(response_time_hrs), 2) AS average_response_time_hrs FROM tickets",
                "Calculates the average first-response time.",
            )

        if "average resolution time" in q:
            return self._result(
                "SELECT ROUND(AVG(resolution_time_hrs), 2) AS average_resolution_time_hrs "
                "FROM tickets WHERE resolution_time_hrs IS NOT NULL",
                "Calculates the average resolution time over resolved tickets.",
            )

        if "count" in q or "how many" in q:
            return self._result(
                "SELECT COUNT(*) AS total_tickets FROM tickets",
                "Falls back to a total ticket count when no narrower condition can be inferred.",
            )

        raise LLMProviderError(
            "Mock mode could not confidently map this question to SQL. Configure GROQ_API_KEY or Ollama for broader NL coverage."
        )

    def _resolution_iqr_upper_bound(self) -> float:
        df = self.database.to_dataframe()
        values = pd.to_numeric(df["resolution_time_hrs"], errors="coerce").dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        return float(q3 + 1.5 * (q3 - q1))

    @staticmethod
    def _result(sql: str, rationale: str) -> GeneratedQuery:
        return GeneratedQuery(sql=sql, rationale=rationale, provider="mock", model=None)


def _parse_generated_query(content: str, provider: str, model: str | None) -> GeneratedQuery:
    """Parse LLM output into a strict GeneratedQuery object."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMProviderError("LLM did not return valid JSON.")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMProviderError("LLM JSON output could not be parsed.") from exc

    sql = str(payload.get("sql", "")).strip()
    if not sql:
        raise LLMProviderError("LLM response did not contain a SQL query.")

    rationale = str(payload.get("rationale", "Generated from the user's question."))
    return GeneratedQuery(sql=sql, rationale=rationale, provider=provider, model=model)


def _format_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _labelize(column: str) -> str:
    """Convert SQL-style aliases into readable labels."""
    special = {
        "open_tickets": "open tickets",
        "critical_unresolved": "critical unresolved tickets",
        "resolved_tickets": "resolved tickets",
        "average_customer_rating": "average customer rating",
        "average_response_time_hrs": "average response time",
        "average_resolution_time_hrs": "average resolution time",
        "total_tickets": "total tickets",
        "tickets_this_week": "tickets this week",
    }
    return special.get(column, column.replace("_", " "))


class LLMEngine:
    """Orchestrate SQL generation, read-only execution, repair, and summarization."""

    def __init__(self, database: TicketDatabase) -> None:
        self.database = database
        self.max_rows = int(os.getenv("MAX_QUERY_ROWS", "200"))
        self.provider_mode = os.getenv("LLM_PROVIDER", "auto").lower().strip()
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
        self.mock = MockProvider(database)

        self._groq: GroqProvider | None = None
        if self.groq_key and self.provider_mode in {"auto", "groq"}:
            try:
                self._groq = GroqProvider(self.groq_key, self.groq_model)
            except LLMProviderError:
                self._groq = None

        self._ollama = OllamaProvider(self.ollama_url, self.ollama_model)

    def configured_provider(self) -> str:
        """Return the configured/fallback strategy for health checks."""
        if self.provider_mode == "mock":
            return "mock"
        if self.provider_mode == "ollama":
            return "ollama -> mock"
        if self.provider_mode == "groq":
            return "groq -> ollama -> mock"
        return "auto: groq -> ollama -> mock"

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a natural-language question with safe SQL and structured output."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        prompt = self._build_system_prompt()
        providers = self._provider_chain()
        provider_errors: list[str] = []

        for provider in providers:
            try:
                generated = provider.generate(question, prompt)
                rows, final_generated = self._execute_with_optional_repair(
                    provider, generated, question, prompt
                )
                return {
                    "success": True,
                    "question": question,
                    "sql": final_generated.sql,
                    "data": rows,
                    "row_count": len(rows),
                    "summary": self._summarize(question, rows),
                    "provider": final_generated.provider,
                    "model": final_generated.model,
                    "rationale": final_generated.rationale,
                }
            except (LLMProviderError, SQLExecutionError, UnsafeQueryError) as exc:
                provider_errors.append(f"{provider.name}: {exc}")
                continue

        return {
            "success": False,
            "question": question,
            "sql": None,
            "data": [],
            "row_count": 0,
            "summary": "The question could not be answered safely.",
            "provider": "none",
            "model": None,
            "rationale": "All configured providers failed or produced an unsafe SQL statement.",
            "error": "; ".join(provider_errors),
        }

    def _provider_chain(self) -> list[QueryProvider]:
        if self.provider_mode == "mock":
            return [self.mock]

        chain: list[QueryProvider] = []
        if self.provider_mode in {"auto", "groq"} and self._groq is not None:
            chain.append(self._groq)

        if self.provider_mode in {"auto", "groq", "ollama"} and self._ollama.available():
            chain.append(self._ollama)

        chain.append(self.mock)
        return chain

    def _execute_with_optional_repair(
        self,
        provider: QueryProvider,
        generated: GeneratedQuery,
        question: str,
        prompt: str,
    ) -> tuple[list[dict[str, Any]], GeneratedQuery]:
        try:
            rows = self.database.execute_read_only(
                generated.sql,
                max_rows=self.max_rows,
            )
            return rows, generated
        except UnsafeQueryError as exc:
            first_error = str(exc)
        except SQLExecutionError as exc:
            first_error = str(exc)

        repaired = provider.repair(question, generated.sql, first_error, prompt)
        rows = self.database.execute_read_only(
            repaired.sql,
            max_rows=self.max_rows,
        )
        return rows, repaired

    def _build_system_prompt(self) -> str:
        reference = self.database.dataset_max_created_at
        return f"""
You translate a user's natural-language analytics question into ONE safe SQLite SELECT/WITH query.

Database schema:
{self.database.schema_description()}

Rules:
1. Return JSON only: {{"sql": "...", "rationale": "..."}}.
2. The only table you may read is `tickets`.
3. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, DETACH, REPLACE,
   TRUNCATE, VACUUM, REINDEX, PRAGMA, transactions, or multiple statements.
4. Use explicit NULL handling. `resolution_time_hrs IS NULL` means unresolved.
5. `customer_rating IS NULL` means no customer rating is available.
6. Use SQLite syntax only. Do not invent tables or columns.
7. Prefer aggregates for summary questions and include ORDER BY/LIMIT when the user asks for a top/bottom entity.
8. For time phrases such as "this month" or "this week", anchor them to the dataset's latest created_at,
   which is {reference} in this assessment dataset, instead of the machine's current date.
9. When the user asks about unresolved tickets, use status != 'Resolved' and/or the documented NULL semantics.
10. Never include markdown fences in the JSON response.
""".strip()

    @staticmethod
    def _summarize(question: str, rows: list[dict[str, Any]]) -> str:
        """Turn structured query results into a concise natural-language answer."""
        question_l = question.lower()
        if not rows:
            return "No matching tickets were found for that question."

        if len(rows) == 1:
            row = rows[0]

            # Common assessment queries get a directly readable answer instead of
            # exposing internal SQL aliases such as `open_tickets`.
            if len(row) == 1:
                key, value = next(iter(row.items()))
                number = _format_value(value)
                if key == "open_tickets":
                    return f"There are {number} currently open tickets."
                if key == "critical_unresolved":
                    return f"There are {number} unresolved Critical tickets."
                if key == "average_customer_rating":
                    category_match = re.search(r"for (billing|technical|general)", question_l)
                    if category_match:
                        category = category_match.group(1).title()
                        return f"The average customer rating for {category} category tickets is {number}."
                    return f"The average customer rating is {number}."
                if key == "average_response_time_hrs":
                    return f"The average first-response time is {number} hours."
                if key == "average_resolution_time_hrs":
                    return f"The average resolution time is {number} hours."
                if key == "total_tickets":
                    return f"The dataset contains {number} tickets."
                if key == "tickets_this_week":
                    return f"There are {number} tickets in the latest seven-day dataset window."
                return f"{_labelize(key).capitalize()}: {number}."

            if {"agent_id", "resolved_tickets"}.issubset(row):
                return (
                    f"{row['agent_id']} resolved the most tickets in the latest dataset month, "
                    f"with {_format_value(row['resolved_tickets'])} resolved tickets."
                )

            if {"agent_id", "average_rating"}.issubset(row):
                return (
                    f"{row['agent_id']} has the lowest average customer rating, "
                    f"at {_format_value(row['average_rating'])}."
                )

            parts = [f"{_labelize(key)}={_format_value(value)}" for key, value in row.items()]
            return "Result: " + ", ".join(parts) + "."

        # For lists, provide a useful headline plus a small preview.
        preview_lines: list[str] = []
        for item in rows[:3]:
            ticket = item.get("ticket_id")
            if ticket:
                preview_lines.append(str(ticket))

        if "critical" in question_l and "12 hours" in question_l:
            suffix = f" Example tickets: {', '.join(preview_lines)}." if preview_lines else ""
            return f"Found {len(rows)} Critical tickets that were unresolved or took more than 12 hours to resolve.{suffix}"

        suffix = f" Example tickets: {', '.join(preview_lines)}." if preview_lines else ""
        return f"Found {len(rows)} matching rows.{suffix}"
