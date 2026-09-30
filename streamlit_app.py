"""Streamlit dashboard for the DOTMappers Ticket AI assessment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "support_tickets.csv"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

SAMPLE_QUERIES = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets this month?",
    "Show me all Critical tickets not resolved within 12 hours.",
    "What is the average customer rating for Technical category tickets?",
    "Are there any anomalies in resolution times this week?",
]

st.set_page_config(
    page_title="DOTMappers Ticket AI",
    page_icon="🎫",
    layout="wide",
)

st.title("🎫 DOTMappers Ticket AI")
st.caption("Natural-language analytics + deterministic anomaly detection")


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("System")
    st.code(API_BASE_URL, language="text")
    try:
        health = api_get("/health")
        st.success("API online")
        st.metric("Tickets loaded", health["ticket_count"])
        st.caption(f"LLM strategy: {health['llm_strategy']}")
    except requests.RequestException as exc:
        st.error("API is offline")
        st.caption(str(exc))
        st.info("Start the application with `bash run.sh`.")


tab_query, tab_anomalies, tab_raw = st.tabs(
    ["💬 Natural Language Query", "🚨 Anomaly Dashboard", "🗂️ Raw Data Explorer"]
)

with tab_query:
    st.subheader("Ask questions about the ticket dataset")
    st.write("Pick a sample or type your own question. The backend generates a read-only SQLite query.")

    chip_columns = st.columns(len(SAMPLE_QUERIES))
    for index, sample in enumerate(SAMPLE_QUERIES):
        if chip_columns[index].button(sample, key=f"sample_{index}", use_container_width=True):
            st.session_state["selected_question"] = sample

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(message["sql"], language="sql")
            if message.get("data"):
                st.dataframe(pd.DataFrame(message["data"]), use_container_width=True)

    selected_question = st.session_state.get("selected_question", "")
    prompt = st.chat_input("Ask a question, e.g. Which agent has the lowest average rating?")
    question = prompt or selected_question

    if question:
        st.session_state["selected_question"] = ""
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            result = api_post("/query", {"question": question})
        except requests.RequestException as exc:
            result = {"success": False, "error": str(exc), "summary": "API request failed."}

        if result.get("success"):
            assistant_content = result.get("summary", "Query completed.")
            if result.get("provider"):
                assistant_content += f"\n\n*Provider: **{result['provider']}**"
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "sql": result.get("sql"),
                    "data": result.get("data", []),
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"❌ {result.get('error', 'The query could not be completed safely.')}",
                }
            )
        st.rerun()

with tab_anomalies:
    st.subheader("Anomaly Detection Dashboard")
    try:
        payload = api_get("/anomalies")
        summary = payload["summary"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Resolution-time outliers", summary["resolution_outliers"])
        col2.metric("Stale High/Critical", summary["stale_high_priority"])
        col3.metric("Low-rating clusters", summary["low_rating_clusters"])

        resolution = payload["resolution_time_outliers"]
        st.markdown("### 1. Resolution-time IQR outliers")
        stats = resolution["stats"]
        st.caption(
            f"Q1={stats['q1']}, Q3={stats['q3']}, IQR={stats['iqr']}, "
            f"upper bound={stats['upper_bound']} hrs"
        )
        st.dataframe(pd.DataFrame(resolution["tickets"]), use_container_width=True)

        stale = payload["stale_high_priority"]
        st.markdown("### 2. Open/Escalated High & Critical tickets older than 24h")
        st.caption(f"Reference timestamp: {stale['reference_time']}")
        st.dataframe(pd.DataFrame(stale["tickets"]), use_container_width=True)

        clusters = payload["low_rating_clusters"]
        st.markdown("### 3. Low-rating clusters")
        st.caption(
            "Cluster = at least 3 rated tickets in a week where >=50% are rated 1-2 "
            "or the weekly average is <=2.5."
        )
        st.dataframe(pd.DataFrame(clusters["clusters"]), use_container_width=True)
    except requests.RequestException as exc:
        st.error(f"Could not load anomaly data: {exc}")

with tab_raw:
    st.subheader("Raw Data Explorer")
    try:
        df = pd.read_csv(DATA_PATH)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        c1, c2, c3, c4 = st.columns(4)
        selected_categories = c1.multiselect("Category", sorted(df["category"].dropna().unique()))
        selected_priorities = c2.multiselect("Priority", sorted(df["priority"].dropna().unique()))
        selected_statuses = c3.multiselect("Status", sorted(df["status"].dropna().unique()))
        search = c4.text_input("Search issue summary")

        filtered = df.copy()
        if selected_categories:
            filtered = filtered[filtered["category"].isin(selected_categories)]
        if selected_priorities:
            filtered = filtered[filtered["priority"].isin(selected_priorities)]
        if selected_statuses:
            filtered = filtered[filtered["status"].isin(selected_statuses)]
        if search:
            filtered = filtered[
                filtered["issue_summary"].astype(str).str.contains(search, case=False, na=False)
            ]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows shown", len(filtered))
        m2.metric("Open", int((filtered["status"] == "Open").sum()))
        m3.metric("Escalated", int((filtered["status"] == "Escalated").sum()))
        m4.metric("Avg. rating", f"{filtered['customer_rating'].mean():.2f}" if filtered["customer_rating"].notna().any() else "N/A")

        st.dataframe(filtered, use_container_width=True, height=520)
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
