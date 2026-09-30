"""Deterministic anomaly detection for support tickets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class IQROutlierStats:
    """Summary statistics for resolution-time outlier detection."""

    q1: float | None
    q3: float | None
    iqr: float | None
    lower_bound: float | None
    upper_bound: float | None


class AnomalyDetector:
    """Run the assessment's three required anomaly detectors."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        stale_hours: float = 24.0,
        low_rating_threshold: float = 2.0,
        low_cluster_share: float = 0.50,
        low_cluster_average: float = 2.50,
        minimum_cluster_size: int = 3,
    ) -> None:
        self.df = df.copy()
        self.stale_hours = stale_hours
        self.low_rating_threshold = low_rating_threshold
        self.low_cluster_share = low_cluster_share
        self.low_cluster_average = low_cluster_average
        self.minimum_cluster_size = minimum_cluster_size

        if "created_at" in self.df.columns:
            self.df["created_at"] = pd.to_datetime(self.df["created_at"], errors="coerce")

    def resolution_time_outliers(self) -> dict[str, Any]:
        """Detect resolution times beyond the 1.5*IQR upper bound."""
        series = pd.to_numeric(self.df["resolution_time_hrs"], errors="coerce").dropna()
        if series.empty:
            stats = IQROutlierStats(None, None, None, None, None)
            return {"stats": stats.__dict__, "count": 0, "tickets": []}

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = pd.to_numeric(self.df["resolution_time_hrs"], errors="coerce") > upper
        outliers = self.df.loc[mask, [
            "ticket_id",
            "created_at",
            "category",
            "priority",
            "status",
            "resolution_time_hrs",
            "agent_id",
            "issue_summary",
        ]].copy()
        outliers["outlier_type"] = "resolution_time_iqr"
        outliers["upper_bound_hrs"] = upper

        return {
            "stats": IQROutlierStats(q1, q3, iqr, lower, upper).__dict__,
            "count": int(len(outliers)),
            "tickets": self._records(outliers),
        }

    def stale_high_priority(self, reference_time: datetime | str | None = None) -> dict[str, Any]:
        """Detect open/escalated High/Critical tickets older than 24 hours.

        For deterministic assessment data, the default reference is the latest
        timestamp in the dataset rather than the machine's wall clock.
        """
        ref = self._resolve_reference_time(reference_time)
        created = pd.to_datetime(self.df["created_at"], errors="coerce")
        age_hours = (ref - created).dt.total_seconds() / 3600.0

        mask = (
            self.df["status"].isin(["Open", "Escalated"])
            & self.df["priority"].isin(["High", "Critical"])
            & (age_hours > self.stale_hours)
        )

        stale = self.df.loc[mask, [
            "ticket_id",
            "created_at",
            "category",
            "priority",
            "status",
            "agent_id",
            "response_time_hrs",
            "resolution_time_hrs",
            "customer_rating",
            "issue_summary",
        ]].copy()
        stale["age_hours"] = age_hours.loc[mask].round(2).values
        stale["anomaly_type"] = "stale_high_priority"

        return {
            "reference_time": ref.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold_hours": self.stale_hours,
            "count": int(len(stale)),
            "tickets": self._records(stale),
        }

    def low_rating_clusters(self) -> dict[str, Any]:
        """Find concentrated low-rating periods by agent and by category.

        A cluster is a calendar week containing at least three rated tickets
        where either at least 50% are rated 1-2 or the average rating is <= 2.5.
        """
        rated = self.df.dropna(subset=["customer_rating"]).copy()
        if rated.empty:
            return {"count": 0, "clusters": []}

        rated["week"] = rated["created_at"].dt.to_period("W-SUN").astype(str)
        rated["is_low"] = rated["customer_rating"] <= self.low_rating_threshold

        frames: list[pd.DataFrame] = []
        for group_columns, group_type in [
            (["agent_id", "week"], "agent_week"),
            (["category", "week"], "category_week"),
        ]:
            grouped = (
                rated.groupby(group_columns, dropna=False)
                .agg(
                    rated_tickets=("customer_rating", "count"),
                    average_rating=("customer_rating", "mean"),
                    low_rating_count=("is_low", "sum"),
                )
                .reset_index()
            )
            grouped["low_rating_share"] = (
                grouped["low_rating_count"] / grouped["rated_tickets"]
            )
            grouped["cluster_type"] = group_type
            keep = (
                (grouped["rated_tickets"] >= self.minimum_cluster_size)
                & (
                    (grouped["low_rating_share"] >= self.low_cluster_share)
                    | (grouped["average_rating"] <= self.low_cluster_average)
                )
            )
            frames.append(grouped.loc[keep])

        clusters = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not clusters.empty:
            clusters = clusters.sort_values(
                ["average_rating", "low_rating_share"], ascending=[True, False]
            ).reset_index(drop=True)

        return {
            "count": int(len(clusters)),
            "thresholds": {
                "low_rating": self.low_rating_threshold,
                "minimum_cluster_size": self.minimum_cluster_size,
                "low_rating_share": self.low_cluster_share,
                "average_rating": self.low_cluster_average,
            },
            "clusters": self._records(clusters),
        }

    def detect_all(self, reference_time: datetime | str | None = None) -> dict[str, Any]:
        """Run all three detectors and return a unified dashboard payload."""
        resolution = self.resolution_time_outliers()
        stale = self.stale_high_priority(reference_time)
        clusters = self.low_rating_clusters()
        return {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "summary": {
                "resolution_outliers": resolution["count"],
                "stale_high_priority": stale["count"],
                "low_rating_clusters": clusters["count"],
            },
            "resolution_time_outliers": resolution,
            "stale_high_priority": stale,
            "low_rating_clusters": clusters,
        }

    def _resolve_reference_time(self, reference_time: datetime | str | None) -> datetime:
        if reference_time is None:
            maximum = self.df["created_at"].max()
            if pd.isna(maximum):
                return datetime.utcnow()
            return maximum.to_pydatetime()
        if isinstance(reference_time, datetime):
            return reference_time
        parsed = pd.to_datetime(reference_time, errors="coerce")
        if pd.isna(parsed):
            raise ValueError("Invalid anomaly reference timestamp.")
        return parsed.to_pydatetime()

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        clean = frame.copy()
        for column in clean.columns:
            if pd.api.types.is_datetime64_any_dtype(clean[column]):
                clean[column] = clean[column].dt.strftime("%Y-%m-%d %H:%M:%S")
        clean = clean.astype(object).where(pd.notna(clean), None)
        return clean.to_dict(orient="records")
