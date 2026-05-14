"""Group-level analysis utilities for collecting and aggregating metrics."""

import pandas as pd
from pathlib import Path
from typing import List, Optional


class MetricsCollector:
    """Collects metrics in tidy (long) format for statistical analysis."""

    def __init__(self):
        self.rows: List[Dict] = []

    def add_row(
        self,
        subject: str,
        analysis_type: str,
        condition: str,
        channel: str,
        component: str,
        metric: str,
        value: float,
    ) -> None:
        """Add a single metric row."""
        self.rows.append({
            "subject": subject,
            "analysis_type": analysis_type,
            "condition": condition,
            "channel": channel,
            "component": component,
            "metric": metric,
            "value": value,
        })

    def add_peak_to_peak(
        self,
        subject: str,
        analysis_type: str,
        condition: str,
        channel: str,
        component: str,
        value: float,
    ) -> None:
        """Add a peak-to-peak amplitude row."""
        self.add_row(
            subject=subject,
            analysis_type=analysis_type,
            condition=condition,
            channel=channel,
            component=component,
            metric="peak_to_peak_uV",
            value=value,
        )

    def add_mfp_peaks(
        self,
        subject: str,
        analysis_type: str,
        condition: str,
        channel: str,
        component: str,
        amplitude: float,
        latency: float,
    ) -> None:
        """Add both amplitude and latency rows for an MFP peak."""
        self.add_row(
            subject=subject,
            analysis_type=analysis_type,
            condition=condition,
            channel=channel,
            component=component,
            metric="peak_amplitude_uV",
            value=amplitude,
        )
        self.add_row(
            subject=subject,
            analysis_type=analysis_type,
            condition=condition,
            channel=channel,
            component=component,
            metric="peak_latency_ms",
            value=latency,
        )

    def collect_peak_to_peak_from_df(
        self,
        subject: str,
        analysis_type: str,
        df: pd.DataFrame,
        component: str,
    ) -> None:
        """Collect peak-to-peak rows from a DataFrame.

        Args:
            subject: Subject ID.
            analysis_type: "condition" or "context".
            df: DataFrame with columns [condition, channel, component, peak_to_peak_uV].
            component: Component label (e.g., "N15-P30").
        """
        for _, row in df.iterrows():
            self.add_peak_to_peak(
                subject=subject,
                analysis_type=analysis_type,
                condition=row["condition"],
                channel=row["channel"],
                component=component,
                value=row["peak_to_peak_uV"],
            )

    def collect_mfp_peaks_from_df(
        self,
        subject: str,
        analysis_type: str,
        df: pd.DataFrame,
        channel: str,
    ) -> None:
        """Collect MFP peak rows from a DataFrame.

        Args:
            subject: Subject ID.
            analysis_type: "condition" or "context".
            df: DataFrame with columns [condition, measure, component, peak_amplitude_uV, peak_latency_ms].
            channel: Channel label ("GMFP" or "LMFP").
        """
        for _, row in df.iterrows():
            self.add_mfp_peaks(
                subject=subject,
                analysis_type=analysis_type,
                condition=row["condition"],
                channel=channel,
                component=row["component"],
                amplitude=row["peak_amplitude_uV"],
                latency=row["peak_latency_ms"],
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected rows to a tidy DataFrame."""
        return pd.DataFrame(self.rows)

    def export_csv(
        self,
        output_path: str = "data/group/all_subjects_metrics.csv",
        export_enabled: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Export collected rows to CSV if enabled.

        Args:
            output_path: Path to save the CSV file.
            export_enabled: Whether to actually export.

        Returns:
            The DataFrame if exported, None otherwise.
        """
        df = self.to_dataframe()

        if export_enabled:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"\nMetrics exported to: {path}")
            print(f"Total rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")

        return df