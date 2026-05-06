from __future__ import annotations

from typing import Sequence

import pandas as pd


DEFAULT_RANGE_BINS_M: tuple[float, ...] = (10, 25, 40, 55, 70, 85, 100)


def summarize_methods(
    results: pd.DataFrame,
    *,
    exhaustive_method: str = "exhaustive",
) -> pd.DataFrame:
    """Aggregate the main attach-search metrics by method."""
    required = {
        "method",
        "num_total_probes",
        "attached",
        "beam_index_error",
        "selected_equals_oracle",
        "final_snr_db",
    }
    _require_columns(results, required)

    summary = (
        results.groupby("method", as_index=False)
        .agg(
            avg_probes=("num_total_probes", "mean"),
            attach_success=("attached", "mean"),
            avg_beam_error=("beam_index_error", "mean"),
            exact_beam_match=("selected_equals_oracle", "mean"),
            avg_final_snr_db=("final_snr_db", "mean"),
        )
    )

    baseline = summary.loc[summary["method"] == exhaustive_method, "avg_probes"]
    if baseline.empty:
        summary["probe_reduction_vs_exhaustive"] = pd.NA
        return summary

    exhaustive_avg_probes = float(baseline.iloc[0])
    summary["probe_reduction_vs_exhaustive"] = (
        1.0 - summary["avg_probes"] / exhaustive_avg_probes
    )
    return summary


def summarize_by_range(
    results: pd.DataFrame,
    *,
    bins: Sequence[float] = DEFAULT_RANGE_BINS_M,
    include_lowest: bool = True,
) -> pd.DataFrame:
    """Aggregate attach metrics by method and UE range bin."""
    required = {
        "method",
        "ue_range_m",
        "attached",
        "beam_index_error",
        "final_snr_db",
    }
    _require_columns(results, required)

    df_range = results.copy()
    df_range["range_bin_m"] = pd.cut(
        df_range["ue_range_m"],
        bins=list(bins),
        include_lowest=include_lowest,
    )

    return (
        df_range.groupby(["method", "range_bin_m"], as_index=False, observed=False)
        .agg(
            attach_success=("attached", "mean"),
            avg_beam_error=("beam_index_error", "mean"),
            avg_final_snr_db=("final_snr_db", "mean"),
        )
    )


def summarize_threshold_frame(
    results: pd.DataFrame,
    *,
    threshold_col: str = "threshold_db",
) -> pd.DataFrame:
    """Aggregate a per-episode threshold-sweep result frame."""
    required = {
        threshold_col,
        "method",
        "attached",
        "num_total_probes",
        "beam_index_error",
        "final_snr_db",
    }
    _require_columns(results, required)

    return (
        results.groupby([threshold_col, "method"], as_index=False)
        .agg(
            attach_success=("attached", "mean"),
            avg_probes=("num_total_probes", "mean"),
            avg_beam_error=("beam_index_error", "mean"),
            avg_final_snr_db=("final_snr_db", "mean"),
        )
    )


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
