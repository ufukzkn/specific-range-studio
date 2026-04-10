from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BatchPredictFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass(slots=True)
class NomogramReportResult:
    """Artifacts for draft nomogram-style comparison outputs."""

    row_level_csv: Path
    slice_summary_csv: Path
    nomogram_png: Path


def build_nomogram_comparison(
    reference_df: pd.DataFrame,
    *,
    model_name: str,
    batch_predict_fn: BatchPredictFn,
) -> pd.DataFrame:
    """Attach model predictions for nomogram-style slice comparisons."""

    data = reference_df.copy().reset_index(drop=True)
    model_input = data[
        [
            "altitude",
            "gross_weight",
            "drag_index",
            "mach",
            "fuel_flow",
            "engine_type",
            "specific_range",
        ]
    ].copy()
    predictions = np.asarray(batch_predict_fn(model_input), dtype=float)
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    data["actual_specific_range"] = data["specific_range"].astype(float)
    data[f"{safe_name}_predicted"] = predictions
    data[f"{safe_name}_absolute_error"] = np.abs(predictions - data["actual_specific_range"].to_numpy(dtype=float))
    return data


def summarize_nomogram_slices(comparison_df: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    """Aggregate by engine/altitude/gross_weight slices for draft nomogram reports."""

    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    pred_col = f"{safe_name}_predicted"
    abs_col = f"{safe_name}_absolute_error"
    rows: list[dict[str, float | str | int]] = []
    for (engine_type, altitude, gross_weight), group in comparison_df.groupby(
        ["engine_type", "altitude", "gross_weight"],
        dropna=False,
    ):
        actual = group["actual_specific_range"].to_numpy(dtype=float)
        predicted = group[pred_col].to_numpy(dtype=float)
        metrics = regression_metrics(actual, predicted)
        rows.append(
            {
                "engine_type": str(engine_type),
                "altitude": float(altitude),
                "gross_weight": float(gross_weight),
                "rows": int(len(group)),
                "drag_index_count": int(group["drag_index"].nunique()),
                "mae": float(group[abs_col].mean()),
                "max_abs_error": float(group[abs_col].max()),
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
            }
        )
    return pd.DataFrame(rows).sort_values(["engine_type", "altitude", "gross_weight"]).reset_index(drop=True)


def plot_nomogram_comparison(
    comparison_df: pd.DataFrame,
    *,
    model_name: str,
    output_path: Path,
    engine_type: str,
    altitude: float,
    gross_weight: float,
) -> None:
    """Plot a draft nomogram-style comparison for one categorical slice."""

    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    pred_col = f"{safe_name}_predicted"
    abs_col = f"{safe_name}_absolute_error"

    subset = comparison_df[
        (comparison_df["engine_type"] == engine_type)
        & np.isclose(comparison_df["altitude"].astype(float), float(altitude), atol=1e-9, rtol=0.0)
        & np.isclose(comparison_df["gross_weight"].astype(float), float(gross_weight), atol=1e-9, rtol=0.0)
    ].copy()
    if subset.empty:
        return

    subset = subset.sort_values(["drag_index", "mach"]).reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(subset["drag_index"].nunique(), 2)))

    for color, (drag_index, group) in zip(colors, subset.groupby("drag_index", dropna=False)):
        group = group.sort_values("mach")
        label = f"drag_index={int(float(drag_index)) if float(drag_index).is_integer() else float(drag_index):g}"
        fuel_flow_label = f"fuel_flow~{group['fuel_flow'].mean():.0f}"
        axes[0].plot(group["mach"], group["actual_specific_range"], color=color, linewidth=2.0, label=f"{label} actual")
        axes[0].plot(group["mach"], group[pred_col], color=color, linewidth=1.5, linestyle="--", label=f"{label} pred")
        axes[0].text(
            float(group["mach"].iloc[-1]),
            float(group["actual_specific_range"].iloc[-1]),
            fuel_flow_label,
            fontsize=8,
            color=color,
        )
        axes[1].plot(group["mach"], group[abs_col], color=color, marker="o", label=label)

    axes[0].set_title(
        f"Draft nomogram comparison | {engine_type} | {int(altitude)} ft | gross_weight={int(gross_weight)}"
    )
    axes[0].set_xlabel("mach")
    axes[0].set_ylabel("specific_range")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("Absolute error by drag-index curve")
    axes[1].set_xlabel("mach")
    axes[1].set_ylabel("absolute_error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_nomogram_report(
    reference_df: pd.DataFrame,
    *,
    model_name: str,
    batch_predict_fn: BatchPredictFn,
    output_dir: Path,
    engine_type: str,
    altitude: float,
    gross_weight: float,
) -> NomogramReportResult:
    """Generate a draft nomogram-style report for a selected slice."""

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    comparison_df = build_nomogram_comparison(reference_df, model_name=safe_name, batch_predict_fn=batch_predict_fn)
    slice_summary_df = summarize_nomogram_slices(comparison_df, model_name=safe_name)

    row_level_csv = output_dir / f"{safe_name}_nomogram_row_level.csv"
    slice_summary_csv = output_dir / f"{safe_name}_nomogram_slice_summary.csv"
    nomogram_png = output_dir / (
        f"{safe_name}_nomogram_{engine_type}_{int(altitude)}ft_{int(gross_weight)}lb.png"
    )

    comparison_df.to_csv(row_level_csv, index=False)
    slice_summary_df.to_csv(slice_summary_csv, index=False)
    plot_nomogram_comparison(
        comparison_df,
        model_name=safe_name,
        output_path=nomogram_png,
        engine_type=engine_type,
        altitude=altitude,
        gross_weight=gross_weight,
    )
    return NomogramReportResult(
        row_level_csv=row_level_csv,
        slice_summary_csv=slice_summary_csv,
        nomogram_png=nomogram_png,
    )
