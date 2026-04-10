from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.inference.predictors import build_single_row_frame

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PredictFn = Callable[[pd.DataFrame], float]
BatchPredictFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass(slots=True)
class TableReportResult:
    """Paths to generated table-comparison artifacts."""

    row_level_csv: Path
    slice_summary_csv: Path
    overall_summary_csv: Path
    workbook_xlsx: Path
    slice_plot_png: Path
    summary_plot_png: Path


def build_row_level_comparison(
    reference_df: pd.DataFrame,
    *,
    model_name: str,
    predict_fn: PredictFn,
    batch_predict_fn: BatchPredictFn | None = None,
) -> pd.DataFrame:
    """Predict every real table row and attach per-row errors."""

    rows: list[dict[str, float | str | int]] = []
    data = reference_df.reset_index(drop=True).copy()
    if batch_predict_fn is not None:
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
        predictions = batch_predict_fn(model_input)
    else:
        predictions = []
        for _, row in data.iterrows():
            frame = build_single_row_frame(
                altitude=float(row["altitude"]),
                gross_weight=float(row["gross_weight"]),
                drag_index=float(row["drag_index"]),
                mach=float(row["mach"]),
                fuel_flow=float(row["fuel_flow"]),
                engine_type=str(row["engine_type"]),
            )
            predictions.append(float(predict_fn(frame)))
        predictions = np.asarray(predictions, dtype=float)

    for idx, row in data.iterrows():
        prediction = float(predictions[idx])
        actual = float(row["specific_range"])
        rows.append(
            {
                "row_id": int(idx),
                "engine_type": str(row["engine_type"]),
                "altitude": float(row["altitude"]),
                "gross_weight": float(row["gross_weight"]),
                "drag_index": float(row["drag_index"]),
                "mach": float(row["mach"]),
                "fuel_flow": float(row["fuel_flow"]),
                "actual_specific_range": actual,
                f"{model_name}_predicted": prediction,
                f"{model_name}_absolute_error": abs(prediction - actual),
                f"{model_name}_signed_error": prediction - actual,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_slice(row_level_df: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    """Aggregate row-level comparison by engine type and altitude slice."""

    pred_col = f"{model_name}_predicted"
    abs_col = f"{model_name}_absolute_error"
    grouped_rows: list[dict[str, float | str | int]] = []
    for (engine_type, altitude), group in row_level_df.groupby(["engine_type", "altitude"], dropna=False):
        actual = group["actual_specific_range"].to_numpy(dtype=float)
        predicted = group[pred_col].to_numpy(dtype=float)
        metrics = regression_metrics(actual, predicted)
        grouped_rows.append(
            {
                "engine_type": str(engine_type),
                "altitude": float(altitude),
                "rows": int(len(group)),
                "mae": float(group[abs_col].mean()),
                "max_abs_error": float(group[abs_col].max()),
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(["engine_type", "altitude"]).reset_index(drop=True)


def summarize_overall(row_level_df: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    """Build a one-row overall metric summary."""

    pred_col = f"{model_name}_predicted"
    abs_col = f"{model_name}_absolute_error"
    actual = row_level_df["actual_specific_range"].to_numpy(dtype=float)
    predicted = row_level_df[pred_col].to_numpy(dtype=float)
    metrics = regression_metrics(actual, predicted)
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "rows": int(len(row_level_df)),
                "mae": float(row_level_df[abs_col].mean()),
                "max_abs_error": float(row_level_df[abs_col].max()),
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
            }
        ]
    )


def _plot_slice_predictions(row_level_df: pd.DataFrame, *, model_name: str, output_path: Path) -> None:
    pred_col = f"{model_name}_predicted"
    sample = (
        row_level_df.sort_values(["engine_type", "altitude", "gross_weight", "drag_index", "mach"])
        .groupby(["engine_type", "altitude"], dropna=False)
        .head(12)
        .copy()
    )

    if sample.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    for (engine_type, altitude), group in sample.groupby(["engine_type", "altitude"], dropna=False):
        label_base = f"{engine_type} @ {int(float(altitude))} ft"
        axes[0].plot(group["mach"], group["actual_specific_range"], marker="o", label=f"{label_base} actual")
        axes[0].plot(group["mach"], group[pred_col], marker="x", linestyle="--", label=f"{label_base} pred")
        axes[1].plot(
            group["mach"],
            group[f"{model_name}_absolute_error"],
            marker="o",
            label=f"{label_base} abs err",
        )

    axes[0].set_title(f"{model_name}: actual vs predicted specific range")
    axes[0].set_xlabel("Mach")
    axes[0].set_ylabel("Specific Range")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title(f"{model_name}: absolute error by Mach")
    axes[1].set_xlabel("Mach")
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_slice_summary(slice_summary_df: pd.DataFrame, *, model_name: str, output_path: Path) -> None:
    if slice_summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for engine_type, group in slice_summary_df.groupby("engine_type", dropna=False):
        ax.plot(group["altitude"], group["mae"], marker="o", label=f"{engine_type} MAE")
        ax.plot(group["altitude"], group["rmse"], marker="x", linestyle="--", label=f"{engine_type} RMSE")

    ax.set_title(f"{model_name}: slice-level error summary")
    ax.set_xlabel("Altitude (ft)")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_table_report(
    reference_df: pd.DataFrame,
    *,
    model_name: str,
    predict_fn: PredictFn,
    batch_predict_fn: BatchPredictFn | None = None,
    output_dir: Path,
) -> TableReportResult:
    """Generate full-table comparison CSVs and plots for one model."""

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")

    row_level_df = build_row_level_comparison(
        reference_df,
        model_name=safe_name,
        predict_fn=predict_fn,
        batch_predict_fn=batch_predict_fn,
    )
    slice_summary_df = summarize_by_slice(row_level_df, model_name=safe_name)
    overall_summary_df = summarize_overall(row_level_df, model_name=safe_name)

    row_level_csv = output_dir / f"{safe_name}_row_level_comparison.csv"
    slice_summary_csv = output_dir / f"{safe_name}_slice_summary.csv"
    overall_summary_csv = output_dir / f"{safe_name}_overall_summary.csv"
    workbook_xlsx = output_dir / f"{safe_name}_table_report.xlsx"
    slice_plot_png = output_dir / f"{safe_name}_slice_predictions.png"
    summary_plot_png = output_dir / f"{safe_name}_slice_summary.png"

    row_level_df.to_csv(row_level_csv, index=False)
    slice_summary_df.to_csv(slice_summary_csv, index=False)
    overall_summary_df.to_csv(overall_summary_csv, index=False)
    with pd.ExcelWriter(workbook_xlsx) as writer:
        row_level_df.to_excel(writer, index=False, sheet_name="row_level")
        slice_summary_df.to_excel(writer, index=False, sheet_name="slice_summary")
        overall_summary_df.to_excel(writer, index=False, sheet_name="overall_summary")
    _plot_slice_predictions(row_level_df, model_name=safe_name, output_path=slice_plot_png)
    _plot_slice_summary(slice_summary_df, model_name=safe_name, output_path=summary_plot_png)

    return TableReportResult(
        row_level_csv=row_level_csv,
        slice_summary_csv=slice_summary_csv,
        overall_summary_csv=overall_summary_csv,
        workbook_xlsx=workbook_xlsx,
        slice_plot_png=slice_plot_png,
        summary_plot_png=summary_plot_png,
    )
