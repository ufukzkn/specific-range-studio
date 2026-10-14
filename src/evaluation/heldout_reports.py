"""UI CSV/PNG reports scoped exclusively to the frozen test partition."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.table_report import build_row_level_comparison, summarize_by_slice, summarize_overall


def write_heldout_reports(test: pd.DataFrame, key: str, predict, folder: Path):
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    rows = build_row_level_comparison(test, model_name=key,
                 predict_fn=lambda f: float(predict(f)[0]), batch_predict_fn=predict)
    slices = summarize_by_slice(rows, model_name=key)
    overall = summarize_overall(rows, model_name=key)
    rows.to_csv(folder / f"{key}_row_level_comparison.csv", index=False)
    slices.to_csv(folder / f"{key}_slice_summary.csv", index=False)
    overall.to_csv(folder / f"{key}_overall_summary.csv", index=False)
    actual = rows.actual_specific_range.to_numpy(dtype=float)
    predicted = rows[f"{key}_predicted"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(actual, predicted, s=3, alpha=0.3)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(limits, limits, color="black", linewidth=1)
    axes[0].set(xlabel="Table target (source numeric unit)", ylabel="Prediction", title=f"{key}: held-out test only")
    axes[1].scatter(actual, predicted-actual, s=3, alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Table target", ylabel="Prediction - target", title="Test residuals")
    fig.savefig(folder / f"{key}_slice_predictions.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(slices))
    axis.bar(x, slices.rmse)
    axis.set_xticks(x, [f"{r.engine_type}\n{r.altitude:g} ft" for r in slices.itertuples()], rotation=60, ha="right", fontsize=7)
    axis.set(ylabel="RMSE (source numeric unit)", title=f"{key}: held-out test error by engine / altitude")
    fig.savefig(folder / f"{key}_slice_summary.png", dpi=150)
    plt.close(fig)
