from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.table_report import generate_table_report
from src.inference.predictors import FTTransformerPredictor, XGBoostPredictor
from src.utils.config import DataConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full-table comparison reports from trained artifacts.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to processed CSV.")
    parser.add_argument("--model", choices=["xgboost", "ft_transformer", "both"], default="both")
    parser.add_argument("--ft-device", type=str, default="cpu")
    args = parser.parse_args()

    data_config = DataConfig()
    frame = pd.read_csv(args.dataset) if args.dataset else pd.read_csv(data_config.processed_path)

    if args.model in {"xgboost", "both"}:
        predictor = XGBoostPredictor.from_artifacts()
        report = generate_table_report(
            frame,
            model_name="xgboost",
            predict_fn=predictor.predict_from_frame,
            batch_predict_fn=predictor.predict_many_from_frame,
            output_dir=data_config.xgboost_artifact_dir / "reports",
        )
        print(f"xgboost row_level_report: {report.row_level_csv}")
        print(f"xgboost slice_summary_report: {report.slice_summary_csv}")
        print(f"xgboost overall_summary_report: {report.overall_summary_csv}")
        print(f"xgboost excel_report: {report.workbook_xlsx}")
        print(f"xgboost slice_plot: {report.slice_plot_png}")
        print(f"xgboost summary_plot: {report.summary_plot_png}")

    if args.model in {"ft_transformer", "both"}:
        predictor = FTTransformerPredictor.from_artifacts(device=args.ft_device)
        report = generate_table_report(
            frame,
            model_name="ft_transformer",
            predict_fn=predictor.predict_from_frame,
            batch_predict_fn=predictor.predict_many_from_frame,
            output_dir=data_config.ft_transformer_artifact_dir / "reports",
        )
        print(f"ft_transformer row_level_report: {report.row_level_csv}")
        print(f"ft_transformer slice_summary_report: {report.slice_summary_csv}")
        print(f"ft_transformer overall_summary_report: {report.overall_summary_csv}")
        print(f"ft_transformer excel_report: {report.workbook_xlsx}")
        print(f"ft_transformer slice_plot: {report.slice_plot_png}")
        print(f"ft_transformer summary_plot: {report.summary_plot_png}")


if __name__ == "__main__":
    main()
