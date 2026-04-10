from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nomogram_report import generate_nomogram_report
from src.inference.predictors import FTTransformerPredictor, XGBoostPredictor
from src.utils.config import DataConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a draft nomogram-style comparison report.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to processed CSV.")
    parser.add_argument("--model", choices=["xgboost", "ft_transformer"], required=True)
    parser.add_argument("--engine-type", choices=["one_engine", "two_engine"], required=True)
    parser.add_argument("--altitude", type=float, required=True)
    parser.add_argument("--gross-weight", type=float, required=True)
    parser.add_argument("--ft-device", type=str, default="cpu")
    args = parser.parse_args()

    data_config = DataConfig()
    frame = pd.read_csv(args.dataset) if args.dataset else pd.read_csv(data_config.processed_path)

    if args.model == "xgboost":
        predictor = XGBoostPredictor.from_artifacts()
        output_dir = data_config.xgboost_artifact_dir / "nomogram_reports"
        batch_predict_fn = predictor.predict_many_from_frame
    else:
        predictor = FTTransformerPredictor.from_artifacts(device=args.ft_device)
        output_dir = data_config.ft_transformer_artifact_dir / "nomogram_reports"
        batch_predict_fn = predictor.predict_many_from_frame

    result = generate_nomogram_report(
        frame,
        model_name=args.model,
        batch_predict_fn=batch_predict_fn,
        output_dir=output_dir,
        engine_type=args.engine_type,
        altitude=args.altitude,
        gross_weight=args.gross_weight,
    )

    print(f"row_level_report: {result.row_level_csv}")
    print(f"slice_summary_report: {result.slice_summary_csv}")
    print(f"nomogram_plot: {result.nomogram_png}")
    print("Note: This is a draft nomogram-style plot. It preserves categorical slice logic,")
    print("but it is not yet a pixel-faithful recreation of the original handbook chart.")


if __name__ == "__main__":
    main()
