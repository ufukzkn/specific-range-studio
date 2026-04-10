from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_combined_dataset
from src.data.preprocess import fit_preprocessor, transform_split
from src.data.split import split_dataframe
from src.evaluation.table_report import generate_table_report
from src.inference.predictors import model_card_payload, save_metadata
from src.models.xgboost_baseline import XGBoostRegressorWrapper
from src.utils.config import DataConfig, PreprocessConfig, SplitConfig, XGBoostConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the XGBoost baseline.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to a preprocessed CSV.")
    parser.add_argument("--device", type=str, default="cpu", help="XGBoost device: cpu or cuda.")
    parser.add_argument("--run-table-report", action="store_true", help="Generate full-table comparison report after training.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data_config = DataConfig()
    frame = pd.read_csv(args.dataset) if args.dataset else load_combined_dataset(data_config)
    splits = split_dataframe(frame, SplitConfig())
    state = fit_preprocessor(splits.train, data_config, PreprocessConfig())

    train = transform_split(splits.train, state)
    valid = transform_split(splits.valid, state)
    test = transform_split(splits.test, state)

    artifact_dir = data_config.xgboost_artifact_dir
    config = XGBoostConfig(
        device=args.device,
        tree_method="hist",
    )
    logging.info("XGBoost requested device: %s", args.device)
    model = XGBoostRegressorWrapper(config).fit(train=train, valid=valid)
    valid_metrics = model.evaluate(valid)
    test_metrics = model.evaluate(test)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, artifact_dir / "preprocessor.joblib")
    model.save(artifact_dir / "model.json", artifact_dir / "config.json")
    save_metadata(
        artifact_dir / "metrics.json",
        model_card_payload("xgboost", {"validation": valid_metrics, "test": test_metrics}, extra={"config": asdict(config)}),
    )

    if args.run_table_report:
        def predict_frame(single_frame: pd.DataFrame) -> float:
            prepared = transform_split(single_frame, state)
            return float(model.predict_from_prepared(prepared)[0])

        report = generate_table_report(
            frame,
            model_name="xgboost",
            predict_fn=predict_frame,
            batch_predict_fn=lambda batch_frame: model.predict_from_prepared(transform_split(batch_frame, state)),
            output_dir=artifact_dir / "reports",
        )
        print(f"row_level_report: {report.row_level_csv}")
        print(f"slice_summary_report: {report.slice_summary_csv}")
        print(f"overall_summary_report: {report.overall_summary_csv}")
        print(f"excel_report: {report.workbook_xlsx}")
        print(f"slice_plot: {report.slice_plot_png}")
        print(f"summary_plot: {report.summary_plot_png}")

    print("validation:", json.dumps(valid_metrics, indent=2))
    print("test:", json.dumps(test_metrics, indent=2))
    print(f"artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
