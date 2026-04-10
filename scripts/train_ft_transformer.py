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
from src.data.preprocess import category_cardinalities, fit_preprocessor, transform_split
from src.data.split import split_dataframe
from src.evaluation.table_report import generate_table_report
from src.inference.predictors import model_card_payload, save_metadata
from src.models.ft_transformer import FTTransformerTrainer
from src.utils.config import DataConfig, FTTransformerConfig, PreprocessConfig, SplitConfig
from src.utils.device import resolve_torch_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the FT-Transformer regressor.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to a preprocessed CSV.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
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

    resolved_device = resolve_torch_device(args.device)
    if args.device != resolved_device:
        logging.warning("Requested device '%s' is unavailable. Falling back to '%s'.", args.device, resolved_device)
    else:
        logging.info("FT-Transformer will train on device: %s", resolved_device)

    try:
        import torch

        logging.info("torch version: %s", torch.__version__)
        logging.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logging.info("CUDA device: %s", torch.cuda.get_device_name(0))
    except ImportError:
        logging.warning("PyTorch import failed while checking runtime device details.")

    config = FTTransformerConfig(device=resolved_device)
    trainer = FTTransformerTrainer(
        num_numeric_features=train.X_num.shape[1],
        categorical_cardinalities=category_cardinalities(state),
        config=config,
    )
    fit_result = trainer.fit(train, valid)
    valid_metrics = trainer.evaluate(valid)
    test_metrics = trainer.evaluate(test)

    artifact_dir = data_config.ft_transformer_artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, artifact_dir / "preprocessor.joblib")
    trainer.save_checkpoint(
        artifact_dir / "model.pt",
        extra_metadata={
            "fit_result": asdict(fit_result),
            "validation_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    )
    save_metadata(
        artifact_dir / "metrics.json",
        model_card_payload("ft_transformer", {"validation": valid_metrics, "test": test_metrics}, extra={"config": asdict(config)}),
    )

    if args.run_table_report:
        def predict_frame(single_frame: pd.DataFrame) -> float:
            prepared = transform_split(single_frame, state)
            return float(trainer.predict(prepared.X_num, prepared.X_cat)[0])

        def predict_batch(batch_frame: pd.DataFrame):
            prepared = transform_split(batch_frame, state)
            return trainer.predict(prepared.X_num, prepared.X_cat)

        report = generate_table_report(
            frame,
            model_name="ft_transformer",
            predict_fn=predict_frame,
            batch_predict_fn=predict_batch,
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
