from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_combined_dataset
from src.data.preprocess import category_cardinalities, fit_preprocessor, transform_split
from src.data.split import split_dataframe
from src.models.ft_transformer import FTTransformerTrainer
from src.models.xgboost_baseline import XGBoostRegressorWrapper
from src.optimization.objective import evaluate_ft_transformer_config
from src.optimization.pso_search import FTTransformerSearchSpace, run_pso
from src.utils.config import DataConfig, FTTransformerConfig, PSOConfig, PreprocessConfig, SplitConfig, XGBoostConfig


def _format_row(model_name: str, split_name: str, metrics: dict[str, float]) -> str:
    return (
        f"{model_name:<18} {split_name:<10} "
        f"RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} "
        f"R2={metrics['r2']:.6f} MAPE={metrics['mape']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare XGBoost and FT-Transformer.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to a preprocessed CSV.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--run-pso", action="store_true", help="Optionally run a short FT-Transformer PSO search.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data_config = DataConfig()
    frame = pd.read_csv(args.dataset) if args.dataset else load_combined_dataset(data_config)
    splits = split_dataframe(frame, SplitConfig())
    state = fit_preprocessor(splits.train, data_config, PreprocessConfig())

    train = transform_split(splits.train, state)
    valid = transform_split(splits.valid, state)
    test = transform_split(splits.test, state)

    xgb = XGBoostRegressorWrapper(XGBoostConfig()).fit(train, valid)
    ft = FTTransformerTrainer(
        num_numeric_features=train.X_num.shape[1],
        categorical_cardinalities=category_cardinalities(state),
        config=FTTransformerConfig(device=args.device),
    )
    ft.fit(train, valid)

    comparison_rows = [
        _format_row("xgboost", "validation", xgb.evaluate(valid)),
        _format_row("xgboost", "test", xgb.evaluate(test)),
        _format_row("ft_transformer", "validation", ft.evaluate(valid)),
        _format_row("ft_transformer", "test", ft.evaluate(test)),
    ]

    if args.run_pso:
        pso_config = PSOConfig(population_size=4, iterations=3)
        base_ft_config = FTTransformerConfig(device=args.device)

        def objective(position):
            ft_config = FTTransformerSearchSpace.to_config(position, base_ft_config)
            return evaluate_ft_transformer_config(
                train, valid, category_cardinalities(state), ft_config, pso_config
            )

        pso_result = run_pso(objective, pso_config)
        comparison_rows.append(f"pso_best_params      search     {json.dumps(pso_result.best_params)}")

    print("\n".join(comparison_rows))


if __name__ == "__main__":
    main()
