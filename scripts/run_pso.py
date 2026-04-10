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
from src.optimization.objective import evaluate_ft_transformer_config
from src.optimization.pso_search import FTTransformerSearchSpace, run_pso
from src.utils.config import DataConfig, FTTransformerConfig, PSOConfig, PreprocessConfig, SplitConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSO search for FT-Transformer hyperparameters.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to a preprocessed CSV.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--population", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data_config = DataConfig()
    frame = pd.read_csv(args.dataset) if args.dataset else load_combined_dataset(data_config)
    splits = split_dataframe(frame, SplitConfig())
    state = fit_preprocessor(splits.train, data_config, PreprocessConfig())

    train = transform_split(splits.train, state)
    valid = transform_split(splits.valid, state)
    pso_config = PSOConfig(population_size=args.population, iterations=args.iterations)
    base_ft_config = FTTransformerConfig(device=args.device)
    cat_cardinalities = category_cardinalities(state)

    def objective(position):
        ft_config = FTTransformerSearchSpace.to_config(position, base_ft_config)
        return evaluate_ft_transformer_config(train, valid, cat_cardinalities, ft_config, pso_config)

    result = run_pso(objective, pso_config)
    print(json.dumps({"best_score": result.best_score, "best_params": result.best_params}, indent=2))


if __name__ == "__main__":
    main()
