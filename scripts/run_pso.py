from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_combined_dataset
from src.data.preprocess import category_cardinalities, fit_preprocessor, transform_split
from src.data.split import split_dataframe
from src.optimization.objective import evaluate_ft_transformer_config, evaluate_xgboost_config
from src.optimization.pso_search import FTTransformerSearchSpace, XGBoostSearchSpace, run_pso
from src.utils.config import DataConfig, FTTransformerConfig, PSOConfig, PreprocessConfig, SplitConfig, XGBoostConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deployment-aware PSO hyperparameter search.")
    parser.add_argument(
        "--model",
        choices=["ft_transformer", "xgboost"],
        default="ft_transformer",
        help="Model family to optimize.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to a preprocessed CSV.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--w-rmse", type=float, default=0.70, help="RMSE objective weight.")
    parser.add_argument("--w-latency", type=float, default=0.20, help="Latency objective weight.")
    parser.add_argument("--w-size", type=float, default=0.10, help="Model size objective weight.")
    parser.add_argument("--rmse-ref", type=float, default=0.003, help="RMSE normalization reference.")
    parser.add_argument("--latency-ref", type=float, default=10.0, help="Latency normalization reference in ms.")
    parser.add_argument("--size-ref", type=float, default=1.0, help="Model size normalization reference in MB.")
    parser.add_argument("--latency-repetitions", type=int, default=30, help="Single-row inference repetitions per candidate.")
    parser.add_argument("--latency-warmup", type=int, default=5, help="Warmup predictions per candidate.")
    parser.add_argument("--ft-epochs", type=int, default=50, help="FT-Transformer epochs per candidate.")
    parser.add_argument("--ft-patience", type=int, default=8, help="FT-Transformer early stopping patience.")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "pso", help="Directory for PSO JSON/CSV outputs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data_config = DataConfig()
    frame = pd.read_csv(args.dataset) if args.dataset else load_combined_dataset(data_config)
    splits = split_dataframe(frame, SplitConfig())
    state = fit_preprocessor(splits.train, data_config, PreprocessConfig())

    train = transform_split(splits.train, state)
    valid = transform_split(splits.valid, state)
    pso_config = PSOConfig(
        population_size=args.population,
        iterations=args.iterations,
        weights=(args.w_rmse, args.w_latency, args.w_size),
        rmse_ref=args.rmse_ref,
        latency_ref=args.latency_ref,
        size_ref=args.size_ref,
        latency_repetitions=args.latency_repetitions,
        latency_warmup=args.latency_warmup,
    )
    base_ft_config = FTTransformerConfig(device=args.device, epochs=args.ft_epochs, patience=args.ft_patience)
    base_xgb_config = XGBoostConfig(device=args.device)
    cat_cardinalities = category_cardinalities(state)

    def objective(position):
        if args.model == "xgboost":
            xgb_config = XGBoostSearchSpace.to_config(position, base_xgb_config)
            return evaluate_xgboost_config(train, valid, xgb_config, pso_config)
        ft_config = FTTransformerSearchSpace.to_config(position, base_ft_config)
        return evaluate_ft_transformer_config(train, valid, cat_cardinalities, ft_config, pso_config)

    search_space = XGBoostSearchSpace if args.model == "xgboost" else FTTransformerSearchSpace
    result = run_pso(objective, pso_config, search_space=search_space)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(result.history)
    history_csv = output_dir / f"{args.model}_pso_history_{run_id}.csv"
    latest_history_csv = output_dir / f"{args.model}_pso_history_latest.csv"
    history_frame.to_csv(history_csv, index=False)
    history_frame.to_csv(latest_history_csv, index=False)
    raw_weights = [max(args.w_rmse, 0.0), max(args.w_latency, 0.0), max(args.w_size, 0.0)]
    weight_total = sum(raw_weights) or 1.0
    normalized_weights = [value / weight_total for value in raw_weights]

    payload = {
        "run_id": run_id,
        "model": args.model,
        "dataset": str(args.dataset) if args.dataset else str(data_config.processed_path),
        "best_score": result.best_score,
        "best_params": result.best_params,
        "best_position": result.best_position.tolist(),
        "objective": {
            "formula": "w_rmse*RMSE/RMSE_ref + w_latency*T_inf/T_ref + w_size*S/S_ref",
            "weights": {
                "rmse": normalized_weights[0],
                "latency": normalized_weights[1],
                "size": normalized_weights[2],
            },
            "references": {
                "rmse_ref": args.rmse_ref,
                "latency_ref_ms": args.latency_ref,
                "size_ref_mb": args.size_ref,
            },
        },
        "history_csv": str(history_csv),
        "latest_history_csv": str(latest_history_csv),
        "history": result.history,
        "pareto_front": result.pareto_front,
    }
    run_json = output_dir / f"{args.model}_pso_result_{run_id}.json"
    latest_json = output_dir / f"{args.model}_pso_latest.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    run_json.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    legacy_latest_json = output_dir / "pso_latest.json"
    legacy_latest_json.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved PSO result: {latest_json}")


if __name__ == "__main__":
    main()
