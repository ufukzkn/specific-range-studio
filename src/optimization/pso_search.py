from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from src.optimization.objective import ObjectiveResult
from src.utils.config import FTTransformerConfig, PSOConfig, XGBoostConfig


@dataclass(slots=True)
class SearchResult:
    """Best PSO outcome and trace for later analysis."""

    best_position: np.ndarray
    best_score: float
    best_params: dict[str, float | int]
    history: list[dict[str, float]]
    pareto_front: list[dict[str, float]]


class FTTransformerSearchSpace:
    """Continuous search space plus projection into valid FT-Transformer hyperparameters."""

    bounds = np.array(
        [
            [2, 6],
            [1, 8],
            [32, 256],
            [64, 512],
            [0.0, 0.5],
            [1e-4, 5e-3],
        ],
        dtype=np.float64,
    )

    @classmethod
    def project(cls, position: np.ndarray) -> dict[str, float | int]:
        clipped = np.clip(position, cls.bounds[:, 0], cls.bounds[:, 1])
        n_layers = int(round(clipped[0]))
        n_heads = max(1, int(round(clipped[1])))
        d_model = int(round(clipped[2] / n_heads) * n_heads)
        d_model = max(n_heads, d_model)
        d_ff = int(round(clipped[3]))
        dropout = float(clipped[4])
        learning_rate = float(clipped[5])
        return {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_model": d_model,
            "d_ff": d_ff,
            "dropout": dropout,
            "learning_rate": learning_rate,
        }

    @classmethod
    def to_config(cls, position: np.ndarray, base_config: FTTransformerConfig) -> FTTransformerConfig:
        params = cls.project(position)
        return FTTransformerConfig(
            d_model=int(params["d_model"]),
            n_layers=int(params["n_layers"]),
            n_heads=int(params["n_heads"]),
            d_ff=int(params["d_ff"]),
            dropout=float(params["dropout"]),
            learning_rate=float(params["learning_rate"]),
            batch_size=base_config.batch_size,
            epochs=base_config.epochs,
            weight_decay=base_config.weight_decay,
            patience=base_config.patience,
            random_state=base_config.random_state,
            device=base_config.device,
        )


class SearchSpace(Protocol):
    """Minimal interface needed by the shared PSO loop."""

    bounds: np.ndarray

    @classmethod
    def project(cls, position: np.ndarray) -> dict[str, float | int]:
        ...


class XGBoostSearchSpace:
    """Continuous search space plus projection into valid XGBoost hyperparameters."""

    bounds = np.array(
        [
            [80, 800],
            [2, 10],
            [0.01, 0.20],
            [0.60, 1.0],
            [0.60, 1.0],
            [0.0, 1.0],
            [0.10, 5.0],
        ],
        dtype=np.float64,
    )

    @classmethod
    def project(cls, position: np.ndarray) -> dict[str, float | int]:
        clipped = np.clip(position, cls.bounds[:, 0], cls.bounds[:, 1])
        return {
            "n_estimators": int(round(clipped[0])),
            "max_depth": int(round(clipped[1])),
            "learning_rate": float(clipped[2]),
            "subsample": float(clipped[3]),
            "colsample_bytree": float(clipped[4]),
            "reg_alpha": float(clipped[5]),
            "reg_lambda": float(clipped[6]),
        }

    @classmethod
    def to_config(cls, position: np.ndarray, base_config: XGBoostConfig) -> XGBoostConfig:
        params = cls.project(position)
        return XGBoostConfig(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=base_config.random_state,
            objective=base_config.objective,
            eval_metric=base_config.eval_metric,
            tree_method=base_config.tree_method,
            device=base_config.device,
        )


def run_pso(
    objective_fn: Callable[[np.ndarray], ObjectiveResult],
    config: PSOConfig | None = None,
    search_space: type[SearchSpace] = FTTransformerSearchSpace,
) -> SearchResult:
    """Run a simple PSO loop over a projected hyperparameter search space."""

    config = config or PSOConfig()
    rng = np.random.default_rng(config.random_state)
    bounds = search_space.bounds
    low, high = bounds[:, 0], bounds[:, 1]

    positions = rng.uniform(low=low, high=high, size=(config.population_size, bounds.shape[0]))
    velocities = np.zeros_like(positions)
    personal_best_positions = positions.copy()
    personal_best_scores = np.full(config.population_size, np.inf, dtype=np.float64)
    global_best_position = positions[0].copy()
    global_best_score = float("inf")
    history: list[dict[str, float]] = []

    for iteration in range(config.iterations):
        for idx in range(config.population_size):
            result = objective_fn(positions[idx])
            score = result.score
            if score < personal_best_scores[idx]:
                personal_best_scores[idx] = score
                personal_best_positions[idx] = positions[idx].copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[idx].copy()
            params = search_space.project(positions[idx])
            row = {
                "iteration": float(iteration),
                "particle": float(idx),
                "score": float(score),
                "rmse": float(result.rmse),
                "mae": float(result.mae),
                "mape": float(result.mape),
                "r2": float(result.r2),
                "latency_ms": float(result.latency_ms),
                "model_size_mb": float(result.model_size_mb),
                "param_count": float(result.param_count),
                "rmse_component": float(result.score_components.get("rmse_component", 0.0)),
                "latency_component": float(result.score_components.get("latency_component", 0.0)),
                "size_component": float(result.score_components.get("size_component", 0.0)),
                "weighted_rmse": float(result.score_components.get("weighted_rmse", 0.0)),
                "weighted_latency": float(result.score_components.get("weighted_latency", 0.0)),
                "weighted_size": float(result.score_components.get("weighted_size", 0.0)),
            }
            row.update({key: float(value) for key, value in params.items()})
            history.append(row)

        r1 = rng.random(size=velocities.shape)
        r2 = rng.random(size=velocities.shape)
        velocities = (
            config.inertia * velocities
            + config.cognitive * r1 * (personal_best_positions - positions)
            + config.social * r2 * (global_best_position - positions)
        )
        positions = np.clip(positions + velocities, low, high)

    return SearchResult(
        best_position=global_best_position,
        best_score=float(global_best_score),
        best_params=search_space.project(global_best_position),
        history=history,
        pareto_front=non_dominated_history(history),
    )


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    objectives = ("rmse", "latency_ms", "model_size_mb")
    no_worse = all(float(left[key]) <= float(right[key]) for key in objectives)
    strictly_better = any(float(left[key]) < float(right[key]) for key in objectives)
    return no_worse and strictly_better


def non_dominated_history(history: list[dict[str, float]]) -> list[dict[str, float]]:
    """Return Pareto-like diagnostic candidates from scalarized PSO history."""

    front: list[dict[str, float]] = []
    for candidate in history:
        if not any(_dominates(other, candidate) for other in history):
            front.append(candidate)
    return sorted(front, key=lambda item: float(item["score"]))
