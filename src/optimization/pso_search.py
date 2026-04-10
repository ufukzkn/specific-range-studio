from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.optimization.objective import ObjectiveResult
from src.utils.config import FTTransformerConfig, PSOConfig


@dataclass(slots=True)
class SearchResult:
    """Best PSO outcome and trace for later analysis."""

    best_position: np.ndarray
    best_score: float
    best_params: dict[str, float | int]
    history: list[dict[str, float]]


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


def run_pso(
    objective_fn: Callable[[np.ndarray], ObjectiveResult],
    config: PSOConfig | None = None,
) -> SearchResult:
    """Run a simple PSO loop over the FT-Transformer search space."""

    config = config or PSOConfig()
    rng = np.random.default_rng(config.random_state)
    bounds = FTTransformerSearchSpace.bounds
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
            history.append(
                {
                    "iteration": float(iteration),
                    "particle": float(idx),
                    "score": float(score),
                    "rmse": float(result.rmse),
                    "latency_ms": float(result.latency_ms),
                    "model_size_mb": float(result.model_size_mb),
                }
            )

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
        best_params=FTTransformerSearchSpace.project(global_best_position),
        history=history,
    )
