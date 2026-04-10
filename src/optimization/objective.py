from __future__ import annotations

from dataclasses import dataclass

from src.data.preprocess import PreparedSplit
from src.evaluation.benchmark import benchmark_stub
from src.models.ft_transformer import FTTransformerTrainer
from src.utils.config import FTTransformerConfig, PSOConfig


@dataclass(slots=True)
class ObjectiveResult:
    """Scalarized objective and supporting diagnostics."""

    score: float
    rmse: float
    latency_ms: float
    model_size_mb: float
    metadata: dict[str, float | str]


def scalarized_cost(
    rmse: float,
    latency_ms: float,
    model_size_mb: float,
    config: PSOConfig,
) -> float:
    """Shared multi-objective scalarization used by PSO."""

    w1, w2, w3 = config.weights
    return float(
        w1 * (rmse / config.rmse_ref)
        + w2 * (latency_ms / config.latency_ref)
        + w3 * (model_size_mb / config.size_ref)
    )


def evaluate_ft_transformer_config(
    train_split: PreparedSplit,
    valid_split: PreparedSplit,
    categorical_cardinalities: list[int],
    ft_config: FTTransformerConfig,
    pso_config: PSOConfig,
) -> ObjectiveResult:
    """Train and score an FT-Transformer candidate configuration."""

    trainer = FTTransformerTrainer(
        num_numeric_features=train_split.X_num.shape[1],
        categorical_cardinalities=categorical_cardinalities,
        config=ft_config,
    )
    fit_result = trainer.fit(train_split=train_split, valid_split=valid_split)
    valid_metrics = trainer.evaluate(valid_split)
    benchmark = benchmark_stub(
        notes=[
            "Latency and size are placeholder values until ONNX/TensorRT benchmarking is added.",
            f"epochs_trained={fit_result.epochs_trained}",
        ]
    )
    parameter_size_mb = (trainer.parameter_count() * 4) / (1024 * 1024)
    score = scalarized_cost(
        rmse=valid_metrics["rmse"],
        latency_ms=benchmark.latency_ms,
        model_size_mb=parameter_size_mb,
        config=pso_config,
    )
    return ObjectiveResult(
        score=score,
        rmse=valid_metrics["rmse"],
        latency_ms=benchmark.latency_ms,
        model_size_mb=parameter_size_mb,
        metadata={
            "epochs_trained": float(fit_result.epochs_trained),
            "best_validation_rmse": fit_result.best_validation_rmse,
            "benchmark_note": "; ".join(benchmark.notes),
        },
    )
