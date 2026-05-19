from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.data.preprocess import PreparedSplit, combine_for_tree_model
from src.models.ft_transformer import FTTransformerTrainer
from src.models.xgboost_baseline import XGBoostRegressorWrapper
from src.utils.config import FTTransformerConfig, PSOConfig, XGBoostConfig


@dataclass(slots=True)
class ObjectiveResult:
    """Scalarized objective and supporting diagnostics."""

    score: float
    rmse: float
    mae: float
    mape: float
    r2: float
    latency_ms: float
    model_size_mb: float
    param_count: int
    score_components: dict[str, float]
    metadata: dict[str, float | str]


def _normalize_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    clean = tuple(max(float(value), 0.0) for value in weights)
    total = sum(clean)
    if total <= 0:
        return (1.0, 0.0, 0.0)
    return tuple(value / total for value in clean)


def _safe_component(value: float, reference: float) -> float:
    reference = float(reference)
    if reference <= 0:
        reference = 1.0
    return float(value / reference)


def score_components(
    rmse: float,
    latency_ms: float,
    model_size_mb: float,
    config: PSOConfig,
) -> dict[str, float]:
    """Normalize objective terms and apply configured PSO weights."""

    w_rmse, w_latency, w_size = _normalize_weights(config.weights)
    rmse_component = _safe_component(rmse, config.rmse_ref)
    latency_component = _safe_component(latency_ms, config.latency_ref)
    size_component = _safe_component(model_size_mb, config.size_ref)
    return {
        "rmse_component": rmse_component,
        "latency_component": latency_component,
        "size_component": size_component,
        "weighted_rmse": w_rmse * rmse_component,
        "weighted_latency": w_latency * latency_component,
        "weighted_size": w_size * size_component,
        "weight_rmse": w_rmse,
        "weight_latency": w_latency,
        "weight_size": w_size,
    }


def scalarized_cost(
    rmse: float,
    latency_ms: float,
    model_size_mb: float,
    config: PSOConfig,
) -> float:
    """Shared multi-objective scalarization used by PSO."""

    components = score_components(
        rmse=rmse,
        latency_ms=latency_ms,
        model_size_mb=model_size_mb,
        config=config,
    )
    return float(components["weighted_rmse"] + components["weighted_latency"] + components["weighted_size"])


def _synchronize_if_needed(trainer: FTTransformerTrainer) -> None:
    try:
        import torch

        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
    except Exception:
        return


def measure_trainer_latency_ms(
    trainer: FTTransformerTrainer,
    split: PreparedSplit,
    *,
    repetitions: int,
    warmup: int,
) -> float:
    """Measure single-row FT-Transformer inference latency for a trained candidate."""

    if len(split.y) == 0:
        return 0.0
    x_num = np.asarray(split.X_num[:1])
    x_cat = np.asarray(split.X_cat[:1])
    warmup = max(0, int(warmup))
    repetitions = max(1, int(repetitions))

    for _ in range(warmup):
        trainer.predict(x_num, x_cat)
    _synchronize_if_needed(trainer)

    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        trainer.predict(x_num, x_cat)
        _synchronize_if_needed(trainer)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.percentile(np.asarray(timings, dtype=float), 95))


def measure_xgboost_latency_ms(
    model: XGBoostRegressorWrapper,
    split: PreparedSplit,
    *,
    repetitions: int,
    warmup: int,
) -> float:
    """Measure single-row XGBoost inference p95 latency for a trained candidate."""

    if len(split.y) == 0:
        return 0.0
    sample = combine_for_tree_model(split)[:1]
    warmup = max(0, int(warmup))
    repetitions = max(1, int(repetitions))

    for _ in range(warmup):
        model.predict(sample)

    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        model.predict(sample)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.percentile(np.asarray(timings, dtype=float), 95))


def estimate_xgboost_size_mb(model: XGBoostRegressorWrapper) -> float:
    """Estimate serialized booster size without writing candidate artifacts to disk."""

    try:
        raw = model.model.get_booster().save_raw()
        return float(len(raw) / (1024 * 1024))
    except Exception:
        dump = "\n".join(model.model.get_booster().get_dump())
        return float(len(dump.encode("utf-8")) / (1024 * 1024))


def estimate_xgboost_node_count(model: XGBoostRegressorWrapper) -> int:
    """Return a lightweight structural complexity proxy for XGBoost candidates."""

    try:
        frame = model.model.get_booster().trees_to_dataframe()
        return int(len(frame))
    except Exception:
        try:
            return int(model.model.get_booster().num_boosted_rounds())
        except Exception:
            return 0


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
    latency_ms = measure_trainer_latency_ms(
        trainer,
        valid_split,
        repetitions=pso_config.latency_repetitions,
        warmup=pso_config.latency_warmup,
    )
    param_count = trainer.parameter_count()
    parameter_size_mb = (param_count * 4) / (1024 * 1024)
    components = score_components(
        rmse=valid_metrics["rmse"],
        latency_ms=latency_ms,
        model_size_mb=parameter_size_mb,
        config=pso_config,
    )
    score = scalarized_cost(
        rmse=valid_metrics["rmse"],
        latency_ms=latency_ms,
        model_size_mb=parameter_size_mb,
        config=pso_config,
    )
    return ObjectiveResult(
        score=score,
        rmse=valid_metrics["rmse"],
        mae=valid_metrics["mae"],
        mape=valid_metrics["mape"],
        r2=valid_metrics["r2"],
        latency_ms=latency_ms,
        model_size_mb=parameter_size_mb,
        param_count=param_count,
        score_components=components,
        metadata={
            "epochs_trained": float(fit_result.epochs_trained),
            "best_validation_rmse": fit_result.best_validation_rmse,
            "benchmark_note": "Measured single-row p95 latency during PSO objective evaluation.",
        },
    )


def evaluate_xgboost_config(
    train_split: PreparedSplit,
    valid_split: PreparedSplit,
    xgboost_config: XGBoostConfig,
    pso_config: PSOConfig,
) -> ObjectiveResult:
    """Train and score an XGBoost candidate with the shared deployment-aware objective."""

    model = XGBoostRegressorWrapper(xgboost_config).fit(train_split, valid_split)
    valid_metrics = model.evaluate(valid_split)
    latency_ms = measure_xgboost_latency_ms(
        model,
        valid_split,
        repetitions=pso_config.latency_repetitions,
        warmup=pso_config.latency_warmup,
    )
    model_size_mb = estimate_xgboost_size_mb(model)
    node_count = estimate_xgboost_node_count(model)
    components = score_components(
        rmse=valid_metrics["rmse"],
        latency_ms=latency_ms,
        model_size_mb=model_size_mb,
        config=pso_config,
    )
    score = scalarized_cost(
        rmse=valid_metrics["rmse"],
        latency_ms=latency_ms,
        model_size_mb=model_size_mb,
        config=pso_config,
    )
    return ObjectiveResult(
        score=score,
        rmse=valid_metrics["rmse"],
        mae=valid_metrics["mae"],
        mape=valid_metrics["mape"],
        r2=valid_metrics["r2"],
        latency_ms=latency_ms,
        model_size_mb=model_size_mb,
        param_count=node_count,
        score_components=components,
        metadata={
            "node_count": float(node_count),
            "benchmark_note": "Measured single-row p95 latency during XGBoost PSO objective evaluation.",
        },
    )
