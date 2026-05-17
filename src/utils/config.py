from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    """Filesystem and schema configuration for the aviation dataset."""

    project_root: Path = Path(__file__).resolve().parents[2]
    one_engine_path: Path = project_root / "One_Engine_Data.xlsx"
    two_engine_path: Path = project_root / "Two_Engine_Data.xlsx"
    processed_dir: Path = project_root / "data" / "processed"
    artifacts_dir: Path = project_root / "artifacts"
    processed_filename: str = "combined_specific_range.csv"
    target_column: str = "specific_range"
    required_columns: tuple[str, ...] = (
        "altitude",
        "gross_weight",
        "drag_index",
        "mach",
        "fuel_flow",
        "engine_type",
        "specific_range",
    )
    categorical_features: tuple[str, ...] = ("engine_type",)
    numerical_features: tuple[str, ...] = (
        "altitude",
        "gross_weight",
        "drag_index",
        "mach",
        "fuel_flow",
    )

    @property
    def processed_path(self) -> Path:
        return self.processed_dir / self.processed_filename

    @property
    def xgboost_artifact_dir(self) -> Path:
        return self.artifacts_dir / "xgboost"

    @property
    def ft_transformer_artifact_dir(self) -> Path:
        return self.artifacts_dir / "ft_transformer"


@dataclass(slots=True)
class SplitConfig:
    """Train/validation/test split configuration."""

    train_size: float = 0.7
    valid_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


@dataclass(slots=True)
class PreprocessConfig:
    """Preprocessing behaviour shared across models."""

    clip_outliers: bool = False
    clip_quantile_low: float = 0.01
    clip_quantile_high: float = 0.99


@dataclass(slots=True)
class FTTransformerConfig:
    """Training and architecture settings for FT-Transformer."""

    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 50
    weight_decay: float = 1e-5
    patience: int = 8
    random_state: int = 42
    device: str = "cpu"


@dataclass(slots=True)
class XGBoostConfig:
    """Baseline XGBoost hyperparameters."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    tree_method: str = "hist"
    device: str = "cpu"


@dataclass(slots=True)
class PSOConfig:
    """Particle Swarm Optimization settings."""

    population_size: int = 8
    iterations: int = 10
    inertia: float = 0.7
    cognitive: float = 1.4
    social: float = 1.4
    weights: tuple[float, float, float] = (0.70, 0.20, 0.10)
    rmse_ref: float = 0.003
    latency_ref: float = 10.0
    size_ref: float = 1.0
    latency_repetitions: int = 30
    latency_warmup: int = 5
    random_state: int = 42


@dataclass(slots=True)
class BenchmarkConfig:
    """Future deployment hooks for export and edge benchmarking."""

    onnx_opset: int = 17
    tensorrt_precision: str = "fp16"
    latency_repetitions: int = 100
    notes: list[str] = field(default_factory=list)
