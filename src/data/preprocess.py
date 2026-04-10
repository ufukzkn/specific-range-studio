from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.config import DataConfig, PreprocessConfig


@dataclass(slots=True)
class PreprocessorState:
    """Fitted preprocessing components and metadata."""

    numerical_features: list[str]
    categorical_features: list[str]
    numeric_imputer: SimpleImputer
    categorical_imputer: SimpleImputer
    scaler: StandardScaler
    encoder: OrdinalEncoder
    clip_bounds: dict[str, tuple[float, float]] | None
    target_column: str


@dataclass(slots=True)
class PreparedSplit:
    """Transformed arrays ready for model training and evaluation."""

    X_num: np.ndarray
    X_cat: np.ndarray
    y: np.ndarray


def _compute_clip_bounds(
    frame: pd.DataFrame,
    columns: list[str],
    config: PreprocessConfig,
) -> dict[str, tuple[float, float]] | None:
    if not config.clip_outliers:
        return None
    bounds: dict[str, tuple[float, float]] = {}
    for column in columns:
        lower = float(frame[column].quantile(config.clip_quantile_low))
        upper = float(frame[column].quantile(config.clip_quantile_high))
        bounds[column] = (lower, upper)
    return bounds


def _apply_clip_bounds(frame: pd.DataFrame, bounds: dict[str, tuple[float, float]] | None) -> pd.DataFrame:
    frame = frame.copy()
    if not bounds:
        return frame
    for column, (lower, upper) in bounds.items():
        frame[column] = frame[column].clip(lower=lower, upper=upper)
    return frame


def fit_preprocessor(
    train_df: pd.DataFrame,
    data_config: DataConfig | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> PreprocessorState:
    """Fit imputers, encoders, and scalers using train data only."""

    data_config = data_config or DataConfig()
    preprocess_config = preprocess_config or PreprocessConfig()
    num_features = list(data_config.numerical_features)
    cat_features = list(data_config.categorical_features)
    clip_bounds = _compute_clip_bounds(train_df, num_features, preprocess_config)
    clipped = _apply_clip_bounds(train_df, clip_bounds)

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    num_values = numeric_imputer.fit_transform(clipped[num_features])
    cat_values = categorical_imputer.fit_transform(clipped[cat_features])
    scaler.fit(num_values)
    encoder.fit(cat_values)

    return PreprocessorState(
        numerical_features=num_features,
        categorical_features=cat_features,
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        scaler=scaler,
        encoder=encoder,
        clip_bounds=clip_bounds,
        target_column=data_config.target_column,
    )


def transform_split(frame: pd.DataFrame, state: PreprocessorState) -> PreparedSplit:
    """Apply a fitted preprocessing state to an arbitrary dataframe split."""

    transformed = _apply_clip_bounds(frame, state.clip_bounds)
    x_num = state.numeric_imputer.transform(transformed[state.numerical_features])
    x_num = state.scaler.transform(x_num)
    x_cat = state.categorical_imputer.transform(transformed[state.categorical_features])
    x_cat = state.encoder.transform(x_cat).astype(np.int64)
    y = transformed[state.target_column].to_numpy(dtype=np.float32)
    return PreparedSplit(X_num=x_num.astype(np.float32), X_cat=x_cat, y=y)


def combine_for_tree_model(prepared: PreparedSplit) -> np.ndarray:
    """Concatenate numerical and encoded categorical features for tree models."""

    return np.hstack([prepared.X_num, prepared.X_cat.astype(np.float32)])


def category_cardinalities(state: PreprocessorState) -> list[int]:
    """Return category counts expected by embedding-based models."""

    return [len(categories) + 1 for categories in state.encoder.categories_]
