from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE while protecting against division by zero."""

    denominator = np.clip(np.abs(y_true), a_min=1e-8, a_max=None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the standard regression metric bundle used in the project."""

    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
