from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from src.data.preprocess import PreparedSplit, combine_for_tree_model
from src.evaluation.metrics import regression_metrics
from src.utils.config import XGBoostConfig


class XGBoostRegressorWrapper:
    """Thin wrapper around xgboost.XGBRegressor with project-friendly defaults."""

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        self.config = config or XGBoostConfig()
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the baseline model. Install requirements.txt first."
            ) from exc

        self.model = XGBRegressor(**asdict(self.config))

    def fit(self, train: PreparedSplit, valid: PreparedSplit | None = None) -> "XGBoostRegressorWrapper":
        X_train = combine_for_tree_model(train)
        eval_set = None
        if valid is not None:
            eval_set = [(combine_for_tree_model(valid), valid.y)]
        self.model.fit(X_train, train.y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_from_prepared(self, split: PreparedSplit) -> np.ndarray:
        return self.predict(combine_for_tree_model(split))

    def evaluate(self, split: PreparedSplit) -> dict[str, float]:
        predictions = self.predict_from_prepared(split)
        return regression_metrics(split.y, predictions)

    def save(self, model_path: Path, config_path: Path | None = None) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        if config_path is not None:
            config_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_path: Path, config_path: Path | None = None) -> "XGBoostRegressorWrapper":
        config = XGBoostConfig()
        if config_path is not None and config_path.exists():
            config = XGBoostConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        wrapper = cls(config)
        wrapper.model.load_model(model_path)
        return wrapper
