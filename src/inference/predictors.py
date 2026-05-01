from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import load_combined_dataset
from src.data.preprocess import PreprocessorState, transform_split
from src.models.ft_transformer import FTTransformerTrainer
from src.models.xgboost_baseline import XGBoostRegressorWrapper
from src.utils.config import DataConfig


def build_single_row_frame(
    *,
    altitude: float,
    gross_weight: float,
    drag_index: float,
    mach: float,
    fuel_flow: float,
    engine_type: str,
) -> pd.DataFrame:
    """Create a one-row canonical feature frame for UI or CLI prediction."""

    return pd.DataFrame(
        [
            {
                "altitude": altitude,
                "gross_weight": gross_weight,
                "drag_index": drag_index,
                "mach": mach,
                "fuel_flow": fuel_flow,
                "engine_type": engine_type,
                "specific_range": 0.0,
            }
        ]
    )


def _load_preprocessor(path: Path) -> PreprocessorState:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {path}")
    return joblib.load(path)


class XGBoostPredictor:
    """Artifact-backed XGBoost inference helper."""

    def __init__(self, model: XGBoostRegressorWrapper, preprocessor: PreprocessorState) -> None:
        self.model = model
        self.preprocessor = preprocessor

    @classmethod
    def from_artifacts(cls, artifact_dir: Path | None = None) -> "XGBoostPredictor":
        data_config = DataConfig()
        artifact_dir = artifact_dir or data_config.xgboost_artifact_dir
        model = XGBoostRegressorWrapper.load(artifact_dir / "model.json", artifact_dir / "config.json")
        preprocessor = _load_preprocessor(artifact_dir / "preprocessor.joblib")
        return cls(model=model, preprocessor=preprocessor)

    def predict_from_frame(self, frame: pd.DataFrame) -> float:
        prepared = transform_split(frame, self.preprocessor)
        prediction = self.model.predict_from_prepared(prepared)
        return float(prediction[0])

    def predict_many_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = transform_split(frame, self.preprocessor)
        return self.model.predict_from_prepared(prepared)


class FTTransformerPredictor:
    """Artifact-backed FT-Transformer inference helper."""

    def __init__(self, trainer: FTTransformerTrainer, preprocessor: PreprocessorState) -> None:
        self.trainer = trainer
        self.preprocessor = preprocessor

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: Path | None = None,
        *,
        device: str = "cpu",
    ) -> "FTTransformerPredictor":
        data_config = DataConfig()
        artifact_dir = artifact_dir or data_config.ft_transformer_artifact_dir
        trainer = FTTransformerTrainer.load_checkpoint(artifact_dir / "model.pt", device=device)
        preprocessor = _load_preprocessor(artifact_dir / "preprocessor.joblib")
        return cls(trainer=trainer, preprocessor=preprocessor)

    def predict_from_frame(self, frame: pd.DataFrame) -> float:
        prepared = transform_split(frame, self.preprocessor)
        prediction = self.trainer.predict(prepared.X_num, prepared.X_cat)
        return float(prediction[0])

    def predict_many_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = transform_split(frame, self.preprocessor)
        return self.trainer.predict(prepared.X_num, prepared.X_cat)


def save_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def model_card_payload(model_name: str, metrics: dict[str, dict[str, float]], extra: dict | None = None) -> dict:
    payload = {"model_name": model_name, "metrics": metrics}
    if extra:
        payload["extra"] = extra
    return payload


def config_payload(config) -> dict:
    return asdict(config)


def load_reference_dataset() -> pd.DataFrame:
    """Load the cleaned reference dataset, preferring the cached CSV when present."""

    data_config = DataConfig()
    if data_config.processed_path.exists():
        return pd.read_csv(data_config.processed_path)
    return load_combined_dataset(data_config)


def find_exact_match(
    reference_df: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """Return rows that exactly match the query values within tolerance."""

    query = query_frame.iloc[0]
    subset = reference_df[reference_df["engine_type"] == query["engine_type"]].copy()
    numeric_columns = ["altitude", "gross_weight", "drag_index", "mach", "fuel_flow"]
    mask = np.ones(len(subset), dtype=bool)
    for column in numeric_columns:
        mask &= np.isclose(subset[column].to_numpy(dtype=float), float(query[column]), atol=atol, rtol=0.0)
    return subset.loc[mask].reset_index(drop=True)


def find_nearest_reference_rows(
    reference_df: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Find nearest real rows for the current query using normalized numeric distance."""

    query = query_frame.iloc[0]
    subset = reference_df[reference_df["engine_type"] == query["engine_type"]].copy()
    numeric_columns = ["altitude", "gross_weight", "drag_index", "mach", "fuel_flow"]
    if subset.empty:
        return subset

    numeric_subset = subset[numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float)
    numeric_query = query_frame[numeric_columns].iloc[0].apply(pd.to_numeric, errors="coerce").astype(float)
    scales = numeric_subset.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    deltas = (numeric_subset - numeric_query) / scales
    subset["distance"] = np.sqrt(np.square(deltas.to_numpy(dtype=float)).sum(axis=1))
    return subset.sort_values("distance", ascending=True).head(top_k).reset_index(drop=True)


def build_test_scenarios(reference_df: pd.DataFrame) -> list[dict[str, object]]:
    """Create UI-friendly preset scenarios from real rows.

    Presets intentionally avoid synthetic intermediate points. In this UI the
    interpolation result is used as the custom-input reference, so exact real
    rows are clearer and less surprising for quick demos.
    """

    scenarios: list[dict[str, object]] = [
        {
            "name": "Custom input",
            "description": "Manually enter any flight condition.",
            "altitude": 11000.0,
            "gross_weight": 30000.0,
            "drag_index": 0.0,
            "mach": 0.22,
            "fuel_flow": 5000.0,
            "engine_type": "one_engine",
            "source": "manual",
        }
    ]

    if reference_df.empty:
        return scenarios

    seen_rows: set[tuple] = set()
    sorted_reference = reference_df.sort_values(
        ["engine_type", "altitude", "gross_weight", "drag_index", "mach", "fuel_flow"]
    ).reset_index(drop=True)

    def add_scenario(row: pd.Series, name: str, description: str) -> None:
        key = (
            float(row["altitude"]),
            float(row["gross_weight"]),
            float(row["drag_index"]),
            float(row["mach"]),
            float(row["fuel_flow"]),
            str(row["engine_type"]),
        )
        if key in seen_rows:
            return
        seen_rows.add(key)
        scenarios.append(
            {
                "name": name,
                "description": description,
                "altitude": float(row["altitude"]),
                "gross_weight": float(row["gross_weight"]),
                "drag_index": float(row["drag_index"]),
                "mach": float(row["mach"]),
                "fuel_flow": float(row["fuel_flow"]),
                "engine_type": str(row["engine_type"]),
                "source": "exact",
            }
        )

    for engine_type, engine_group in sorted_reference.groupby("engine_type", dropna=False):
        if engine_group.empty:
            continue
        engine_label = "Tek motor" if str(engine_type) == "one_engine" else "Cift motor"
        candidate_specs = (
            ("Dusuk irtifa / hafif konfigurasyon", 0.15, 0.20, 0.15),
            ("Orta irtifa / nominal konfigurasyon", 0.45, 0.50, 0.45),
            ("Yuksek irtifa / hizli cruise", 0.80, 0.75, 0.80),
        )
        for label, altitude_q, weight_q, mach_q in candidate_specs:
            target_altitude = engine_group["altitude"].quantile(altitude_q)
            altitude_group = engine_group.iloc[
                (engine_group["altitude"].astype(float) - float(target_altitude)).abs().argsort()[: max(20, len(engine_group) // 12)]
            ]
            target_weight = altitude_group["gross_weight"].quantile(weight_q)
            weight_group = altitude_group.iloc[
                (altitude_group["gross_weight"].astype(float) - float(target_weight)).abs().argsort()[: max(10, len(altitude_group) // 4)]
            ]
            target_mach = weight_group["mach"].quantile(mach_q)
            row = weight_group.iloc[(weight_group["mach"].astype(float) - float(target_mach)).abs().argsort().iloc[0]]
            add_scenario(
                row,
                f"{engine_label}: {label}",
                (
                    f"Gercek tablo satiri. actual specific_range="
                    f"{float(row['specific_range']):.6f}, altitude={int(row['altitude'])} ft."
                ),
            )

        # Add one high-drag / endurance-style exact row per engine when present.
        high_drag = engine_group[engine_group["drag_index"].astype(float) >= engine_group["drag_index"].quantile(0.80)]
        if not high_drag.empty:
            row = high_drag.iloc[(high_drag["specific_range"].astype(float) - high_drag["specific_range"].median()).abs().argsort().iloc[0]]
            add_scenario(
                row,
                f"{engine_label}: yuksek drag kontrol senaryosu",
                (
                    f"Gercek tablo satiri; yuksek drag bolgesinde model davranisini kontrol eder. "
                    f"actual specific_range={float(row['specific_range']):.6f}."
                ),
            )

    # Keep the list compact and demo-friendly.
    return scenarios[:10]


def list_engine_types(reference_df: pd.DataFrame) -> list[str]:
    """Return sorted engine types present in the reference dataset."""

    return sorted(reference_df["engine_type"].dropna().astype(str).unique().tolist())


def list_altitudes(reference_df: pd.DataFrame, engine_type: str | None = None) -> list[float]:
    """Return sorted altitude levels, optionally filtered by engine type."""

    subset = reference_df
    if engine_type:
        subset = subset[subset["engine_type"] == engine_type]
    return sorted(subset["altitude"].dropna().astype(float).unique().tolist())


def get_reference_slice(
    reference_df: pd.DataFrame,
    *,
    engine_type: str,
    altitude: float | None = None,
) -> pd.DataFrame:
    """Return a filtered real-data slice for table-based comparison."""

    subset = reference_df[reference_df["engine_type"] == engine_type].copy()
    if altitude is not None:
        subset = subset[np.isclose(subset["altitude"].astype(float), float(altitude), atol=1e-9, rtol=0.0)]
    return subset.sort_values(["gross_weight", "drag_index", "mach", "fuel_flow"]).reset_index(drop=True)


def add_reference_row_id(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable UI row id to the reference dataframe."""

    df = reference_df.copy().reset_index(drop=True)
    df["row_id"] = df.index.astype(int)
    return df
