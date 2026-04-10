from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import DataConfig

LOGGER = logging.getLogger(__name__)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def canonicalize_text(value: object) -> str:
    """Normalize arbitrary labels to snake_case ASCII text."""

    text = _strip_accents(str(value).strip().lower())
    text = text.replace("%", "percent")
    text = re.sub(r"[()/\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


COLUMN_ALIASES: dict[str, str] = {
    "altitude_ft": "altitude",
    "altitude": "altitude",
    "irtifa": "altitude",
    "gross_weight_lb": "gross_weight",
    "gross_weight": "gross_weight",
    "gross_agirlik": "gross_weight",
    "agirlik": "gross_weight",
    "drag_index": "drag_index",
    "surukleme_indeksi": "drag_index",
    "mach": "mach",
    "mach_number_ma": "mach",
    "mach_number": "mach",
    "mach_sayisi": "mach",
    "specific_range_nm": "specific_range",
    "specific_range": "specific_range",
    "ozgul_menzil": "specific_range",
    "fuel_flow_lb_h": "fuel_flow",
    "fuel_flow_lb_per_h": "fuel_flow",
    "fuel_flow": "fuel_flow",
    "yakit_akisi": "fuel_flow",
    "engine_type": "engine_type",
}


@dataclass(slots=True)
class WorkbookSpec:
    """Workbook metadata used by the loader."""

    path: Path
    engine_type: str


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert mixed-format numeric text to floats."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    as_text = (
        series.astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "None": None, "-": None})
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(as_text, errors="coerce")


def extract_altitude_from_sheet(sheet_name: str) -> float | None:
    """Extract altitude in feet from common sheet naming patterns."""

    normalized = canonicalize_text(sheet_name)
    if "sea_level" in normalized:
        return 0.0
    match = re.search(r"(\d[\d,\.]*)", sheet_name)
    if not match:
        return None
    return float(match.group(1).replace(",", "").replace(".", ""))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize workbook column names into a canonical schema."""

    renamed: dict[str, str] = {}
    for column in frame.columns:
        normalized = canonicalize_text(column)
        renamed[column] = COLUMN_ALIASES.get(normalized, normalized)
    return frame.rename(columns=renamed)


def _drop_empty_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(axis=0, how="all").reset_index(drop=True)


def _select_relevant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in frame.columns if column in set(COLUMN_ALIASES.values())]
    return frame.loc[:, keep]


def _validate_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")


def read_workbook(spec: WorkbookSpec) -> pd.DataFrame:
    """Read all parseable sheets from a workbook and concatenate them."""

    if not spec.path.exists():
        raise FileNotFoundError(f"Workbook not found: {spec.path}")

    try:
        workbook = pd.ExcelFile(spec.path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to read the Excel workbooks. "
            "Install dependencies from requirements.txt before running the pipeline."
        ) from exc

    frames: list[pd.DataFrame] = []
    for sheet_name in workbook.sheet_names:
        try:
            raw = pd.read_excel(spec.path, sheet_name=sheet_name, engine="openpyxl")
        except Exception as exc:
            LOGGER.warning("Skipping sheet '%s' from %s: %s", sheet_name, spec.path.name, exc)
            continue

        raw = _drop_empty_rows(raw)
        if raw.empty:
            LOGGER.warning("Skipping empty sheet '%s' from %s", sheet_name, spec.path.name)
            continue

        normalized = normalize_columns(raw)
        normalized = _select_relevant_columns(normalized)
        if normalized.empty:
            LOGGER.warning("Skipping schema-less sheet '%s' from %s", sheet_name, spec.path.name)
            continue

        altitude_from_sheet = extract_altitude_from_sheet(sheet_name)
        if "altitude" not in normalized.columns:
            normalized["altitude"] = altitude_from_sheet
        else:
            normalized["altitude"] = normalized["altitude"].fillna(altitude_from_sheet)

        normalized["engine_type"] = spec.engine_type
        frames.append(normalized)

    if not frames:
        raise ValueError(f"No usable sheets were found in workbook: {spec.path}")

    return pd.concat(frames, ignore_index=True)


def clean_combined_dataframe(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Finalize numeric parsing, schema validation, and row filtering."""

    frame = frame.copy()
    frame = normalize_columns(frame)
    frame = _drop_empty_rows(frame)

    for column in config.numerical_features + (config.target_column,):
        if column in frame.columns:
            frame[column] = _safe_to_numeric(frame[column])

    frame["engine_type"] = frame["engine_type"].astype(str).str.strip().str.lower()
    _validate_required_columns(frame, config.required_columns)

    before = len(frame)
    frame = frame.dropna(subset=list(config.required_columns))
    dropped = before - len(frame)
    if dropped > 0:
        LOGGER.info("Dropped %s rows with missing required values.", dropped)

    if frame.empty:
        raise ValueError("Combined dataframe is empty after cleaning.")

    return frame.reset_index(drop=True)


def load_combined_dataset(config: DataConfig | None = None) -> pd.DataFrame:
    """Load and merge one-engine and two-engine workbook rows."""

    config = config or DataConfig()
    specs = (
        WorkbookSpec(config.one_engine_path, "one_engine"),
        WorkbookSpec(config.two_engine_path, "two_engine"),
    )
    frames = [read_workbook(spec) for spec in specs]
    combined = pd.concat(frames, ignore_index=True)
    return clean_combined_dataframe(combined, config)


def save_processed_dataset(frame: pd.DataFrame, config: DataConfig | None = None) -> Path:
    """Persist the combined dataset for reproducible downstream training."""

    config = config or DataConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.processed_path, index=False)
    return config.processed_path
