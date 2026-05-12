from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.config import DataConfig

InterpolationMethod = Literal["spline", "linear", "newton"]

DEFAULT_INTERPOLATION_METHOD: InterpolationMethod = "spline"
INTERPOLATION_METHODS: dict[str, str] = {
    "spline": "Cubic Spline",
    "linear": "Piecewise Linear",
    "newton": "Newton Divided Difference",
}


def _unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order].astype(float)
    y_sorted = y[order].astype(float)
    if len(x_sorted) <= 1:
        return x_sorted, y_sorted
    keep = np.concatenate(([True], np.diff(x_sorted) > 0))
    return x_sorted[keep], y_sorted[keep]


class LinearInterpolator:
    def __init__(self, x, y) -> None:
        self.x, self.y = _unique_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if len(self.x) < 1:
            raise ValueError("At least one point is required for interpolation.")

    def evaluate(self, x_value: float) -> float:
        if len(self.x) == 1:
            return float(max(0.0, self.y[0]))
        return float(max(0.0, np.interp(float(x_value), self.x, self.y)))


class NewtonInterpolator:
    def __init__(self, x, y) -> None:
        self.x, y_values = _unique_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if len(self.x) < 1:
            raise ValueError("At least one point is required for interpolation.")
        self.y = y_values
        self.coefficients = self._calculate_divided_differences(self.x, y_values)

    @staticmethod
    def _calculate_divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        coefficients = y.copy()
        n = len(x)
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                denom = x[i] - x[i - j]
                if denom == 0:
                    coefficients[i] = 0.0
                else:
                    coefficients[i] = (coefficients[i] - coefficients[i - 1]) / denom
        return coefficients

    def evaluate(self, x_value: float) -> float:
        exact = np.where(np.isclose(self.x, float(x_value), atol=1e-12, rtol=0.0))[0]
        if exact.size:
            return float(max(0.0, self.y[int(exact[0])]))
        x_nodes = self.x
        coefs = self.coefficients
        if len(self.x) > 8:
            # A local Newton window keeps the method useful for dense tables and
            # avoids the Runge-style blow-ups of one huge global polynomial.
            nearest = np.argsort(np.abs(self.x - float(x_value)))[:6]
            nearest = np.sort(nearest)
            x_nodes = self.x[nearest]
            coefs = self._calculate_divided_differences(x_nodes, self.y[nearest])

        n = len(x_nodes)
        result = float(coefs[n - 1])
        for i in range(n - 2, -1, -1):
            result = result * (float(x_value) - x_nodes[i]) + coefs[i]
        return float(max(0.0, result))


class CubicSplineInterpolator:
    def __init__(self, x, y) -> None:
        self.x, self.y = _unique_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self.n = len(self.x)
        if self.n < 2:
            raise ValueError("At least two points are required for cubic spline interpolation.")
        if self.n == 2:
            self._linear = True
            self._slope = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
            return

        self._linear = False
        h = np.diff(self.x)
        inner = self.n - 2
        lower = np.zeros(inner, dtype=float)
        diag = np.zeros(inner, dtype=float)
        upper = np.zeros(inner, dtype=float)
        rhs = np.zeros(inner, dtype=float)

        for i in range(inner):
            idx = i + 1
            diag[i] = 2.0 * (h[idx - 1] + h[idx])
            rhs[i] = 6.0 * (
                (self.y[idx + 1] - self.y[idx]) / h[idx]
                - (self.y[idx] - self.y[idx - 1]) / h[idx - 1]
            )
            if i > 0:
                lower[i] = h[idx - 1]
            if i < inner - 1:
                upper[i] = h[idx]

        if inner == 1:
            moments_inner = np.array([rhs[0] / diag[0]], dtype=float)
        else:
            moments_inner = _solve_tridiagonal(lower, diag, upper, rhs)

        moments = np.zeros(self.n, dtype=float)
        moments[1 : self.n - 1] = moments_inner
        self._a = self.y[:-1].copy()
        self._b = np.zeros(self.n - 1, dtype=float)
        self._c = np.zeros(self.n - 1, dtype=float)
        self._d = np.zeros(self.n - 1, dtype=float)

        for i in range(self.n - 1):
            self._b[i] = ((self.y[i + 1] - self.y[i]) / h[i]) - h[i] * (2.0 * moments[i] + moments[i + 1]) / 6.0
            self._c[i] = moments[i] / 2.0
            self._d[i] = (moments[i + 1] - moments[i]) / (6.0 * h[i])

    def evaluate(self, x_value: float) -> float:
        x_value = float(x_value)
        if self._linear:
            return float(max(0.0, self.y[0] + self._slope * (x_value - self.x[0])))
        if x_value < self.x[0]:
            slope = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
            return float(max(0.0, self.y[0] + slope * (x_value - self.x[0])))
        if x_value > self.x[-1]:
            slope = (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])
            return float(max(0.0, self.y[-1] + slope * (x_value - self.x[-1])))

        idx = int(np.searchsorted(self.x, x_value, side="right") - 1)
        idx = min(max(idx, 0), self.n - 2)
        dx = x_value - self.x[idx]
        value = self._a[idx] + dx * (self._b[idx] + dx * (self._c[idx] + dx * self._d[idx]))
        return float(max(0.0, value))


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    diag = diag.copy()
    rhs = rhs.copy()
    n = len(diag)
    for i in range(1, n):
        factor = lower[i] / diag[i - 1]
        diag[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    out = np.zeros(n, dtype=float)
    out[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        out[i] = (rhs[i] - upper[i] * out[i + 1]) / diag[i]
    return out


def _make_interpolator(method: InterpolationMethod, x, y):
    if method == "linear":
        return LinearInterpolator(x, y)
    if method == "newton":
        return NewtonInterpolator(x, y)
    return CubicSplineInterpolator(x, y)


def _canonical_engine_type(engine_type: str) -> str:
    normalized = str(engine_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"one", "one_engine", "1", "single", "single_engine"}:
        return "one_engine"
    if normalized in {"two", "two_engine", "2", "double", "dual", "dual_engine"}:
        return "two_engine"
    return normalized


def _load_excel_grid(path: Path) -> tuple[dict[int, dict[int, dict[int, dict[str, np.ndarray]]]], list[int], list[int], list[int]]:
    xls = pd.ExcelFile(path)
    data: dict[int, dict[int, dict[int, dict[str, np.ndarray]]]] = {}
    altitudes: set[int] = set()
    weights: set[int] = set()
    drags: set[int] = set()

    for sheet_name in xls.sheet_names:
        frame = pd.read_excel(xls, sheet_name=sheet_name)
        rename: dict[str, str] = {}
        for column in frame.columns:
            lower = str(column).strip().lower()
            if "altitude" in lower:
                rename[column] = "altitude"
            elif "gross" in lower and "weight" in lower:
                rename[column] = "weight"
            elif "drag" in lower:
                rename[column] = "drag"
            elif "mach" in lower:
                rename[column] = "mach"
            elif "specific" in lower and "range" in lower:
                rename[column] = "sr"
        frame = frame.rename(columns=rename)
        required = {"altitude", "weight", "drag", "mach", "sr"}
        missing = required - set(frame.columns)
        if missing:
            continue

        frame["altitude"] = frame["altitude"].astype(str).str.strip().replace({"Sea Level": "0", "sea level": "0"})
        frame["altitude"] = pd.to_numeric(frame["altitude"], errors="coerce").fillna(0)
        for column in ("mach", "sr", "weight", "drag"):
            frame[column] = frame[column].astype(str).str.replace(",", ".", regex=False)
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["altitude", "weight", "drag", "mach", "sr"])

        for alt_value, alt_frame in frame.groupby(frame["altitude"].astype(int)):
            alt = int(alt_value)
            altitudes.add(alt)
            alt_bucket = data.setdefault(alt, {})
            for weight_value, weight_frame in alt_frame.groupby(alt_frame["weight"].astype(int)):
                weight = int(weight_value)
                weights.add(weight)
                weight_bucket = alt_bucket.setdefault(weight, {})
                for drag_value, drag_frame in weight_frame.groupby(weight_frame["drag"].astype(int)):
                    drag = int(drag_value)
                    drags.add(drag)
                    mach_frame = drag_frame.groupby("mach", as_index=False)["sr"].mean().sort_values("mach")
                    if len(mach_frame) >= 2:
                        weight_bucket[drag] = {
                            "mach": mach_frame["mach"].to_numpy(dtype=float),
                            "sr": mach_frame["sr"].to_numpy(dtype=float),
                        }

    return data, sorted(altitudes), sorted(weights), sorted(drags)


@dataclass
class SpecificRangeInterpolationModel:
    excel_path: Path

    def __post_init__(self) -> None:
        self.data, self.altitudes, self.weights, self.drags = _load_excel_grid(self.excel_path)
        if not self.data:
            raise ValueError(f"No interpolation data could be loaded from {self.excel_path}")
        self._mach_interpolators: dict[InterpolationMethod, dict[tuple[int, int, int], object]] = {}

    def _get_mach_interpolator(self, method: InterpolationMethod, altitude: int, weight: int, drag: int):
        method_cache = self._mach_interpolators.setdefault(method, {})
        key = (altitude, weight, drag)
        if key not in method_cache:
            group = self.data.get(altitude, {}).get(weight, {}).get(drag)
            if not group:
                return None
            method_cache[key] = _make_interpolator(method, group["mach"], group["sr"])
        return method_cache[key]

    def _interpolate_mach(self, method: InterpolationMethod, altitude: int, weight: int, drag: int, mach: float) -> float | None:
        interpolator = self._get_mach_interpolator(method, altitude, weight, drag)
        if interpolator is None:
            return None
        return interpolator.evaluate(mach)

    def compute(self, method: InterpolationMethod, altitude: float, weight: float, drag_index: float, mach: float) -> float | None:
        alt_values: list[float] = []
        sr_at_alt: list[float] = []

        for alt in self.altitudes:
            alt_bucket = self.data.get(alt, {})
            weight_values: list[float] = []
            sr_at_weight: list[float] = []
            for weight_key, weight_bucket in alt_bucket.items():
                drag_values: list[float] = []
                sr_at_drag: list[float] = []
                for drag_key in weight_bucket:
                    sr_value = self._interpolate_mach(method, alt, weight_key, drag_key, mach)
                    if sr_value is not None and np.isfinite(sr_value):
                        drag_values.append(float(drag_key))
                        sr_at_drag.append(float(sr_value))
                if not drag_values:
                    continue
                if len(drag_values) == 1:
                    sr_from_drag = sr_at_drag[0]
                else:
                    sr_from_drag = _make_interpolator(method, drag_values, sr_at_drag).evaluate(drag_index)
                weight_values.append(float(weight_key))
                sr_at_weight.append(sr_from_drag)
            if not weight_values:
                continue
            if len(weight_values) == 1:
                sr_from_weight = sr_at_weight[0]
            else:
                sr_from_weight = _make_interpolator(method, weight_values, sr_at_weight).evaluate(weight)
            alt_values.append(float(alt))
            sr_at_alt.append(sr_from_weight)

        if not alt_values:
            return None
        if len(alt_values) == 1:
            return float(sr_at_alt[0])
        return _make_interpolator(method, alt_values, sr_at_alt).evaluate(altitude)


class SpecificRangeInterpolationService:
    """Artifact-free table interpolation baseline.

    This service intentionally does not depend on ``external_apps`` at runtime.
    It ports the SR_project interpolation logic into the main codebase and reads
    the canonical workbook paths from ``DataConfig``.
    """

    def __init__(self, data_config: DataConfig | None = None) -> None:
        self.data_config = data_config or DataConfig()
        self._models: dict[str, SpecificRangeInterpolationModel] = {}

    def _get_model(self, engine_type: str) -> SpecificRangeInterpolationModel:
        engine_type = _canonical_engine_type(engine_type)
        if engine_type not in self._models:
            path = self.data_config.one_engine_path if engine_type == "one_engine" else self.data_config.two_engine_path
            if not path.exists():
                raise FileNotFoundError(f"Interpolation workbook not found: {path}")
            self._models[engine_type] = SpecificRangeInterpolationModel(path)
        return self._models[engine_type]

    def predict_one(
        self,
        *,
        engine_type: str,
        altitude: float,
        gross_weight: float,
        drag_index: float,
        mach: float,
        method: str = DEFAULT_INTERPOLATION_METHOD,
    ) -> float:
        method_key = self.normalize_method(method)
        model = self._get_model(engine_type)
        value = model.compute(method_key, altitude, gross_weight, drag_index, mach)
        if value is None or not np.isfinite(value):
            raise ValueError("Interpolation could not produce a value for this input.")
        return float(value)

    def predict_many_from_frame(self, frame: pd.DataFrame, method: str = DEFAULT_INTERPOLATION_METHOD) -> np.ndarray:
        method_key = self.normalize_method(method)
        values = []
        for row in frame[["engine_type", "altitude", "gross_weight", "drag_index", "mach"]].itertuples(index=False, name=None):
            model = self._get_model(str(row[0]))
            value = model.compute(method_key, float(row[1]), float(row[2]), float(row[3]), float(row[4]))
            if value is None or not np.isfinite(value):
                raise ValueError("Interpolation could not produce a value for this input.")
            values.append(float(value))
        return np.asarray(values, dtype=float)

    @staticmethod
    def normalize_method(method: str | None) -> InterpolationMethod:
        key = str(method or DEFAULT_INTERPOLATION_METHOD).strip().lower().replace("-", "_").replace(" ", "_")
        if key in {"cubic", "cubic_spline", "spline"}:
            return "spline"
        if key in {"piecewise_linear", "linear"}:
            return "linear"
        if key in {"newton", "divided_difference", "newton_divided_difference"}:
            return "newton"
        raise ValueError(f"Unknown interpolation method: {method}")

    @staticmethod
    def method_label(method: str = DEFAULT_INTERPOLATION_METHOD) -> str:
        return INTERPOLATION_METHODS[SpecificRangeInterpolationService.normalize_method(method)]
