from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class PredictProtocol(Protocol):
    """Minimal prediction interface used by benchmark helpers."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass(slots=True)
class BenchmarkResult:
    """Future-ready benchmark payload used in PSO objective scalarization."""

    latency_ms: float
    model_size_mb: float
    notes: list[str]


def export_to_onnx(*args, **kwargs) -> Path:
    """Placeholder for future ONNX export integration."""

    raise NotImplementedError("ONNX export is not wired yet. Add model-specific export logic here.")


def build_tensorrt_engine(*args, **kwargs) -> Path:
    """Placeholder for future TensorRT engine build integration."""

    raise NotImplementedError("TensorRT engine build is not wired yet. Add deployment tooling here.")


def measure_inference_latency(model: PredictProtocol, X: np.ndarray, repetitions: int = 100) -> float:
    """Measure average CPU inference latency in milliseconds."""

    if len(X) == 0:
        raise ValueError("Latency cannot be measured on an empty input array.")
    sample = X[:1]
    start = time.perf_counter()
    for _ in range(repetitions):
        model.predict(sample)
    elapsed = time.perf_counter() - start
    return float((elapsed / repetitions) * 1000.0)


def estimate_model_size_mb(path: Path | None = None, *, num_parameters: int | None = None) -> float:
    """Estimate model footprint from a file or parameter count."""

    if path is not None and path.exists():
        return float(path.stat().st_size / (1024 * 1024))
    if num_parameters is not None:
        return float((num_parameters * 4) / (1024 * 1024))
    raise ValueError("Provide either a model file path or a parameter count.")


def benchmark_stub(notes: list[str] | None = None) -> BenchmarkResult:
    """Return neutral benchmark values until deployment benchmarking is available."""

    return BenchmarkResult(latency_ms=1.0, model_size_mb=1.0, notes=notes or ["Benchmark stub used."])
