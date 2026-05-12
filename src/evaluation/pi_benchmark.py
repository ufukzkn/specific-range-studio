from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised on deployment machines.
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from src.evaluation.benchmark import estimate_model_size_mb
from src.evaluation.metrics import regression_metrics
from src.inference.predictors import FTTransformerPredictor, XGBoostPredictor, load_reference_dataset
from src.interpolation import DEFAULT_INTERPOLATION_METHOD, SpecificRangeInterpolationService
from src.utils.config import DataConfig


DEFAULT_MODELS = ("interpolation", "xgboost", "ft_transformer")
OPTIONAL_MODELS = ("interpolation", "xgboost", "ft_transformer")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_vcgencmd(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=3, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    return text or None


def _raspberry_pi_probe() -> dict:
    temp_raw = _read_vcgencmd(["measure_temp"])
    throttled_raw = _read_vcgencmd(["get_throttled"])
    temp_c = None
    if temp_raw and "temp=" in temp_raw:
        try:
            temp_c = float(temp_raw.split("temp=", 1)[1].split("'")[0])
        except (IndexError, ValueError):
            temp_c = None
    return {
        "temperature_c": temp_c,
        "temperature_raw": temp_raw,
        "throttled_raw": throttled_raw,
        "vcgencmd_available": temp_raw is not None or throttled_raw is not None,
    }


def _system_info() -> dict:
    virtual_memory = psutil.virtual_memory() if psutil else None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(logical=True) if psutil else None,
        "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else None,
        "total_ram_mb": float(virtual_memory.total / (1024 * 1024)) if virtual_memory else None,
        "psutil_available": psutil is not None,
    }


def _sample_reference_rows(reference_df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if reference_df.empty:
        raise ValueError("Reference dataset is empty; benchmark cannot select input rows.")
    sample_size = max(1, min(int(sample_size), len(reference_df)))
    return reference_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def _latency_stats(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _safe_model_size(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return estimate_model_size_mb(path)
    except Exception:
        return None


def _workbook_size_mb(config: DataConfig) -> float | None:
    paths = [config.one_engine_path, config.two_engine_path]
    if not all(path.exists() for path in paths):
        return None
    return float(sum(path.stat().st_size for path in paths) / (1024 * 1024))


def _predict_many(model_key: str, predictor, frame: pd.DataFrame, device: str) -> np.ndarray:
    if model_key == "interpolation":
        return predictor.predict_many_from_frame(frame, DEFAULT_INTERPOLATION_METHOD)
    return predictor.predict_many_from_frame(frame)


def _load_predictor(model_key: str, config: DataConfig, device: str):
    if model_key == "interpolation":
        return SpecificRangeInterpolationService(config), _workbook_size_mb(config)
    if model_key == "xgboost":
        return XGBoostPredictor.from_artifacts(config.xgboost_artifact_dir), _safe_model_size(config.xgboost_artifact_dir / "model.json")
    if model_key == "ft_transformer":
        return FTTransformerPredictor.from_artifacts(config.ft_transformer_artifact_dir, device=device), _safe_model_size(config.ft_transformer_artifact_dir / "model.pt")
    raise ValueError(f"Unknown benchmark model: {model_key}")


def benchmark_model(
    *,
    model_key: str,
    sample_df: pd.DataFrame,
    warmup: int,
    repetitions: int,
    device: str,
    data_config: DataConfig,
) -> dict:
    started_at = _utc_now()
    before_pi = _raspberry_pi_probe()
    try:
        predictor, model_size_mb = _load_predictor(model_key, data_config, device)
    except Exception as exc:
        return {
            "model_key": model_key,
            "available": False,
            "skipped_reason": str(exc),
            "started_at": started_at,
            "finished_at": _utc_now(),
        }

    process = psutil.Process() if psutil else None
    rss_before = float(process.memory_info().rss / (1024 * 1024)) if process else None
    cpu_times_before = process.cpu_times() if process else None
    if process:
        process.cpu_percent(interval=None)

    rows = [sample_df.iloc[[idx]].copy() for idx in range(len(sample_df))]
    for idx in range(max(0, warmup)):
        _predict_many(model_key, predictor, rows[idx % len(rows)], device)

    latencies: list[float] = []
    cpu_samples: list[float] = []
    peak_rss = rss_before or 0.0
    start = time.perf_counter()
    for idx in range(max(1, repetitions)):
        row = rows[idx % len(rows)]
        t0 = time.perf_counter()
        _predict_many(model_key, predictor, row, device)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if process and (idx % 5 == 0 or idx == repetitions - 1):
            peak_rss = max(peak_rss, float(process.memory_info().rss / (1024 * 1024)))
            cpu_samples.append(float(process.cpu_percent(interval=None)))
    elapsed = max(time.perf_counter() - start, 1e-9)

    rss_after = float(process.memory_info().rss / (1024 * 1024)) if process else None
    cpu_avg = None
    if process and cpu_times_before:
        cpu_times_after = process.cpu_times()
        cpu_delta = (
            (cpu_times_after.user - cpu_times_before.user)
            + (cpu_times_after.system - cpu_times_before.system)
        )
        cpu_avg = float((cpu_delta / elapsed) * 100.0)

    accuracy = None
    if model_key == "interpolation":
        accuracy = {
            "note": (
                "Not scored as model accuracy. Interpolation is treated as the deterministic "
                "table/reference family; benchmark records runtime, memory, CPU and artifact size."
            )
        }
    elif "specific_range" in sample_df.columns:
        try:
            y_true = sample_df["specific_range"].to_numpy(dtype=float)
            y_pred = _predict_many(model_key, predictor, sample_df, device)
            accuracy = regression_metrics(y_true, y_pred)
        except Exception as exc:
            accuracy = {"error": str(exc)}

    after_pi = _raspberry_pi_probe()
    latency = _latency_stats(latencies)
    return {
        "model_key": model_key,
        "available": True,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "device": device,
        "sample_count": int(len(sample_df)),
        "warmup_count": int(max(0, warmup)),
        "repetitions": int(max(1, repetitions)),
        "elapsed_sec": float(elapsed),
        "throughput_per_sec": float(max(1, repetitions) / elapsed),
        "latency_ms": latency,
        "memory_mb": {
            "rss_before": rss_before,
            "rss_after": rss_after,
            "rss_peak": peak_rss if process else None,
        },
        "cpu_percent": {
            "process_avg": cpu_avg,
            "process_peak_sample": float(max(cpu_samples)) if cpu_samples else None,
            "sample_count": len(cpu_samples),
        },
        "model_size_mb": model_size_mb,
        "accuracy": accuracy,
        "raspberry_pi": {
            "before": before_pi,
            "after": after_pi,
        },
    }


def run_pi_benchmark(
    *,
    models: list[str],
    sample_size: int,
    warmup: int,
    repetitions: int,
    device: str,
    seed: int,
    output_dir: Path,
    dataset_path: Path | None = None,
) -> dict:
    config = DataConfig()
    if dataset_path and dataset_path.exists():
        reference_df = pd.read_csv(dataset_path)
    else:
        reference_df = load_reference_dataset()
    sample_df = _sample_reference_rows(reference_df, sample_size, seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _utc_now(),
        "system": _system_info(),
        "config": {
            "models": models,
            "sample_size": int(sample_size),
            "warmup": int(warmup),
            "repetitions": int(repetitions),
            "device": device,
            "seed": int(seed),
            "dataset_path": str(dataset_path) if dataset_path else str(config.processed_path),
        },
        "models": {},
        "notes": [
            "Inference-only benchmark. Training is intentionally not performed on Raspberry Pi.",
            "Latency includes the same artifact-backed prediction path used by the app.",
            "Interpolation is benchmarked for runtime cost only; accuracy comparison is for XGBoost and FT-Transformer.",
        ],
    }

    for model_key in models:
        payload["models"][model_key] = benchmark_model(
            model_key=model_key,
            sample_df=sample_df,
            warmup=warmup,
            repetitions=repetitions,
            device=device,
            data_config=config,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "pi_benchmark_latest.json"
    run_path = output_dir / f"pi_benchmark_{run_id}.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    latest_path.write_text(text, encoding="utf-8")
    run_path.write_text(text, encoding="utf-8")
    append_history(output_dir / "pi_benchmark_history.csv", payload)
    return payload


def append_history(path: Path, payload: dict) -> None:
    rows = []
    for model_key, item in payload.get("models", {}).items():
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "created_at": payload.get("created_at"),
                "hostname": payload.get("system", {}).get("hostname"),
                "model_key": model_key,
                "available": item.get("available"),
                "sample_count": item.get("sample_count"),
                "repetitions": item.get("repetitions"),
                "latency_p50_ms": (item.get("latency_ms") or {}).get("p50"),
                "latency_p95_ms": (item.get("latency_ms") or {}).get("p95"),
                "rss_peak_mb": (item.get("memory_mb") or {}).get("rss_peak"),
                "cpu_avg_percent": (item.get("cpu_percent") or {}).get("process_avg"),
                "model_size_mb": item.get("model_size_mb"),
                "rmse": (item.get("accuracy") or {}).get("rmse"),
                "mae": (item.get("accuracy") or {}).get("mae"),
                "mape": (item.get("accuracy") or {}).get("mape"),
                "r2": (item.get("accuracy") or {}).get("r2"),
                "skipped_reason": item.get("skipped_reason"),
            }
        )
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(OPTIONAL_MODELS)
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in models if item not in OPTIONAL_MODELS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown model(s): {', '.join(invalid)}")
    return models or list(DEFAULT_MODELS)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real inference benchmark metrics for Raspberry Pi demos.")
    parser.add_argument("--models", type=parse_models, default=list(DEFAULT_MODELS), help="all or comma list: interpolation,xgboost,ft_transformer")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DataConfig().artifacts_dir / "benchmarks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_pi_benchmark(
        models=args.models,
        sample_size=args.sample_size,
        warmup=args.warmup,
        repetitions=args.repetitions,
        device=args.device,
        seed=args.seed,
        output_dir=args.output,
        dataset_path=args.dataset,
    )
    print(f"Saved latest benchmark: {args.output / 'pi_benchmark_latest.json'}")
    for model_key, item in payload.get("models", {}).items():
        if not item.get("available"):
            print(f"- {model_key}: skipped ({item.get('skipped_reason')})")
            continue
        latency = item.get("latency_ms", {})
        memory = item.get("memory_mb", {})
        cpu = item.get("cpu_percent", {})
        rss_peak = memory.get("rss_peak")
        cpu_avg = cpu.get("process_avg")
        rss_text = f"{rss_peak:.1f} MB" if rss_peak is not None else "n/a"
        cpu_text = f"{cpu_avg:.1f}%" if cpu_avg is not None else "n/a"
        print(
            f"- {model_key}: p95={latency.get('p95', 0):.3f} ms, "
            f"rss_peak={rss_text}, "
            f"cpu_avg={cpu_text}"
        )
    return 0
