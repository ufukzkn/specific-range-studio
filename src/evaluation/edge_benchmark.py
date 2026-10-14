"""Inference-only, isolated-process benchmark for a verified research bundle."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid

from src.evaluation.research_bundle import FEATURES, git_info, sha256, verify_bundle

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["interpolation", "xgboost", "ft_transformer"]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_text(path):
    try:
        return Path(path).read_text().strip().replace("\x00", "")
    except (OSError, UnicodeError):
        return None


def probe_system():
    import psutil
    result = {"hostname": socket.gethostname(), "platform": platform.platform(),
              "machine": platform.machine(), "python": sys.version,
              "cpu_count": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
              "jetson_model": read_text("/proc/device-tree/model"),
              "l4t_release": read_text("/etc/nv_tegra_release"),
              "container": Path("/.dockerenv").exists(), "power_settings": {}}
    result["thermal_zones"] = {read_text(zone / "type") or zone.name: read_text(zone / "temp")
                               for zone in Path("/sys/class/thermal").glob("thermal_zone*")}
    for cmd in [["nvpmodel", "-q"], ["jetson_clocks", "--show"], ["dpkg-query", "-W", "nvidia-jetpack"]]:
        if shutil.which(cmd[0]):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
                result["power_settings"][" ".join(cmd)] = {"exit_code": proc.returncode,
                    "stdout": proc.stdout, "stderr": proc.stderr}
            except subprocess.SubprocessError as exc:
                result["power_settings"][" ".join(cmd)] = {"error": str(exc)}
    return result


def latency_stats(values):
    import numpy as np
    return {"min": float(np.min(values)), "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(np.max(values))}


def run_worker(options):
    import numpy as np
    import pandas as pd
    import psutil
    from src.evaluation.metrics import regression_metrics
    from src.inference.predictors import FTTransformerPredictor, XGBoostPredictor
    from src.interpolation.specific_range import SpecificRangeInterpolationService

    bundle = Path(options["bundle"])
    manifest = verify_bundle(bundle)
    for name in ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]:
        expected = manifest["packages"][name]
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name} version mismatch ({actual} != {expected}); install bundle/requirements-portable.txt")
    key = options["model"]
    sample = pd.read_csv(options["sample"])
    process = psutil.Process()
    rss_before_load = process.memory_info().rss
    started = time.perf_counter()
    torch = None
    synchronize = lambda: None
    if key == "interpolation":
        predictor = SpecificRangeInterpolationService.from_frame(pd.read_csv(bundle / "train.csv"))
        actual_device = "cpu"
        artifact_files = ["train.csv"]
    elif key == "xgboost":
        predictor = XGBoostPredictor.from_artifacts(bundle / key)
        # Portable CPU baseline, including models previously trained with CUDA.
        predictor.model.model.set_params(device="cpu", n_jobs=options["threads"])
        actual_device = "cpu"
        artifact_files = ["xgboost/model.json", "xgboost/config.json", "xgboost/preprocessor.joblib"]
    else:
        import torch
        torch.set_num_threads(options["threads"])
        torch.set_num_interop_threads(1)
        requested = options["ft_device"]
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable: refusing CPU fallback")
        predictor = FTTransformerPredictor.from_artifacts(bundle / key, device=requested)
        actual_device = str(predictor.trainer.device)
        if requested == "cuda" and not actual_device.startswith("cuda"):
            raise RuntimeError("FT device mismatch")
        if actual_device.startswith("cuda"):
            synchronize = torch.cuda.synchronize
        artifact_files = ["ft_transformer/model.pt", "ft_transformer/preprocessor.joblib"]
    if key != "interpolation" and (predictor.preprocessor.numerical_features != FEATURES
            or predictor.preprocessor.categorical_features != ["engine_type"]):
        raise ValueError("Common-input benchmark requires four inputs plus engine_type; retrain a research bundle")
    load_seconds = time.perf_counter() - started
    # Labels and contextual fuel flow are unavailable to the inference input path.
    features = sample[["engine_type", *FEATURES]].copy()
    features["specific_range"] = 0.0  # preprocessing API carries an unused y field
    rows = [features.iloc[[i]].copy() for i in range(len(features))]
    for i in range(options["warmup"]):
        predictor.predict_many_from_frame(rows[i % len(rows)])
    synchronize()
    rss_after_warmup = process.memory_info().rss
    rss_samples = [rss_after_warmup]
    stop = threading.Event()
    def sample_memory():
        while not stop.wait(0.01):
            rss_samples.append(process.memory_info().rss)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    if torch is not None and actual_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    cpu_start = process.cpu_times()
    measured_start = now()
    latencies = []
    start = time.perf_counter()
    try:
        for i in range(options["repetitions"]):
            synchronize()
            tick = time.perf_counter_ns()
            predictor.predict_many_from_frame(rows[i % len(rows)])
            synchronize()
            latencies.append((time.perf_counter_ns() - tick) / 1e6)
        elapsed = time.perf_counter() - start
        cpu_end = process.cpu_times()
    finally:
        stop.set()
        sampler.join()
    measured_end = now()
    rss_samples.append(process.memory_info().rss)
    cpu_seconds = cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
    gpu = None
    if torch is not None and actual_device.startswith("cuda"):
        gpu = {"name": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
               "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
               "reserved_peak_bytes": torch.cuda.max_memory_reserved()}
    # Accuracy is separate from timing and includes cubic interpolation on the same held-out rows.
    prediction = predictor.predict_many_from_frame(features)
    if prediction.shape != (len(sample),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions")
    metrics = regression_metrics(sample.specific_range.to_numpy(dtype=float), prediction)
    prefix = Path(options["prefix"])
    with prefix.with_suffix(".latencies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "row_id", "latency_ms"])
        for i, latency in enumerate(latencies):
            writer.writerow([i, int(sample.row_id.iloc[i % len(sample)]), latency])
    predictions = sample[["row_id", "engine_type", "specific_range"]].copy()
    predictions["prediction"] = prediction
    predictions.to_csv(prefix.with_suffix(".predictions.csv"), index=False)
    result = {"model_key": key, "available": True, "trial": options["trial"],
        "requested_device": options["ft_device"] if key == "ft_transformer" else "cpu",
        "device": actual_device, "pid": os.getpid(), "load_seconds": load_seconds,
        "measured_start": measured_start, "measured_end": measured_end,
        "sample_count": len(sample), "sample_sha256": sha256(Path(options["sample"])),
        "warmup_count": options["warmup"], "repetitions": options["repetitions"],
        "elapsed_sec": elapsed, "throughput_per_sec": options["repetitions"] / elapsed,
        "latency_ms": latency_stats(latencies), "accuracy": metrics,
        "cpu_percent": {"process_avg": 100 * cpu_seconds / elapsed, "normalization": "one logical CPU = 100%"},
        "memory_bytes": {"rss_before_load": rss_before_load, "rss_after_warmup": rss_after_warmup,
                         "rss_peak_sampled": max(rss_samples), "sample_interval_ms": 10},
        "gpu_memory": gpu,
        "artifact_bytes": sum((bundle / f).stat().st_size for f in artifact_files),
        "artifact_sha256": {f: manifest["files"][f] for f in artifact_files},
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "joblib", "psutil", "xgboost"]}}
    if torch is not None:
        result["packages"]["torch"] = torch.__version__
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/research_bundle_portable")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks")
    parser.add_argument("--ft-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-jetson", action="store_true")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config:
        run_worker(json.loads(args.worker_config.read_text(encoding="utf-8")))
        return 0
    if min(args.sample_size, args.repetitions, args.trials, args.threads) < 1 or args.warmup < 1:
        parser.error("sample-size, warmup, repetitions, trials and threads must be positive")
    manifest = verify_bundle(args.bundle)
    system = probe_system()
    is_jetson = bool(system["l4t_release"] and system["jetson_model"] and platform.machine() == "aarch64")
    if args.require_jetson and not is_jetson:
        parser.error("Jetson identity could not be verified")
    import pandas as pd
    import numpy as np
    test = pd.read_csv(args.bundle / "test.csv")
    if args.sample_size < test.engine_type.nunique():
        parser.error("sample-size must cover all engine types")
    count = min(args.sample_size, len(test))
    # Include each category, then select the rest once; every worker reads the same file.
    base = test.groupby("engine_type", group_keys=False).sample(n=1, random_state=args.seed)
    sample = pd.concat([base, test.drop(base.index).sample(n=count-len(base), random_state=args.seed)])
    sample = sample.sort_values("row_id").reset_index(drop=True)
    run_dir = args.output / ("edge_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    sample_path = run_dir / "test_sample.csv"
    sample.to_csv(sample_path, index=False)
    payload = {"schema_version": 2, "run_id": run_dir.name, "created_at": now(),
        "system": system, "verified_jetson": is_jetson,
        "bundle_manifest_sha256": sha256(args.bundle / "manifest.json"),
        "experiment_label": manifest["experiment_label"], "data_audit": manifest["data_audit"],
        "source_code": git_info(ROOT), "config": vars(args).copy(), "trials": [],
        "notes": ["Common held-out test sample; cubic reference uses training nodes only.",
            "Known-curve interior Mach holdout; not new-flight or new-curve generalization.",
            "Single-row synchronous inference; includes preprocessing and CPU/GPU transfer, excludes HTTP/UI/load.",
            "One fresh process per model and trial; CPU times exclude warmup and accuracy evaluation.",
            "RSS is sampled, not an exact allocation peak; CUDA allocator peaks are separate from shared system RAM.",
            "Runtime includes a 10-ms RSS sampler; raw tegrastats may require host privileges."]}
    payload["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in payload["config"].items()}
    telemetry = None
    telemetry_handle = None
    telemetry_reader = None
    payload["tegrastats"] = {"available": bool(shutil.which("tegrastats")), "file": None}
    if shutil.which("tegrastats"):
        telemetry_handle = (run_dir / "tegrastats.log").open("w", encoding="utf-8")
        telemetry = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
        def read_telemetry():
            for line in telemetry.stdout:
                telemetry_handle.write(now() + " " + line)
                telemetry_handle.flush()
        telemetry_reader = threading.Thread(target=read_telemetry, daemon=True)
        telemetry_reader.start()
        payload["tegrastats"]["file"] = "tegrastats.log"
    env = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[name] = str(args.threads)
    try:
        for trial in range(args.trials):
            # Seeded order reduces a fixed thermal/order advantage across methods.
            for key in np.random.default_rng(args.seed + trial).permutation(MODELS):
                prefix = run_dir / f"{key}_trial{trial+1}"
                options = {"bundle": str(args.bundle.resolve()), "model": str(key), "sample": str(sample_path.resolve()),
                    "prefix": str(prefix.resolve()), "ft_device": args.ft_device, "threads": args.threads,
                    "warmup": args.warmup, "repetitions": args.repetitions, "trial": trial+1}
                worker_file = prefix.with_suffix(".worker.json")
                worker_file.write_text(json.dumps(options), encoding="utf-8")
                with prefix.with_suffix(".log").open("w", encoding="utf-8") as log:
                    proc = subprocess.run([sys.executable, str(ROOT / "scripts/run_edge_benchmark.py"),
                        "--worker-config", str(worker_file.resolve())], env=env, stdout=log,
                        stderr=subprocess.STDOUT, check=False)
                if proc.returncode or not prefix.with_suffix(".json").exists():
                    item = {"model_key": str(key), "trial": trial+1, "available": False,
                            "error": f"Worker exit {proc.returncode}; see {prefix.name}.log"}
                else:
                    item = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
                payload["trials"].append(item)
                print(f"{key} trial {trial+1}: " + (f"p95={item['latency_ms']['p95']:.3f} ms" if item["available"] else item["error"]), flush=True)
    finally:
        if telemetry:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
            payload["tegrastats"]["exit_code"] = telemetry.returncode
            telemetry_reader.join(timeout=5)
            telemetry_handle.close()
    payload["system_after"] = probe_system()
    payload["complete"] = len(payload["trials"]) == len(MODELS)*args.trials and all(i["available"] for i in payload["trials"])
    payload["finished_at"] = now()
    for target in [run_dir / "summary.json", args.output / "edge_benchmark_latest.json"]:
        target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["model", "trial", "device", "p50_ms", "p95_ms", "p99_ms", "rss_peak_sampled_bytes", "cpu_avg_percent", "artifact_bytes", "rmse", "mae", "mape", "r2"]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for item in payload["trials"]:
            if item["available"]:
                writer.writerow([item["model_key"], item["trial"], item["device"],
                    *[item["latency_ms"][k] for k in ["p50", "p95", "p99"]],
                    item["memory_bytes"]["rss_peak_sampled"], item["cpu_percent"]["process_avg"], item["artifact_bytes"],
                    *[item["accuracy"][k] for k in ["rmse", "mae", "mape", "r2"]]])
    print(f"Results: {run_dir} (complete={payload['complete']}, Jetson={is_jetson})", flush=True)
    return 0 if payload["complete"] else 1
