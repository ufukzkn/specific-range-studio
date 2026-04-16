"""Specific Range Studio — Flask backend.

Wraps the existing ``src`` inference / reporting modules and serves a JSON
API alongside the static frontend assets.
"""

from __future__ import annotations

import logging

import json
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file, Response

# ---------------------------------------------------------------------------
# Path bootstrap — make sure ``src`` is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictors import (
    FTTransformerPredictor,
    XGBoostPredictor,
    add_reference_row_id,
    build_single_row_frame,
    build_test_scenarios,
    find_exact_match,
    find_nearest_reference_rows,
    load_reference_dataset,
)
from src.evaluation.nomogram_report import generate_nomogram_report
from src.evaluation.benchmark import estimate_model_size_mb
from src.utils.config import DataConfig

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_data_config = DataConfig()
_reference_df: pd.DataFrame | None = None
_xgb_predictor: XGBoostPredictor | None = None
_ft_predictor: FTTransformerPredictor | None = None


def _get_reference_df() -> pd.DataFrame:
    global _reference_df
    if _reference_df is None:
        _reference_df = add_reference_row_id(load_reference_dataset())
    return _reference_df


def _get_xgb() -> tuple[XGBoostPredictor | None, str]:
    global _xgb_predictor
    if _xgb_predictor is not None:
        return _xgb_predictor, ""
    try:
        _xgb_predictor = XGBoostPredictor.from_artifacts()
        return _xgb_predictor, ""
    except Exception as exc:
        logging.exception("XGBoost model loading failed")
        return None, str(exc)


def _get_ft(device: str = "cpu") -> tuple[FTTransformerPredictor | None, str]:
    global _ft_predictor
    if _ft_predictor is not None:
        return _ft_predictor, ""
    try:
        _ft_predictor = FTTransformerPredictor.from_artifacts(device=device)
        return _ft_predictor, ""
    except Exception as exc:
        logging.exception("FT-Transformer model loading failed")
        return None, str(exc)


def _report_paths(model_key: str) -> dict[str, Path]:
    base = (
        _data_config.xgboost_artifact_dir
        if model_key == "xgboost"
        else _data_config.ft_transformer_artifact_dir
    )
    report_dir = base / "reports"
    return {
        "row_level": report_dir / f"{model_key}_row_level_comparison.csv",
        "slice_summary": report_dir / f"{model_key}_slice_summary.csv",
        "overall_summary": report_dir / f"{model_key}_overall_summary.csv",
        "slice_plot": report_dir / f"{model_key}_slice_predictions.png",
        "summary_plot": report_dir / f"{model_key}_slice_summary.png",
    }


def _model_paths(model_key: str) -> dict[str, Path]:
    base = (
        _data_config.xgboost_artifact_dir
        if model_key == "xgboost"
        else _data_config.ft_transformer_artifact_dir
    )
    model_filename = "model.json" if model_key == "xgboost" else "model.pt"
    return {
        "base": base,
        "model": base / model_filename,
        "metrics": base / "metrics.json",
    }


def _load_model_summary(model_key: str) -> dict:
    report_paths = _report_paths(model_key)
    model_paths = _model_paths(model_key)

    metrics: dict[str, float] = {}
    config: dict = {}

    if report_paths["overall_summary"].exists():
        summary_df = pd.read_csv(report_paths["overall_summary"])
        if not summary_df.empty:
            metrics = summary_df.iloc[0].to_dict()

    if model_paths["metrics"].exists():
        with model_paths["metrics"].open("r", encoding="utf-8") as handle:
            metrics_blob = json.load(handle)
        config = metrics_blob.get("extra", {}).get("config", {})
        if not metrics and metrics_blob.get("metrics", {}).get("test"):
            metrics = metrics_blob["metrics"]["test"]

    model_size_mb = (
        estimate_model_size_mb(model_paths["model"])
        if model_paths["model"].exists()
        else None
    )

    return {
        "model_key": model_key,
        "display_name": "XGBoost" if model_key == "xgboost" else "FT-Transformer",
        "metrics": metrics,
        "config": config,
        "model_size_mb": model_size_mb,
        "artifact_path": model_paths["model"],
    }


def _estimate_runtime_profile(model_key: str, config: dict, model_size_mb: float, cpu_speed_factor: float) -> dict[str, float]:
    cpu_speed_factor = max(cpu_speed_factor, 0.35)

    if model_key == "xgboost":
        n_estimators = float(config.get("n_estimators", 300))
        max_depth = float(config.get("max_depth", 6))
        estimated_latency_ms = (0.75 + 0.0045 * n_estimators + 0.11 * max_depth) / cpu_speed_factor
        estimated_peak_ram_mb = max(
            48.0,
            (model_size_mb * 3.2) + (n_estimators * 0.06) + (max_depth * 4.5),
        )
    else:
        d_model = float(config.get("d_model", 64))
        n_layers = float(config.get("n_layers", 3))
        n_heads = float(config.get("n_heads", 4))
        d_ff = float(config.get("d_ff", 128))
        batch_size = float(config.get("batch_size", 128))
        head_load = n_layers * n_heads
        estimated_latency_ms = (
            1.10 + (0.020 * d_model) + (0.17 * head_load) + (0.004 * d_ff) + (0.0015 * batch_size)
        ) / cpu_speed_factor
        estimated_peak_ram_mb = max(
            96.0,
            (model_size_mb * 5.0) + (0.95 * d_model) + (0.75 * d_ff) + (8.0 * head_load),
        )

    return {
        "estimated_latency_ms": float(estimated_latency_ms),
        "estimated_peak_ram_mb": float(estimated_peak_ram_mb),
    }


def _normalize_weights(*weights: float) -> tuple[float, ...]:
    clamped = [max(float(weight), 0.0) for weight in weights]
    total = sum(clamped)
    if total <= 0:
        return tuple([1.0 / len(clamped)] * len(clamped))
    return tuple(weight / total for weight in clamped)


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types when serialising to JSON."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = _NumpyEncoder  # type: ignore[attr-defined]


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records."""
    return json.loads(df.to_json(orient="records", default_handler=str))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

@app.route("/api/reference/engine-types")
def api_engine_types():
    df = _get_reference_df()
    types = sorted(df["engine_type"].dropna().astype(str).unique().tolist())
    return jsonify(types)


@app.route("/api/reference/altitudes")
def api_altitudes():
    df = _get_reference_df()
    engine_type = request.args.get("engine_type")
    subset = df if not engine_type else df[df["engine_type"] == engine_type]
    altitudes = sorted(subset["altitude"].dropna().astype(float).unique().tolist())
    return jsonify(altitudes)


@app.route("/api/reference/gross-weights")
def api_gross_weights():
    df = _get_reference_df()
    engine_type = request.args.get("engine_type")
    altitude = request.args.get("altitude")
    subset = df
    if engine_type:
        subset = subset[subset["engine_type"] == engine_type]
    if altitude:
        subset = subset[subset["altitude"].astype(float) == float(altitude)]
    weights = sorted(subset["gross_weight"].dropna().astype(float).unique().tolist())
    return jsonify(weights)


@app.route("/api/scenarios")
def api_scenarios():
    scenarios = build_test_scenarios(_get_reference_df())
    return jsonify(scenarios)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    frame = build_single_row_frame(
        altitude=float(body["altitude"]),
        gross_weight=float(body["gross_weight"]),
        drag_index=float(body["drag_index"]),
        mach=float(body["mach"]),
        fuel_flow=float(body["fuel_flow"]),
        engine_type=str(body["engine_type"]),
    )

    result: dict = {}
    ref = _get_reference_df()

    xgb, xgb_err = _get_xgb()
    if xgb:
        result["xgboost"] = float(xgb.predict_from_frame(frame))
    elif xgb_err:
        result["xgboost_error"] = xgb_err

    ft, ft_err = _get_ft()
    if ft:
        result["ft_transformer"] = float(ft.predict_from_frame(frame))
    elif ft_err:
        result["ft_transformer_error"] = ft_err

    # Exact match — use wider tolerance for float precision through JSON
    exact = find_exact_match(ref, frame, atol=1e-4)

    if not exact.empty:
        result["exact_match"] = {
            "actual": float(exact.iloc[0]["specific_range"]),
            "rows": _df_to_records(exact.head(3)),
        }
    else:
        result["exact_match"] = None

        # Nearest rows — only when no exact match
        nearest = find_nearest_reference_rows(ref, frame, top_k=5)
        cols = [c for c in ["row_id", "engine_type", "altitude", "gross_weight",
                            "drag_index", "mach", "fuel_flow", "specific_range",
                            "distance"] if c in nearest.columns]
        result["nearest_rows"] = _df_to_records(nearest[cols]) if not nearest.empty else []

        # Distance-weighted average of nearest rows
        if not nearest.empty and "distance" in nearest.columns:
            distances = nearest["distance"].to_numpy(dtype=float)
            sr_values = nearest["specific_range"].to_numpy(dtype=float)
            # Inverse-distance weighting; if distance=0, it's an exact match (shouldn't reach here)
            inv_distances = np.where(distances > 0, 1.0 / distances, 1e12)
            weights = inv_distances / inv_distances.sum()
            result["weighted_avg_sr"] = float(np.dot(weights, sr_values))
        else:
            result["weighted_avg_sr"] = None

    return jsonify(result)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.route("/api/report/<model>/metrics")
def api_report_metrics(model: str):
    paths = _report_paths(model)
    if not paths["overall_summary"].exists():
        return jsonify({"error": "Report not found"}), 404
    df = pd.read_csv(paths["overall_summary"])
    return jsonify(_df_to_records(df)[0])


@app.route("/api/report/<model>/slice-summary")
def api_report_slice_summary(model: str):
    paths = _report_paths(model)
    if not paths["slice_summary"].exists():
        return jsonify({"error": "Report not found"}), 404
    df = pd.read_csv(paths["slice_summary"])
    return jsonify(_df_to_records(df))


@app.route("/api/report/<model>/rows")
def api_report_rows(model: str):
    paths = _report_paths(model)
    if not paths["row_level"].exists():
        return jsonify({"error": "Report not found"}), 404

    df = pd.read_csv(paths["row_level"])
    engine_type = request.args.get("engine_type")
    altitude = request.args.get("altitude")
    sort = request.args.get("sort", "error_desc")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    if engine_type and engine_type != "All":
        df = df[df["engine_type"] == engine_type]
    if altitude and altitude != "All":
        df = df[df["altitude"].astype(float) == float(altitude)]

    error_col = f"{model}_absolute_error"
    if error_col in df.columns:
        if sort == "error_desc":
            df = df.sort_values(error_col, ascending=False)
        elif sort == "error_asc":
            df = df.sort_values(error_col, ascending=True)
        else:
            df = df.sort_values("row_id")
    else:
        df = df.sort_values("row_id")

    total = len(df)
    start = (page - 1) * per_page
    page_df = df.iloc[start : start + per_page]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, -(-total // per_page)),
        "rows": _df_to_records(page_df),
    })


@app.route("/api/report/<model>/plots/<plot_name>")
def api_report_plot(model: str, plot_name: str):
    paths = _report_paths(model)
    key_map = {"slice_predictions": "slice_plot", "slice_summary": "summary_plot"}
    path = paths.get(key_map.get(plot_name, ""), None)
    if path is None or not path.exists():
        return jsonify({"error": "Plot not found"}), 404
    return send_file(str(path), mimetype="image/png")


# ---------------------------------------------------------------------------
# Comparison (both models merged)
# ---------------------------------------------------------------------------

@app.route("/api/compare/metrics")
def api_compare_metrics():
    result = {}
    for model in ("xgboost", "ft_transformer"):
        paths = _report_paths(model)
        if paths["overall_summary"].exists():
            df = pd.read_csv(paths["overall_summary"])
            result[model] = _df_to_records(df)[0]
    if not result:
        return jsonify({"error": "No reports found"}), 404
    return jsonify(result)


@app.route("/api/compare/rows")
def api_compare_rows():
    """Merge both models' row-level reports for side-by-side comparison."""
    xgb_paths = _report_paths("xgboost")
    ft_paths = _report_paths("ft_transformer")

    if not xgb_paths["row_level"].exists() or not ft_paths["row_level"].exists():
        return jsonify({"error": "Both model reports are required"}), 404

    xgb_df = pd.read_csv(xgb_paths["row_level"])
    ft_df = pd.read_csv(ft_paths["row_level"])

    # Merge on common columns
    common_cols = ["row_id", "engine_type", "altitude", "gross_weight",
                   "drag_index", "mach", "fuel_flow", "actual_specific_range"]
    ft_pred_cols = [c for c in ft_df.columns if c.startswith("ft_transformer")]
    merged = xgb_df.merge(
        ft_df[common_cols + ft_pred_cols],
        on=common_cols,
        how="inner",
        suffixes=("", "_ft"),
    )

    engine_type = request.args.get("engine_type")
    altitude = request.args.get("altitude")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    if engine_type and engine_type != "All":
        merged = merged[merged["engine_type"] == engine_type]
    if altitude and altitude != "All":
        merged = merged[merged["altitude"].astype(float) == float(altitude)]

    merged = merged.sort_values("xgboost_absolute_error", ascending=False)
    total = len(merged)
    start = (page - 1) * per_page
    page_df = merged.iloc[start : start + per_page]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, -(-total // per_page)),
        "rows": _df_to_records(page_df),
    })


@app.route("/api/compare/chart-data")
def api_compare_chart_data():
    """Return all rows with actual + both predictions for charting."""
    xgb_paths = _report_paths("xgboost")
    ft_paths = _report_paths("ft_transformer")

    if not xgb_paths["row_level"].exists() or not ft_paths["row_level"].exists():
        return jsonify({"error": "Both model reports are required"}), 404

    xgb_df = pd.read_csv(xgb_paths["row_level"])
    ft_df = pd.read_csv(ft_paths["row_level"])

    common_cols = ["row_id", "engine_type", "altitude", "gross_weight",
                   "drag_index", "mach", "fuel_flow", "actual_specific_range"]
    ft_pred_cols = [c for c in ft_df.columns if c.startswith("ft_transformer")]
    merged = xgb_df.merge(
        ft_df[common_cols + ft_pred_cols],
        on=common_cols,
        how="inner",
    )

    engine_type = request.args.get("engine_type")
    altitude = request.args.get("altitude")

    if engine_type and engine_type != "All":
        merged = merged[merged["engine_type"] == engine_type]
    if altitude and altitude != "All":
        merged = merged[merged["altitude"].astype(float) == float(altitude)]

    merged = merged.sort_values(["engine_type", "altitude", "mach"])
    return jsonify(_df_to_records(merged))


@app.route("/api/compare/cost-simulator")
def api_compare_cost_simulator():
    accuracy_weight = request.args.get("accuracy_weight", 55, type=float)
    latency_weight = request.args.get("latency_weight", 25, type=float)
    memory_weight = request.args.get("memory_weight", 20, type=float)
    cpu_speed_factor = request.args.get("cpu_speed_factor", 1.0, type=float)
    ram_budget_gb = request.args.get("ram_budget_gb", 8.0, type=float)

    accuracy_weight, latency_weight, memory_weight = _normalize_weights(
        accuracy_weight,
        latency_weight,
        memory_weight,
    )

    bundles = {
        model_key: _load_model_summary(model_key)
        for model_key in ("xgboost", "ft_transformer")
    }

    available = {
        key: value
        for key, value in bundles.items()
        if value["metrics"] and value["model_size_mb"] is not None
    }
    if len(available) < 2:
        return jsonify({"error": "Her iki model için metrik ve artefact bulunamadı"}), 404

    ram_budget_mb = max(ram_budget_gb * 1024.0, 256.0)

    best_rmse = min(float(bundle["metrics"]["rmse"]) for bundle in available.values())
    best_mae = min(float(bundle["metrics"]["mae"]) for bundle in available.values())
    best_mape = min(float(bundle["metrics"]["mape"]) for bundle in available.values())
    best_r2 = max(float(bundle["metrics"]["r2"]) for bundle in available.values())

    prepared: dict[str, dict] = {}
    for model_key, bundle in available.items():
        runtime = _estimate_runtime_profile(
            model_key,
            bundle["config"],
            float(bundle["model_size_mb"]),
            cpu_speed_factor,
        )
        metrics = bundle["metrics"]
        accuracy_component = float(np.mean([
            float(metrics["rmse"]) / max(best_rmse, 1e-12),
            float(metrics["mae"]) / max(best_mae, 1e-12),
            float(metrics["mape"]) / max(best_mape, 1e-12),
            max(best_r2, 1e-12) / max(float(metrics["r2"]), 1e-12),
        ]))

        over_budget_ratio = max(runtime["estimated_peak_ram_mb"] - ram_budget_mb, 0.0) / ram_budget_mb
        effective_memory = runtime["estimated_peak_ram_mb"] * (1.0 + over_budget_ratio)

        prepared[model_key] = {
            **bundle,
            **runtime,
            "accuracy_component": accuracy_component,
            "effective_memory_mb": effective_memory,
            "ram_budget_utilization": runtime["estimated_peak_ram_mb"] / ram_budget_mb,
            "over_budget_ratio": over_budget_ratio,
        }

    min_latency = min(item["estimated_latency_ms"] for item in prepared.values())
    min_memory = min(item["effective_memory_mb"] for item in prepared.values())

    raw_utilities: dict[str, float] = {}
    for model_key, item in prepared.items():
        latency_component = item["estimated_latency_ms"] / max(min_latency, 1e-12)
        memory_component = item["effective_memory_mb"] / max(min_memory, 1e-12)
        combined_cost = (
            accuracy_weight * item["accuracy_component"]
            + latency_weight * latency_component
            + memory_weight * memory_component
        )
        raw_utility = 1.0 / max(combined_cost, 1e-12)
        raw_utilities[model_key] = raw_utility
        item["latency_component"] = latency_component
        item["memory_component"] = memory_component
        item["combined_cost"] = combined_cost

    max_utility = max(raw_utilities.values())
    winner = max(raw_utilities, key=raw_utilities.get)

    response_models = {}
    for model_key, item in prepared.items():
        response_models[model_key] = {
            "display_name": item["display_name"],
            "metrics": item["metrics"],
            "model_size_mb": float(item["model_size_mb"]),
            "estimated_latency_ms": item["estimated_latency_ms"],
            "estimated_peak_ram_mb": item["estimated_peak_ram_mb"],
            "ram_budget_utilization": item["ram_budget_utilization"],
            "over_budget_ratio": item["over_budget_ratio"],
            "accuracy_component": item["accuracy_component"],
            "latency_component": item["latency_component"],
            "memory_component": item["memory_component"],
            "combined_cost": item["combined_cost"],
            "fit_score": 100.0 * (raw_utilities[model_key] / max_utility),
            "note": (
                "Tahmini maliyet modeli; gerçek benchmark değil."
                + (" RAM bütçesini aşıyor." if item["over_budget_ratio"] > 0 else "")
            ),
        }

    return jsonify({
        "weights": {
            "accuracy": accuracy_weight,
            "latency": latency_weight,
            "memory": memory_weight,
        },
        "hardware": {
            "cpu_speed_factor": cpu_speed_factor,
            "ram_budget_gb": ram_budget_gb,
        },
        "winner": winner,
        "models": response_models,
        "note": "Bu paneldeki hız ve RAM değerleri artefact boyutu ve model konfigürasyonundan türetilen tahmini değerlerdir.",
    })


# ---------------------------------------------------------------------------
# Nomogram
# ---------------------------------------------------------------------------

@app.route("/api/nomogram", methods=["POST"])
def api_nomogram():
    body = request.get_json(force=True)
    model_key = body.get("model", "xgboost")
    engine_type = body["engine_type"]
    altitude = float(body["altitude"])
    gross_weight = float(body["gross_weight"])

    if model_key == "xgboost":
        predictor, err = _get_xgb()
    else:
        predictor, err = _get_ft()

    if predictor is None:
        return jsonify({"error": f"Model {model_key} not available: {err}"}), 500

    base = (
        _data_config.xgboost_artifact_dir
        if model_key == "xgboost"
        else _data_config.ft_transformer_artifact_dir
    )

    result = generate_nomogram_report(
        _get_reference_df(),
        model_name=model_key,
        batch_predict_fn=predictor.predict_many_from_frame,
        output_dir=base / "nomogram_reports",
        engine_type=engine_type,
        altitude=altitude,
        gross_weight=gross_weight,
    )

    return jsonify({
        "plot_url": f"/api/nomogram/plot?path={result.nomogram_png}",
        "message": f"Nomogram generated: {result.nomogram_png}",
    })


@app.route("/api/nomogram/plot")
def api_nomogram_plot():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.exists():
        return jsonify({"error": "Plot not found"}), 404
    return send_file(str(p), mimetype="image/png")


# ---------------------------------------------------------------------------
# Setup commands
# ---------------------------------------------------------------------------

@app.route("/api/setup/commands")
def api_setup_commands():
    python = sys.executable
    ds = str(_data_config.processed_path)
    return jsonify([
        {
            "id": "data_pipeline",
            "label": "Veri Pipeline",
            "command": f"{python} scripts/run_data_pipeline.py",
            "eta": "~10-30 sn",
            "artifacts": ["data/processed/combined_specific_range.csv"],
        },
        {
            "id": "xgboost_train",
            "label": "XGBoost Eğit + Rapor",
            "command": f"{python} scripts/train_xgboost.py --dataset {ds} --device cuda --run-table-report",
            "eta": "~30 sn - 3 dk",
            "artifacts": ["artifacts/xgboost/model.json", "artifacts/xgboost/reports/*"],
        },
        {
            "id": "ft_train",
            "label": "FT-Transformer Eğit + Rapor",
            "command": f"{python} scripts/train_ft_transformer.py --dataset {ds} --device cuda --run-table-report",
            "eta": "~2-15 dk",
            "artifacts": ["artifacts/ft_transformer/model.pt", "artifacts/ft_transformer/reports/*"],
        },
        {
            "id": "table_report",
            "label": "Toplu Rapor Üret",
            "command": f"{python} scripts/run_table_report.py --dataset {ds} --model both --ft-device cuda",
            "eta": "~20 sn - 3 dk",
            "artifacts": ["artifacts/*/reports/*"],
        },
    ])


@app.route("/api/setup/run", methods=["POST"])
def api_setup_run():
    """Run a setup command and stream output via SSE."""
    body = request.get_json(force=True)
    command = body.get("command", "")
    if not command:
        return jsonify({"error": "No command provided"}), 400

    def generate():
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(process.stdout.readline, ""):
                yield f"data: {json.dumps({'type': 'output', 'text': line.rstrip()})}\n\n"
            process.wait()
            yield f"data: {json.dumps({'type': 'done', 'exit_code': process.returncode})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """Quick health check — are models and reports available?"""
    xgb_ok = (_data_config.xgboost_artifact_dir / "model.json").exists()
    ft_ok = (_data_config.ft_transformer_artifact_dir / "model.pt").exists()
    xgb_report = _report_paths("xgboost")["row_level"].exists()
    ft_report = _report_paths("ft_transformer")["row_level"].exists()
    data_ok = _data_config.processed_path.exists()
    return jsonify({
        "data_ready": data_ok,
        "xgboost_model": xgb_ok,
        "ft_transformer_model": ft_ok,
        "xgboost_report": xgb_report,
        "ft_transformer_report": ft_report,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = 5000
    url = f"http://localhost:{port}"
    print(f"\n  Specific Range Studio -> {url}\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
