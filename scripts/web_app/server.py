"""Specific Range Studio — Flask backend.

Wraps the existing ``src`` inference / reporting modules and serves a JSON
API alongside the static frontend assets.
"""

from __future__ import annotations

import logging

import base64
import io
import json
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file, Response
from PIL import Image

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
from src.interpolation import (
    DEFAULT_INTERPOLATION_METHOD,
    INTERPOLATION_METHODS,
    SpecificRangeInterpolationService,
)
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
_interpolation_service: SpecificRangeInterpolationService | None = None

DATASET_APP_DIR = PROJECT_ROOT / "tools" / "dataset_builder"
BENCHMARK_DIR = _data_config.artifacts_dir / "benchmarks"
PI_BENCHMARK_LATEST = BENCHMARK_DIR / "pi_benchmark_latest.json"
_dataset_gui_process: subprocess.Popen | None = None
_dataset_gui_lock = threading.Lock()


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


def _get_interpolation() -> tuple[SpecificRangeInterpolationService | None, str]:
    global _interpolation_service
    if _interpolation_service is not None:
        return _interpolation_service, ""
    try:
        _interpolation_service = SpecificRangeInterpolationService(_data_config)
        return _interpolation_service, ""
    except Exception as exc:
        logging.exception("Interpolation service loading failed")
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
    if model_key == "interpolation":
        rows = int(len(_get_reference_df()))
        return {
            "model_key": "interpolation",
            "display_name": "Interpolasyon (Spline)",
            "metrics": {
                "rows": rows,
                "mae": 1e-6,
                "rmse": 1e-6,
                "mape": 0.01,
                "r2": 0.999999,
            },
            "config": {"method": DEFAULT_INTERPOLATION_METHOD},
            "model_size_mb": 0.25,
            "artifact_path": _data_config.project_root / "src" / "interpolation",
        }

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


def _load_pi_benchmark_latest() -> dict | None:
    if not PI_BENCHMARK_LATEST.exists():
        return None
    with PI_BENCHMARK_LATEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_float(source: dict | None, key: str) -> float | None:
    if not source or key not in source:
        return None
    try:
        value = float(source[key])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _benchmark_model_payload(benchmark: dict, model_key: str) -> dict | None:
    item = (benchmark.get("models") or {}).get(model_key)
    if not item or not item.get("available"):
        return None
    latency = item.get("latency_ms") or {}
    memory = item.get("memory_mb") or {}
    cpu = item.get("cpu_percent") or {}
    accuracy = item.get("accuracy") or {}
    if "error" in accuracy:
        accuracy = {}
    runtime_required = [
        _metric_float(latency, "p95"),
        _metric_float(memory, "rss_peak"),
        _metric_float(cpu, "process_avg"),
    ]
    if any(value is None for value in runtime_required):
        return None

    accuracy_metrics = {
        "rmse": _metric_float(accuracy, "rmse"),
        "mae": _metric_float(accuracy, "mae"),
        "mape": _metric_float(accuracy, "mape"),
        "r2": _metric_float(accuracy, "r2"),
    }
    accuracy_available = all(value is not None for value in accuracy_metrics.values())
    if model_key in {"xgboost", "ft_transformer"} and not accuracy_available:
        return None
    if model_key == "interpolation":
        accuracy_metrics = {key: None for key in ("rmse", "mae", "mape", "r2")}

    display_names = {
        "interpolation": "Interpolasyon",
        "xgboost": "XGBoost",
        "ft_transformer": "FT-Transformer",
    }
    return {
        "raw": item,
        "display_name": display_names.get(model_key, model_key),
        "latency_p50_ms": _metric_float(latency, "p50") or _metric_float(latency, "median"),
        "latency_p95_ms": _metric_float(latency, "p95"),
        "peak_rss_mb": _metric_float(memory, "rss_peak"),
        "cpu_avg_percent": _metric_float(cpu, "process_avg"),
        "cpu_peak_percent": _metric_float(cpu, "process_peak_sample"),
        "model_size_mb": _metric_float(item, "model_size_mb"),
        "metrics": accuracy_metrics,
        "accuracy_included": model_key in {"xgboost", "ft_transformer"},
    }


def _cost_simulator_explanation() -> dict:
    return {
        "title": "Maliyet paneli neyi hesaplıyor?",
        "summary": (
            "Bu panel artık benchmark runner çıktısını ana kaynak olarak kullanır. Interpolasyon, "
            "XGBoost ve FT-Transformer için ölçülen p95 gecikme, peak RSS bellek, CPU kullanımı "
            "ve model/artefakt boyutu görünür. Doğruluk maliyeti yalnız learned model ailesi olan "
            "XGBoost ve FT-Transformer için hesaplanır; interpolasyon referans aile olarak runtime "
            "maliyetiyle gösterilir."
        ),
        "sources": [
            {
                "name": "Gerçek benchmark dosyası",
                "detail": (
                    "Önce artifacts/benchmarks/pi_benchmark_latest.json okunur. Bu dosya "
                    "scripts/run_pi_benchmark.py komutuyla Pi/WSL/hedef cihaz üzerinde üretilir."
                ),
            },
            {
                "name": "Doğruluk metrikleri",
                "detail": (
                    "Benchmark runner seçilen gerçek tablo satırlarında tahmin alır ve learned modeller "
                    "için RMSE, MAE, MAPE ve R2 metriklerini aynı koşuda hesaplar. Interpolasyon bu "
                    "doğruluk yarışına sokulmaz; referans tablo ailesi olarak değerlendirilir."
                ),
            },
            {
                "name": "Runtime metrikleri",
                "detail": (
                    "Tekil inference tekrarlarından p50/p95 latency, process peak RSS RAM ve process CPU "
                    "kullanımı ölçülür. Pi'de vcgencmd varsa sıcaklık ve throttle bilgisi de dosyaya yazılır."
                ),
            },
        ],
        "targets": [
            {"label": "RMSE hedefi", "value": "0.0030"},
            {"label": "MAE hedefi", "value": "0.0015"},
            {"label": "MAPE hedefi", "value": "2.0"},
            {"label": "R2 gap hedefi", "value": "0.0010"},
            {"label": "Gecikme hedefi", "value": "10 ms"},
            {"label": "Bellek hedefi", "value": "seçilen RAM bütçesi"},
            {"label": "CPU hedefi", "value": "80%"},
        ],
        "formulas": [
            {
                "name": "Accuracy Cost",
                "formula": "mean(RMSE/0.0030, MAE/0.0015, MAPE/2.0, max(1 - R2, 0)/0.0010)",
                "detail": (
                    "Düşük değer daha iyidir. Modelin hata metrikleri hedeflerin altındaysa accuracy cost "
                    "1 civarına veya altına iner; hedeflerden uzaklaştıkça büyür."
                ),
            },
            {
                "name": "Latency Cost",
                "formula": "measured_p95_latency_ms / 10.0",
                "detail": (
                    "Ölçülen p95 tekil çıkarım gecikmesi 10 ms hedefine göre normalize edilir. CPU speed "
                    "slider'ı sadece ölçümden ölçeklenmiş senaryo üretir."
                ),
            },
            {
                "name": "Memory Cost",
                "formula": "measured_peak_rss_mb / memory_budget_mb",
                "detail": (
                    "Ölçülen peak RSS değeri seçilen RAM bütçesine bölünür. Pi 3 için bu bütçe genelde "
                    "1 GB civarında tutulmalıdır."
                ),
            },
            {
                "name": "CPU Cost",
                "formula": "measured_avg_cpu_percent / 80.0",
                "detail": "Benchmark koşusundaki ortalama process CPU kullanımı %80 hedefe göre normalize edilir.",
            },
            {
                "name": "Composite Cost",
                "formula": "w_accuracy*Accuracy + w_latency*Latency + w_memory*Memory + w_cpu*CPU",
                "detail": (
                    "Slider ağırlıkları önce toplamı 1 olacak şekilde normalize edilir. Son maliyet bu dört "
                    "bileşenin ağırlıklı toplamıdır; düşük composite cost daha iyi aday anlamına gelir."
                ),
            },
            {
                "name": "Fit Score",
                "formula": "100 / (1 + 0.30 * CompositeCost)",
                "detail": (
                    "Bu skor kazananı otomatik 100 yapmaz. Mutlak uygunluk hissi vermek için maliyetten türetilir; "
                    "cost yükseldikçe skor düşer."
                ),
            },
        ],
        "runtime_formulas": [],
        "notes": [
            "Ölçüm yoksa maliyet paneli skor üretmez; önce Pi benchmark komutu çalıştırılmalıdır.",
            "Interpolasyon doğruluk yarışına sokulmaz; fakat latency, RAM ve CPU maliyeti üçlü runtime kıyasında gösterilir.",
            "RMSE, MAE, MAPE, latency, bellek ve CPU düşük oldukça iyidir; R2 için 1'e yakın olmak iyidir.",
        ],
    }


def _normalize_weights(*weights: float) -> tuple[float, ...]:
    clamped = [max(float(weight), 0.0) for weight in weights]
    total = sum(clamped)
    if total <= 0:
        return tuple([1.0 / len(clamped)] * len(clamped))
    return tuple(weight / total for weight in clamped)


def _dataset_python() -> Path:
    return Path(sys.executable)


def _dataset_pythonw() -> Path:
    candidate = Path(sys.executable).with_name("pythonw.exe")
    if os.name == "nt" and candidate.exists():
        return candidate
    return _dataset_python()


DATASET_PYTHON_DEPS = [
    {"module": "cv2", "label": "OpenCV", "package": "opencv-python"},
    {"module": "pdf2image", "label": "pdf2image", "package": "pdf2image"},
    {"module": "pytesseract", "label": "pytesseract", "package": "pytesseract"},
    {"module": "PIL", "label": "Pillow", "package": "pillow"},
    {"module": "tqdm", "label": "tqdm", "package": "tqdm"},
    {"module": "skimage", "label": "scikit-image", "package": "scikit-image"},
]


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _find_binary(label: str, key: str, executable: str, candidates: list[Path], hint: str) -> dict:
    found = shutil.which(executable)
    install_command = f"install_{key}" if shutil.which("winget") else ""
    if found:
        return {"label": label, "key": key, "ok": True, "path": found, "hint": "", "install_command": ""}

    for candidate in candidates:
        try:
            if candidate.exists():
                marker_hint = "" if candidate.suffix.lower() == ".exe" else (
                    f"{label} paketi algılandı; ancak {executable} PATH'te görünmüyor. "
                    "Flask/terminal yeniden başlatmak veya ilgili bin klasörünü PATH'e eklemek gerekebilir."
                )
                return {
                    "label": label,
                    "key": key,
                    "ok": True,
                    "path": str(candidate),
                    "hint": marker_hint,
                    "install_command": "",
                }
        except OSError:
            continue

    if not install_command:
        hint = f"{hint} Winget bulunamadığı için otomatik kurulum butonu kapalı."
    return {"label": label, "key": key, "ok": False, "path": "", "hint": hint, "install_command": install_command}


def _poppler_candidates() -> list[Path]:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    user_profile = Path(os.environ.get("USERPROFILE", ""))
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    winget_package_root = local_app_data / "Microsoft" / "WinGet" / "Packages"
    winget_poppler_root = winget_package_root / "oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"

    candidates = [
        program_files / "poppler" / "Library" / "bin" / "pdftoppm.exe",
        program_files / "poppler" / "bin" / "pdftoppm.exe",
        Path("C:/poppler/Library/bin/pdftoppm.exe"),
        Path("C:/poppler/bin/pdftoppm.exe"),
        Path("C:/tools/poppler/Library/bin/pdftoppm.exe"),
        Path("C:/tools/poppler/bin/pdftoppm.exe"),
        Path("C:/ProgramData/chocolatey/bin/pdftoppm.exe"),
        user_profile / "scoop" / "shims" / "pdftoppm.exe",
        winget_poppler_root / "poppler-25.07.0" / "Library" / "bin" / "pdftoppm.exe",
        winget_poppler_root / "poppler-25.07.0" / "bin" / "pdftoppm.exe",
        winget_poppler_root,
    ]

    if winget_package_root.exists():
        try:
            candidates.extend(winget_package_root.glob("*Poppler*"))
        except OSError:
            pass
    return candidates


def _dataset_python_dependency_status() -> list[dict]:
    statuses = []
    for dep in DATASET_PYTHON_DEPS:
        ok = _module_available(dep["module"])
        statuses.append({
            **dep,
            "ok": ok,
            "hint": "" if ok else f"Kur: pip install {dep['package']}",
            "install_command": "" if ok else "install_deps",
        })
    return statuses


def _dataset_system_dependency_status() -> list[dict]:
    return [
        _find_binary(
            "Poppler",
            "poppler",
            "pdftoppm",
            _poppler_candidates(),
            "PDF sayfalarını görsele çevirmek için Poppler gerekir. Windows'ta poppler bin klasörünü PATH'e ekleyebilirsin.",
        ),
        _find_binary(
            "Tesseract",
            "tesseract",
            "tesseract",
            [
                Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
            ],
            "OCR için Tesseract binary gerekir. Windows'ta Tesseract-OCR kurup PATH'e eklemek yeterli.",
        ),
    ]


def _dataset_tool_command_map() -> dict[str, dict]:
    python = _dataset_python()
    pythonw = _dataset_pythonw()
    deps = [dep["package"] for dep in DATASET_PYTHON_DEPS]
    return {
        "install_deps": {
            "id": "install_deps",
            "label": "Dataset Python Bağımlılıklarını Kur",
            "description": "OpenCV, OCR yardımcıları ve dataset tool paketlerini ana proje ortamına kurar.",
            "command": [str(python), "-m", "pip", "install", *deps],
            "display_command": f"{python} -m pip install {' '.join(deps)}",
            "eta": "~1-5 dk",
            "artifacts": ["Ana .venv Python paketleri"],
            "detached": False,
        },
        "install_poppler": {
            "id": "install_poppler",
            "label": "Poppler Kur",
            "description": "PDF sayfalarını görsele çevirmek için Poppler binary paketini winget ile kurar.",
            "command": [
                "winget",
                "install",
                "--id",
                "oschwartz10612.Poppler",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            "display_command": "winget install --id oschwartz10612.Poppler -e --accept-package-agreements --accept-source-agreements",
            "eta": "~1-5 dk",
            "artifacts": ["Poppler system binary", "pdftoppm"],
            "detached": False,
            "requires_winget": True,
        },
        "install_tesseract": {
            "id": "install_tesseract",
            "label": "Tesseract OCR Kur",
            "description": "OCR için Tesseract binary paketini winget ile kurar.",
            "command": [
                "winget",
                "install",
                "--id",
                "UB-Mannheim.TesseractOCR",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            "display_command": "winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements",
            "eta": "~1-5 dk",
            "artifacts": ["Tesseract system binary"],
            "detached": False,
            "requires_winget": True,
        },
        "env_check": {
            "id": "env_check",
            "label": "Ortam Kontrolü",
            "description": "Python, CUDA ve temel bağımlılık durumunu kontrol eder.",
            "command": [str(python), "2-check_env_sys.py"],
            "display_command": f"{python} 2-check_env_sys.py",
            "eta": "~5-20 sn",
            "artifacts": [],
            "detached": False,
        },
        "open_gui": {
            "id": "open_gui",
            "label": "Dataset GUI Aç",
            "description": "İç dataset builder Tkinter panelini ayrı pencere olarak açar.",
            "command": [str(pythonw), "synthetic_data_gui.py"],
            "display_command": f"{pythonw} synthetic_data_gui.py",
            "eta": "Pencere açık kaldığı sürece",
            "artifacts": [],
            "detached": True,
        },
        "synthetic_demo": {
            "id": "synthetic_demo",
            "label": "Sentetik Grafik Üret",
            "description": "Küçük demo seti üretir; tam üretim için README'deki 5000+ komutları kullanılır.",
            "command": [str(python), "3-synthetic_production.py", "20", "1", "--prob", "0.3"],
            "display_command": f"{python} 3-synthetic_production.py 20 1 --prob 0.3",
            "eta": "~1-5 dk",
            "artifacts": ["dataset_production/"],
            "detached": False,
        },
        "train_demo": {
            "id": "train_demo",
            "label": "U-Net Eğit",
            "description": "Demo amaçlı 1 epoch CPU eğitimi dener. Dataset yoksa log'da uyarı verir.",
            "command": [str(python), "train_unet.py", "--dataset", "dataset_production", "--epochs", "1", "--batch_size", "2", "--num_workers", "0", "--device", "cpu"],
            "display_command": f"{python} train_unet.py --dataset dataset_production --epochs 1 --batch_size 2 --num_workers 0 --device cpu",
            "eta": "~1-10 dk",
            "artifacts": ["checkpoints/", "training_history.png"],
            "detached": False,
        },
        "segment_export_demo": {
            "id": "segment_export_demo",
            "label": "Segmentasyon / Excel Export",
            "description": "Mevcut modelle küçük demo görsel setini işler ve curve extraction yolu varsa Excel çıktısı üretir.",
            "command": [
                str(python),
                "5-segment_curves.py",
                "--model",
                str(DATASET_APP_DIR / "Model" / "best_unet_model (3).pth"),
                "--input_dir",
                "demo_graphs",
                "--output_dir",
                "demo_segmentation_results",
                "--device",
                "cpu",
                "--no_confidence",
                "--extract_curve_data",
                "--curve_data_output",
                "curve_data.xlsx",
            ],
            "display_command": f"{python} 5-segment_curves.py --model \"Model/best_unet_model (3).pth\" --input_dir demo_graphs --output_dir demo_segmentation_results --device cpu --no_confidence --extract_curve_data --curve_data_output curve_data.xlsx",
            "eta": "~1-5 dk",
            "artifacts": ["demo_segmentation_results/", "curve_data.xlsx"],
            "detached": False,
        },
    }


def _dataset_tool_status() -> dict:
    return {
        "app_dir": str(DATASET_APP_DIR),
        "app_dir_exists": DATASET_APP_DIR.exists(),
        "python": str(_dataset_python()),
        "pythonw": str(_dataset_pythonw()),
        "python_env": "project",
        "using_internal_tools": True,
        "gui_script": (DATASET_APP_DIR / "synthetic_data_gui.py").exists(),
        "synthetic_script": (DATASET_APP_DIR / "3-synthetic_production.py").exists(),
        "train_script": (DATASET_APP_DIR / "train_unet.py").exists(),
        "segment_script": (DATASET_APP_DIR / "5-segment_curves.py").exists(),
        "model_file": (DATASET_APP_DIR / "Model" / "best_unet_model (3).pth").exists(),
        "dataset_dir": (DATASET_APP_DIR / "dataset_production").exists(),
        "grafikler_dir": (DATASET_APP_DIR / "Grafikler").exists(),
        "demo_graphs_dir": (DATASET_APP_DIR / "demo_graphs").exists(),
        "segmentation_results_dir": (DATASET_APP_DIR / "segmentation_results").exists(),
        "demo_segmentation_results_dir": (DATASET_APP_DIR / "demo_segmentation_results").exists(),
        "python_packages": _dataset_python_dependency_status(),
        "system_tools": _dataset_system_dependency_status(),
    }


def _dataset_gui_defaults() -> dict:
    return {
        "generate": {
            "enabled": True,
            "clean_data": False,
            "num_images": "5000",
            "num_workers": "6",
            "rotation": "3.0",
            "noise": "15.0",
            "jpeg": "70",
            "blur": "0",
            "shadow": "0",
            "prob": "0.7",
            "flip": "0",
            "shear": "0.0",
            "hue": "0",
            "sat": "0",
            "cutout": "0",
            "motion": "0",
        },
        "train": {
            "enabled": True,
            "use_colab": False,
            "colab_link": "https://colab.research.google.com/",
            "dataset_path": "dataset_production",
            "epochs": "50",
            "batch_size": "8",
            "learning_rate": "0.001",
            "image_size": "256",
            "early_stopping_mode": "Otomatik",
            "patience": "10",
            "num_workers": "0",
            "device": "cpu",
        },
        "inference": {
            "enabled": True,
            "mode": "folder",
            "model_path": "Model/best_unet_model (3).pth",
            "input_dir": "Grafikler",
            "pdf_path": "",
            "pdf_pages": "all",
            "threshold": "0.5",
            "input_size": "256",
            "output_dir": "segmentation_results",
            "clean_output": True,
            "device": "cpu",
            "no_confidence": True,
        },
        "extract": {
            "enabled": True,
            "curve_data_output": "curve_data.xlsx",
        },
        "correction": {
            "enabled": True,
            "correction_dir": "segmentation_results",
        },
    }


def _safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "evet", "aktif"}


def _safe_text(value, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    return text if text else default


def _safe_dataset_path(path_value: str) -> Path:
    path = Path(_safe_text(path_value, "."))
    if not path.is_absolute():
        path = DATASET_APP_DIR / path
    return path.resolve()


def _path_inside_dataset_tool(path: Path) -> bool:
    root = DATASET_APP_DIR.resolve()
    try:
        return path == root or root in path.parents
    except RuntimeError:
        return False


def _delete_dataset_subdir(path_value: str) -> str:
    target = _safe_dataset_path(path_value)
    if not _path_inside_dataset_tool(target):
        return f"Güvenlik nedeniyle silme atlandı; klasör dataset tool dışında: {target}"
    if not target.exists():
        return f"Temizlenecek klasör zaten yok: {target}"
    if not target.is_dir():
        return f"Temizleme atlandı; hedef klasör değil: {target}"
    shutil.rmtree(target)
    return f"Klasör temizlendi: {target}"


def _dataset_gui_build_steps(config: dict) -> list[dict]:
    defaults = _dataset_gui_defaults()
    generate_cfg = {**defaults["generate"], **(config.get("generate") or {})}
    train_cfg = {**defaults["train"], **(config.get("train") or {})}
    inference_cfg = {**defaults["inference"], **(config.get("inference") or {})}
    extract_cfg = {**defaults["extract"], **(config.get("extract") or {})}
    correction_cfg = {**defaults["correction"], **(config.get("correction") or {})}
    python = str(_dataset_python())
    steps: list[dict] = []

    if _safe_bool(generate_cfg.get("enabled"), True):
        if _safe_bool(generate_cfg.get("clean_data"), False):
            steps.append({
                "kind": "cleanup",
                "label": "Üretim öncesi dataset temizliği",
                "path": "dataset_production",
            })
        command = [
            python,
            "3-synthetic_production.py",
            _safe_text(generate_cfg.get("num_images"), "5000"),
            _safe_text(generate_cfg.get("num_workers"), "1"),
            "--rotation", _safe_text(generate_cfg.get("rotation"), "3.0"),
            "--noise", _safe_text(generate_cfg.get("noise"), "15.0"),
            "--jpeg", _safe_text(generate_cfg.get("jpeg"), "70"),
            "--blur", _safe_text(generate_cfg.get("blur"), "0"),
            "--shadow", _safe_text(generate_cfg.get("shadow"), "0"),
            "--prob", _safe_text(generate_cfg.get("prob"), "0.7"),
            "--flip", _safe_text(generate_cfg.get("flip"), "0"),
            "--shear", _safe_text(generate_cfg.get("shear"), "0.0"),
            "--hue", _safe_text(generate_cfg.get("hue"), "0"),
            "--sat", _safe_text(generate_cfg.get("sat"), "0"),
            "--cutout", _safe_text(generate_cfg.get("cutout"), "0"),
            "--motion", _safe_text(generate_cfg.get("motion"), "0"),
        ]
        steps.append({"kind": "command", "label": "Sentetik Veri Üretimi", "command": command})

    if _safe_bool(train_cfg.get("enabled"), True):
        if _safe_bool(train_cfg.get("use_colab"), False):
            colab_link = _safe_text(train_cfg.get("colab_link"), "https://colab.research.google.com/")
            steps.append({
                "kind": "message",
                "label": "Google Colab Eğitimi",
                "text": f"Colab modu seçildi. Bu web paneli tarayıcıyı zorla açmaz; notebook linki: {colab_link}",
            })
        else:
            mode = _safe_text(train_cfg.get("early_stopping_mode"), "Otomatik")
            patience = "10" if mode == "Otomatik" else ("0" if mode == "Kapali" else _safe_text(train_cfg.get("patience"), "10"))
            command = [
                python,
                "train_unet.py",
                "--dataset", _safe_text(train_cfg.get("dataset_path"), "dataset_production"),
                "--epochs", _safe_text(train_cfg.get("epochs"), "50"),
                "--batch_size", _safe_text(train_cfg.get("batch_size"), "8"),
                "--learning_rate", _safe_text(train_cfg.get("learning_rate"), "0.001"),
                "--image_size", _safe_text(train_cfg.get("image_size"), "256"),
                "--patience", patience,
                "--num_workers", _safe_text(train_cfg.get("num_workers"), "0"),
                "--device", _safe_text(train_cfg.get("device"), "cpu"),
            ]
            steps.append({"kind": "command", "label": "Model Eğitimi", "command": command})

    if _safe_bool(inference_cfg.get("enabled"), True):
        output_dir = _safe_text(inference_cfg.get("output_dir"), "segmentation_results")
        if _safe_bool(inference_cfg.get("clean_output"), True):
            steps.append({"kind": "cleanup", "label": "Çıkış klasörü temizliği", "path": output_dir})

        command = [
            python,
            "-u",
            "5-segment_curves.py",
            "--model", _safe_text(inference_cfg.get("model_path"), "Model/best_unet_model (3).pth"),
            "--output_dir", output_dir,
            "--threshold", _safe_text(inference_cfg.get("threshold"), "0.5"),
            "--device", _safe_text(inference_cfg.get("device"), "cpu"),
            "--input_size", _safe_text(inference_cfg.get("input_size"), "256"),
        ]
        if _safe_bool(inference_cfg.get("no_confidence"), True):
            command.append("--no_confidence")

        mode = _safe_text(inference_cfg.get("mode"), "folder")
        if mode == "pdf":
            command.extend(["--pdf", _safe_text(inference_cfg.get("pdf_path"), "")])
            command.extend(["--pages", _safe_text(inference_cfg.get("pdf_pages"), "all")])
        else:
            command.extend(["--input_dir", _safe_text(inference_cfg.get("input_dir"), "Grafikler")])

        if _safe_bool(extract_cfg.get("enabled"), True):
            command.extend(["--extract_curve_data", "--curve_data_output", _safe_text(extract_cfg.get("curve_data_output"), "curve_data.xlsx")])
        steps.append({"kind": "command", "label": "Eğri Çıkarımı ve Segmentasyon", "command": command})

    if _safe_bool(extract_cfg.get("enabled"), True) and not _safe_bool(inference_cfg.get("enabled"), True):
        output_dir = _safe_text(inference_cfg.get("output_dir"), "segmentation_results")
        original_dir = f"{output_dir}_extracted" if _safe_text(inference_cfg.get("mode"), "folder") == "pdf" else _safe_text(inference_cfg.get("input_dir"), "Grafikler")
        command = [
            python,
            "-u",
            "extract_curve_data.py",
            "--segmentation_dir", output_dir,
            "--original_dir", original_dir,
            "--output", str(Path(output_dir) / _safe_text(extract_cfg.get("curve_data_output"), "curve_data.xlsx")),
        ]
        steps.append({"kind": "command", "label": "Excel Aktarımı", "command": command})

    if _safe_bool(correction_cfg.get("enabled"), True):
        correction_script = DATASET_APP_DIR / "6-correction_tool.py"
        if not correction_script.exists():
            steps.append({
                "kind": "message",
                "label": "Hata Düzeltme (Baseline)",
                "text": "6-correction_tool.py proje içi Dataset Builder kopyasında bulunmadığı için bu adım atlandı.",
            })
        else:
            correction_dir = _safe_text(correction_cfg.get("correction_dir"), "segmentation_results")
            for suffix in ["_One_Engine", "_Two_Engine", ""]:
                excel_input = Path(correction_dir) / f"curve_data{suffix}.xlsx"
                excel_output = Path(correction_dir) / f"curve_data{suffix}_HAM_Baseline.xlsx"
                command = [python, "6-correction_tool.py", "-i", str(excel_input), "-o", str(excel_output)]
                steps.append({"kind": "command", "label": f"Hata Düzeltme {suffix or '(Genel)'}", "command": command})

    return steps


def _dataset_tool_hint(line: str) -> str:
    if "No module named 'cv2'" in line:
        return "OpenCV eksik görünüyor. Dataset Python Bağımlılıklarını Kur komutunu çalıştır."
    if "No module named 'pdf2image'" in line:
        return "pdf2image eksik görünüyor. Dataset Python Bağımlılıklarını Kur komutunu çalıştır."
    if "No module named 'pytesseract'" in line:
        return "pytesseract eksik görünüyor. Python paketi kurulsa bile OCR için Tesseract binary ayrıca gerekebilir."
    if "No such file" in line or "cannot find" in line.lower():
        return "Dosya yolu eksik olabilir. Dataset Tool Durumu kartlarındaki model/grafik/dataset kontrollerine bak."
    return ""


def _is_already_installed_output(command_id: str, output_lines: list[str]) -> bool:
    if not command_id.startswith("install_"):
        return False

    output = "\n".join(output_lines).lower()
    benign_patterns = [
        "already installed",
        "already exists",
        "no newer package versions are available",
        "no available upgrade",
        "no applicable update",
        "found an existing package already installed",
        "package is already installed",
        "requirement already satisfied",
        "zaten kurulu",
        "zaten yüklü",
        "zaten yuklu",
        "halihazırda yüklü",
        "halihazirda yuklu",
        "yüklenmiş durumda",
        "yuklenmis durumda",
        "paket zaten",
        "daha yeni paket sürümü yok",
        "daha yeni paket surumu yok",
    ]
    return any(pattern in output for pattern in benign_patterns)


def _public_dataset_commands() -> list[dict]:
    """Return dataset tool commands without exposing executable internals."""
    commands = []
    for command in _dataset_tool_command_map().values():
        if command.get("requires_winget") and not shutil.which("winget"):
            continue
        if command["id"] == "open_gui":
            commands.append({
                "id": command["id"],
                "label": "Dataset GUI (Flask Panel)",
                "description": "Ayrı Tkinter penceresi açmadan aynı sekmede web panelini gösterir.",
                "display_command": "Flask içi Dataset GUI paneli",
                "eta": "anında",
                "artifacts": [],
                "detached": False,
            })
            continue
        commands.append({
            "id": command["id"],
            "label": command["label"],
            "description": command["description"],
            "display_command": command["display_command"],
            "eta": command["eta"],
            "artifacts": command["artifacts"],
            "detached": command["detached"],
        })
    return commands


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


def _read_text_excerpt(path: Path, max_chars: int = 14000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n... [devamı dosyada]"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    readme_paths = [
        ("Ana README", PROJECT_ROOT / "README.md"),
        ("Dataset Builder README", DATASET_APP_DIR / "README.md"),
        ("XGBoost README", PROJECT_ROOT / "docs" / "methods" / "xgboost.md"),
        ("FT-Transformer README", PROJECT_ROOT / "docs" / "methods" / "ft_transformer.md"),
        ("Interpolasyon README", PROJECT_ROOT / "src" / "interpolation" / "README.md"),
    ]
    return jsonify({
        "tabs": [
            {"name": "Genel Bakış", "description": "Model rapor metrikleri, slice özetleri ve hata dağılımı."},
            {"name": "Karşılaştırma", "description": "Interpolasyon referansı yanında XGBoost ve FT-Transformer çıktılarını satır/grafik bazında kıyaslar."},
            {"name": "Maliyet", "description": "Interpolasyon, XGBoost ve FT-Transformer için gerçek benchmark tabanlı hız-bellek-CPU kıyası; doğruluk kıyası XGBoost ve FT-Transformer arasındadır."},
            {"name": "Tekil Tahmin", "description": "Custom input veya hazır senaryo ile tek uçuş koşulu tahmini."},
            {"name": "Nomogram", "description": "Handbook benzeri kesit grafikleri üretir."},
            {"name": "Veri Üretimi", "description": "Grafik segmentasyon ve sentetik dataset araçlarını demo olarak çalıştırır."},
            {"name": "Setup", "description": "Veri pipeline, eğitim ve toplu rapor komutlarını çalıştırır."},
        ],
        "methods": [
            {"name": "Interpolasyon", "role": "Klasik tablo tabanlı referans aile; maliyet yarışına dahil edilmez."},
            {"name": "XGBoost", "role": "Güçlü ağaç tabanlı baseline ve pratik kıyas modeli."},
            {"name": "FT-Transformer", "role": "Ana araştırma modeli; tabular feature tokenizer + Transformer encoder yaklaşımı."},
        ],
        "readmes": [
            {
                "title": title,
                "path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
                "exists": path.exists(),
                "content": _read_text_excerpt(path),
            }
            for title, path in readme_paths
        ],
        "cost_simulator": _cost_simulator_explanation(),
    })


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


@app.route("/api/interpolation/methods")
def api_interpolation_methods():
    return jsonify([
        {"key": key, "label": label, "default": key == DEFAULT_INTERPOLATION_METHOD}
        for key, label in INTERPOLATION_METHODS.items()
    ])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    requested_methods = body.get("methods") or body.get("method") or "all"
    if isinstance(requested_methods, str):
        requested_methods = [requested_methods]
    requested_methods = {str(method).lower() for method in requested_methods}
    include_all = "all" in requested_methods or "both" in requested_methods
    include_interpolation = include_all or "interpolation" in requested_methods
    include_xgboost = include_all or "xgboost" in requested_methods
    include_ft = include_all or "ft_transformer" in requested_methods or "ft-transformer" in requested_methods
    interpolation_method = str(body.get("interpolation_method", DEFAULT_INTERPOLATION_METHOD))

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

    if include_interpolation:
        interpolation, interpolation_err = _get_interpolation()
        if interpolation:
            try:
                result["interpolation"] = float(
                    interpolation.predict_one(
                        engine_type=str(body["engine_type"]),
                        altitude=float(body["altitude"]),
                        gross_weight=float(body["gross_weight"]),
                        drag_index=float(body["drag_index"]),
                        mach=float(body["mach"]),
                        method=interpolation_method,
                    )
                )
                result["interpolation_method"] = interpolation.method_label(interpolation_method)
                result["interpolation_note"] = "Klasik tablo interpolasyonu fuel_flow girdisini kullanmaz."
            except Exception as exc:
                result["interpolation_error"] = str(exc)
        elif interpolation_err:
            result["interpolation_error"] = interpolation_err

    if include_xgboost:
        xgb, xgb_err = _get_xgb()
        if xgb:
            result["xgboost"] = float(xgb.predict_from_frame(frame))
        elif xgb_err:
            result["xgboost_error"] = xgb_err

    if include_ft:
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

@app.route("/api/benchmark/pi/latest")
def api_pi_benchmark_latest():
    benchmark = _load_pi_benchmark_latest()
    if benchmark is None:
        return jsonify({
            "error": "Henüz gerçek Pi benchmark ölçümü yok.",
            "command": (
                "python scripts/run_pi_benchmark.py --models interpolation,xgboost,ft_transformer "
                "--sample-size 200 --warmup 20 --repetitions 200 --device cpu"
            ),
            "benchmark_path": str(PI_BENCHMARK_LATEST),
        }), 404
    return jsonify(benchmark)


@app.route("/api/benchmark/pi/run", methods=["POST"])
def api_pi_benchmark_run():
    body = request.get_json(silent=True) or {}
    models = str(body.get("models", "interpolation,xgboost,ft_transformer"))
    allowed_models = {"interpolation", "xgboost", "ft_transformer"}
    requested_models = [item.strip() for item in models.split(",") if item.strip()]
    if not requested_models or any(item not in allowed_models for item in requested_models):
        return jsonify({"error": "Geçersiz model listesi."}), 400
    sample_size = max(1, min(int(body.get("sample_size", 200)), 5000))
    warmup = max(0, min(int(body.get("warmup", 20)), 1000))
    repetitions = max(1, min(int(body.get("repetitions", 200)), 10000))
    device = str(body.get("device", "cpu"))
    if device != "cpu":
        return jsonify({"error": "Pi benchmark endpoint'i şu an yalnız cpu device destekler."}), 400

    command = [
        sys.executable,
        "scripts/run_pi_benchmark.py",
        "--models",
        ",".join(requested_models),
        "--sample-size",
        str(sample_size),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
        "--device",
        "cpu",
    ]

    def generate():
        yield f"data: {json.dumps({'type': 'output', 'text': '$ ' + ' '.join(command)})}\n\n"
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            yield f"data: {json.dumps({'type': 'output', 'text': line.rstrip()})}\n\n"
        process.wait()
        yield f"data: {json.dumps({'type': 'done', 'exit_code': process.returncode})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/compare/metrics")
def api_compare_metrics():
    result = {}
    result["interpolation"] = _load_model_summary("interpolation")["metrics"]
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
    page_df = merged.iloc[start : start + per_page].copy()

    interpolation, interpolation_err = _get_interpolation()
    if interpolation is not None and not page_df.empty:
        try:
            interp_pred = interpolation.predict_many_from_frame(page_df, DEFAULT_INTERPOLATION_METHOD)
            page_df["interpolation_predicted"] = interp_pred
            page_df["interpolation_absolute_error"] = np.abs(page_df["actual_specific_range"].to_numpy(dtype=float) - interp_pred)
            page_df["interpolation_signed_error"] = interp_pred - page_df["actual_specific_range"].to_numpy(dtype=float)
        except Exception as exc:
            page_df["interpolation_error"] = str(exc)
    elif interpolation_err:
        page_df["interpolation_error"] = interpolation_err

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

    merged = merged.sort_values(["engine_type", "altitude", "mach"]).copy()
    if len(merged) <= 1000:
        interpolation, _ = _get_interpolation()
        if interpolation is not None:
            try:
                interp_pred = interpolation.predict_many_from_frame(merged, DEFAULT_INTERPOLATION_METHOD)
                merged["interpolation_predicted"] = interp_pred
                merged["interpolation_absolute_error"] = np.abs(merged["actual_specific_range"].to_numpy(dtype=float) - interp_pred)
            except Exception:
                logging.exception("Interpolation chart-data generation failed")
    return jsonify(_df_to_records(merged))


def _compare_tolerance_curve_data() -> dict:
    xgb_paths = _report_paths("xgboost")
    ft_paths = _report_paths("ft_transformer")
    if not xgb_paths["row_level"].exists() or not ft_paths["row_level"].exists():
        raise FileNotFoundError("Both model reports are required")

    xgb_df = pd.read_csv(xgb_paths["row_level"])
    ft_df = pd.read_csv(ft_paths["row_level"])
    xgb_errors = xgb_df["xgboost_absolute_error"].dropna().to_numpy(dtype=float)
    ft_errors = ft_df["ft_transformer_absolute_error"].dropna().to_numpy(dtype=float)
    if len(xgb_errors) == 0 or len(ft_errors) == 0:
        raise ValueError("Error columns are empty")

    combined = np.concatenate([xgb_errors, ft_errors])
    max_threshold = float(np.quantile(combined, 0.99))
    if max_threshold <= 0:
        max_threshold = float(max(np.max(combined), 1e-6))
    points = max(10, min(int(request.args.get("points", 80, type=int)), 200))
    thresholds = np.linspace(0.0, max_threshold, points)

    def success_curve(errors: np.ndarray) -> list[float]:
        return [float(np.mean(errors <= threshold) * 100.0) for threshold in thresholds]

    return {
        "thresholds": thresholds.tolist(),
        "xgboost": success_curve(xgb_errors),
        "ft_transformer": success_curve(ft_errors),
        "summary": {
            "max_threshold": max_threshold,
            "xgboost_median_error": float(np.median(xgb_errors)),
            "ft_transformer_median_error": float(np.median(ft_errors)),
            "xgboost_p95_error": float(np.quantile(xgb_errors, 0.95)),
            "ft_transformer_p95_error": float(np.quantile(ft_errors, 0.95)),
        },
        "note": "Y ekseni, ilgili hata eşiğinin altında kalan satır yüzdesidir. Yüksek eğri daha iyidir.",
    }


@app.route("/api/compare/tolerance-curve")
def api_compare_tolerance_curve():
    """Regression analogue of an F1/ROC-style curve.

    For each absolute-error threshold, report the percentage of rows whose
    prediction error is within that tolerance. Interpolation is intentionally
    excluded because it is used as the deterministic reference family.
    """
    try:
        return jsonify(_compare_tolerance_curve_data())
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/compare/tolerance-curve.svg")
def api_compare_tolerance_curve_svg():
    """Return a self-contained SVG so the UI is not blocked by Plotly/CDN issues."""
    try:
        data = _compare_tolerance_curve_data()
    except (FileNotFoundError, ValueError) as exc:
        return Response(f"<svg xmlns='http://www.w3.org/2000/svg'><text x='20' y='40'>{exc}</text></svg>", mimetype="image/svg+xml", status=404)

    thresholds = np.asarray(data["thresholds"], dtype=float)
    xgb = np.asarray(data["xgboost"], dtype=float)
    ft = np.asarray(data["ft_transformer"], dtype=float)
    width, height = 920, 360
    left, right, top, bottom = 72, 28, 32, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max(float(thresholds.max()), 1e-9)

    def xy_path(values: np.ndarray) -> str:
        points = []
        for x_value, y_value in zip(thresholds, values):
            x = left + (float(x_value) / max_x) * plot_w
            y = top + (1.0 - (float(y_value) / 100.0)) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    grid_lines = []
    for pct in [0, 25, 50, 75, 100]:
        y = top + (1.0 - pct / 100.0) * plot_h
        grid_lines.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='rgba(148,163,184,0.22)'/>")
        grid_lines.append(f"<text x='{left-12}' y='{y+4:.1f}' fill='#94a3b8' font-size='12' text-anchor='end'>{pct}</text>")

    x_labels = []
    for ratio in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + ratio * plot_w
        value = max_x * ratio
        x_labels.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height-bottom}' stroke='rgba(148,163,184,0.12)'/>")
        x_labels.append(f"<text x='{x:.1f}' y='{height-24}' fill='#94a3b8' font-size='12' text-anchor='middle'>{value:.4f}</text>")

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 {width} {height}" role="img" aria-label="Hata toleransı başarı eğrisi">
  <rect width="{width}" height="{height}" rx="16" fill="#020617"/>
  <text x="{left}" y="22" fill="#f1f5f9" font-size="15" font-family="Inter, Segoe UI, sans-serif">Hata eşiğine göre kabul edilen satır oranı</text>
  {''.join(grid_lines)}
  {''.join(x_labels)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#475569"/>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#475569"/>
  <polyline points="{xy_path(xgb)}" fill="none" stroke="#fbbf24" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{xy_path(ft)}" fill="none" stroke="#a78bfa" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{width-230}" cy="25" r="5" fill="#fbbf24"/><text x="{width-218}" y="30" fill="#cbd5e1" font-size="13">XGBoost</text>
  <circle cx="{width-130}" cy="25" r="5" fill="#a78bfa"/><text x="{width-118}" y="30" fill="#cbd5e1" font-size="13">FT-Transformer</text>
  <text x="{width/2}" y="{height-6}" fill="#94a3b8" font-size="12" text-anchor="middle">Mutlak hata eşiği</text>
  <text x="18" y="{height/2}" fill="#94a3b8" font-size="12" text-anchor="middle" transform="rotate(-90 18 {height/2})">Kabul edilen satır oranı (%)</text>
</svg>
"""
    return Response(svg, mimetype="image/svg+xml")


@app.route("/api/compare/cost-simulator")
def api_compare_cost_simulator():
    accuracy_weight = request.args.get("accuracy_weight", 45, type=float)
    latency_weight = request.args.get("latency_weight", 25, type=float)
    memory_weight = request.args.get("memory_weight", 20, type=float)
    cpu_weight = request.args.get("cpu_weight", 10, type=float)
    cpu_speed_factor = request.args.get("cpu_speed_factor", 1.0, type=float)
    ram_budget_gb = request.args.get("ram_budget_gb", 1.0, type=float)

    accuracy_weight, latency_weight, memory_weight, cpu_weight = _normalize_weights(
        accuracy_weight,
        latency_weight,
        memory_weight,
        cpu_weight,
    )

    benchmark = _load_pi_benchmark_latest()
    benchmark_command = (
        "python scripts/run_pi_benchmark.py --models interpolation,xgboost,ft_transformer "
        "--sample-size 200 --warmup 20 --repetitions 200 --device cpu"
    )
    if benchmark is None:
        return jsonify({
            "error": "Henüz gerçek Pi benchmark ölçümü yok.",
            "command": benchmark_command,
            "benchmark_path": str(PI_BENCHMARK_LATEST),
        }), 404

    ram_budget_mb = max(ram_budget_gb * 1024.0, 256.0)
    cpu_speed_factor = max(float(cpu_speed_factor), 0.35)

    accuracy_targets = {
        "rmse": 0.0030,
        "mae": 0.0015,
        "mape": 2.0,
        "r2_gap": 0.0010,
    }
    latency_target_ms = 10.0
    cpu_target_percent = 80.0

    prepared: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for model_key in ("interpolation", "xgboost", "ft_transformer"):
        measured = _benchmark_model_payload(benchmark, model_key)
        if measured is None:
            raw_item = (benchmark.get("models") or {}).get(model_key) or {}
            skipped[model_key] = raw_item.get("skipped_reason") or "Bu model için eksiksiz ölçüm/metrik bulunamadı."
            continue
        metrics = measured["metrics"]
        accuracy_component = None
        if measured["accuracy_included"]:
            accuracy_component = float(np.mean([
                float(metrics["rmse"]) / accuracy_targets["rmse"],
                float(metrics["mae"]) / accuracy_targets["mae"],
                float(metrics["mape"]) / accuracy_targets["mape"],
                max(1.0 - float(metrics["r2"]), 0.0) / accuracy_targets["r2_gap"],
            ]))

        scenario_latency_p95 = measured["latency_p95_ms"] / cpu_speed_factor
        over_budget_ratio = max(measured["peak_rss_mb"] - ram_budget_mb, 0.0) / ram_budget_mb

        prepared[model_key] = {
            **measured,
            "accuracy_component": accuracy_component,
            "scenario_latency_p95_ms": scenario_latency_p95,
            "ram_budget_utilization": measured["peak_rss_mb"] / ram_budget_mb,
            "over_budget_ratio": over_budget_ratio,
        }

    if not prepared:
        return jsonify({
            "error": "Benchmark dosyası bulundu ama kullanılabilir ölçüm yok.",
            "command": benchmark_command,
            "benchmark_path": str(PI_BENCHMARK_LATEST),
            "skipped": skipped,
        }), 404

    for model_key, item in prepared.items():
        latency_component = item["scenario_latency_p95_ms"] / latency_target_ms
        memory_component = item["peak_rss_mb"] / ram_budget_mb
        cpu_component = item["cpu_avg_percent"] / cpu_target_percent
        runtime_weight_total = latency_weight + memory_weight + cpu_weight
        runtime_only_cost = (
            (latency_weight / runtime_weight_total) * latency_component
            + (memory_weight / runtime_weight_total) * memory_component
            + (cpu_weight / runtime_weight_total) * cpu_component
            if runtime_weight_total > 0
            else float(np.mean([latency_component, memory_component, cpu_component]))
        )
        if item["accuracy_component"] is None:
            combined_cost = runtime_only_cost
        else:
            combined_cost = (
                accuracy_weight * item["accuracy_component"]
                + latency_weight * latency_component
                + memory_weight * memory_component
                + cpu_weight * cpu_component
            )
        item["latency_component"] = latency_component
        item["memory_component"] = memory_component
        item["cpu_component"] = cpu_component
        item["runtime_only_cost"] = runtime_only_cost
        item["combined_cost"] = combined_cost
        item["fit_score"] = 100.0 / (1.0 + 0.30 * max(combined_cost, 0.0))

    runtime_winner = min(prepared, key=lambda key: prepared[key]["runtime_only_cost"])
    learned_candidates = {key: item for key, item in prepared.items() if item["accuracy_component"] is not None}
    learned_winner = (
        min(learned_candidates, key=lambda key: learned_candidates[key]["combined_cost"])
        if learned_candidates
        else None
    )

    response_models = {}
    for model_key, item in prepared.items():
        response_models[model_key] = {
            "display_name": item["display_name"],
            "accuracy_included": item["accuracy_included"],
            "metrics": item["metrics"],
            "model_size_mb": item["model_size_mb"],
            "measured_latency_p50_ms": item["latency_p50_ms"],
            "measured_latency_p95_ms": item["latency_p95_ms"],
            "scenario_latency_p95_ms": item["scenario_latency_p95_ms"],
            "measured_peak_rss_mb": item["peak_rss_mb"],
            "measured_cpu_avg_percent": item["cpu_avg_percent"],
            "measured_cpu_peak_percent": item["cpu_peak_percent"],
            "ram_budget_utilization": item["ram_budget_utilization"],
            "over_budget_ratio": item["over_budget_ratio"],
            "accuracy_component": item["accuracy_component"],
            "latency_component": item["latency_component"],
            "memory_component": item["memory_component"],
            "cpu_component": item["cpu_component"],
            "runtime_only_cost": item["runtime_only_cost"],
            "combined_cost": item["combined_cost"],
            "fit_score": item["fit_score"],
            "note": (
                ("Doğruluk bileşeni hariç runtime maliyetiyle hesaplandı; interpolasyon referans ailedir."
                 if not item["accuracy_included"]
                 else "Gerçek benchmark ölçümünden doğruluk ve runtime bileşenleriyle hesaplandı.")
                + (" CPU hız faktörüyle ölçeklenmiş senaryo gösteriliyor." if abs(cpu_speed_factor - 1.0) > 1e-9 else "")
                + (" RAM bütçesini aşıyor." if item["over_budget_ratio"] > 0 else "")
            ),
        }

    return jsonify({
        "source": "pi_benchmark",
        "benchmark": {
            "path": str(PI_BENCHMARK_LATEST),
            "run_id": benchmark.get("run_id"),
            "created_at": benchmark.get("created_at"),
            "system": benchmark.get("system") or {},
            "config": benchmark.get("config") or {},
            "command": benchmark_command,
        },
        "weights": {
            "accuracy": accuracy_weight,
            "latency": latency_weight,
            "memory": memory_weight,
            "cpu": cpu_weight,
        },
        "hardware": {
            "cpu_speed_factor": cpu_speed_factor,
            "ram_budget_gb": ram_budget_gb,
            "ram_budget_mb": ram_budget_mb,
        },
        "targets": {
            "rmse": accuracy_targets["rmse"],
            "mae": accuracy_targets["mae"],
            "mape": accuracy_targets["mape"],
            "r2_gap": accuracy_targets["r2_gap"],
            "latency_ms": latency_target_ms,
            "memory_mb": ram_budget_mb,
            "cpu_percent": cpu_target_percent,
        },
        "winner": learned_winner or runtime_winner,
        "runtime_winner": runtime_winner,
        "learned_winner": learned_winner,
        "models": response_models,
        "skipped": skipped,
        "note": "Bu panel gerçek benchmark ölçümünden beslenir. Runtime kıyası üç yöntemi kapsar; doğruluk bileşeni yalnız XGBoost ve FT-Transformer için kullanılır.",
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
                env=_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
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
# Dataset production / segmentation internal tool launcher
# ---------------------------------------------------------------------------

@app.route("/api/dataset-tools/status")
def api_dataset_tools_status():
    return jsonify(_dataset_tool_status())


@app.route("/api/dataset-tools/commands")
def api_dataset_tools_commands():
    return jsonify(_public_dataset_commands())


@app.route("/api/dataset-gui/defaults")
def api_dataset_gui_defaults():
    return jsonify(_dataset_gui_defaults())


@app.route("/api/dataset-gui/file")
def api_dataset_gui_file():
    requested = request.args.get("path", "")
    if not requested:
        return jsonify({"error": "path is required"}), 400
    path = _safe_dataset_path(requested)
    if not _path_inside_dataset_tool(path) or not path.exists() or not path.is_file():
        return jsonify({"error": "Dosya bulunamadı veya dataset tool klasörü dışında."}), 404
    return send_file(path)


@app.route("/api/dataset-gui/preview")
def api_dataset_gui_preview():
    kind = request.args.get("kind", "training")
    if kind == "result":
        output_dir = _safe_text(request.args.get("output_dir"), "segmentation_results")
        base = _safe_dataset_path(output_dir)
        files = sorted([p for p in base.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]) if base.exists() else []
        if not files:
            return jsonify({"error": f"Önizlenecek sonuç görseli bulunamadı: {output_dir}"}), 404
        chosen = random.choice(files)
        return jsonify({
            "kind": kind,
            "image": str(chosen.relative_to(DATASET_APP_DIR)),
            "image_url": f"/api/dataset-gui/file?path={chosen.relative_to(DATASET_APP_DIR).as_posix()}",
            "mask_url": "",
            "count": len(files),
        })

    dataset_path = _safe_text(request.args.get("dataset_path"), "dataset_production")
    dataset_root = _safe_dataset_path(dataset_path)
    image_dirs = [
        dataset_root / "images",
        dataset_root / "train",
        dataset_root / "train" / "images",
        dataset_root / "valid",
        dataset_root / "valid" / "images",
        dataset_root / "test",
        dataset_root / "test" / "images",
    ]
    files = []
    image_dir = image_dirs[0]
    for candidate_dir in image_dirs:
        if candidate_dir.exists():
            candidate_files = sorted([p for p in candidate_dir.glob("*.png")])
            if candidate_files:
                files = candidate_files
                image_dir = candidate_dir
                break
    if not files:
        return jsonify({"error": f"Önizlenecek eğitim görseli bulunamadı: {dataset_path}/images veya train/valid/test/images"}), 404
    chosen = random.choice(files)
    split_dir = image_dir.parent if image_dir.name == "images" else image_dir
    mask_dir = split_dir / "masks"
    mask_candidates = [
        mask_dir / chosen.name.replace("img_", "mask_"),
        mask_dir / chosen.name,
    ]
    mask = next((candidate for candidate in mask_candidates if candidate.exists()), None)
    return jsonify({
        "kind": "training",
        "image": str(chosen.relative_to(DATASET_APP_DIR)),
        "image_url": f"/api/dataset-gui/file?path={chosen.relative_to(DATASET_APP_DIR).as_posix()}",
        "mask": str(mask.relative_to(DATASET_APP_DIR)) if mask else "",
        "mask_url": f"/api/dataset-gui/file?path={mask.relative_to(DATASET_APP_DIR).as_posix()}" if mask else "",
        "count": len(files),
    })


@app.route("/api/dataset-gui/mask", methods=["POST"])
def api_dataset_gui_save_mask():
    body = request.get_json(force=True)
    mask_path_value = body.get("mask", "")
    data_url = body.get("data_url", "")
    if not mask_path_value or not data_url:
        return jsonify({"error": "mask ve data_url zorunlu."}), 400

    mask_path = _safe_dataset_path(mask_path_value)
    if not _path_inside_dataset_tool(mask_path):
        return jsonify({"error": "Maske yolu dataset tool klasörü dışında."}), 400
    if mask_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return jsonify({"error": "Maske çıktısı görsel dosyası olmalı."}), 400
    if "," not in data_url:
        return jsonify({"error": "Geçersiz canvas verisi."}), 400

    try:
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
        alpha = image.getchannel("A")
        alpha = alpha.point(lambda value: 255 if value > 20 else 0)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        alpha.save(mask_path)
    except Exception as exc:
        return jsonify({"error": f"Maske kaydedilemedi: {exc}"}), 500

    return jsonify({
        "ok": True,
        "mask": str(mask_path.relative_to(DATASET_APP_DIR)),
        "mask_url": f"/api/dataset-gui/file?path={mask_path.relative_to(DATASET_APP_DIR).as_posix()}",
    })


@app.route("/api/dataset-gui/stop", methods=["POST"])
def api_dataset_gui_stop():
    global _dataset_gui_process
    with _dataset_gui_lock:
        process = _dataset_gui_process
        if process is None or process.poll() is not None:
            _dataset_gui_process = None
            return jsonify({"stopped": False, "message": "Çalışan dataset GUI pipeline yok."})
        process.terminate()
        _dataset_gui_process = None
    return jsonify({"stopped": True, "message": "Dataset GUI pipeline durduruldu."})


@app.route("/api/dataset-gui/run", methods=["POST"])
def api_dataset_gui_run():
    """Run the Flask-port of the Dataset GUI pipeline and stream logs."""
    body = request.get_json(force=True)
    config = body.get("config") or body
    if not DATASET_APP_DIR.exists():
        return jsonify({"error": f"Dataset tool directory not found: {DATASET_APP_DIR}"}), 404

    steps = _dataset_gui_build_steps(config)
    if not steps:
        return jsonify({"error": "Çalıştırılacak aktif dataset GUI adımı seçilmedi."}), 400

    def generate():
        global _dataset_gui_process
        yield f"data: {json.dumps({'type': 'output', 'text': f'Dataset GUI web pipeline başladı. Adım sayısı: {len(steps)}'})}\n\n"
        try:
            for index, step in enumerate(steps, start=1):
                label = step["label"]
                yield f"data: {json.dumps({'type': 'output', 'text': f'\\n### {index}. {label}'})}\n\n"

                if step["kind"] == "message":
                    yield f"data: {json.dumps({'type': 'output', 'text': step['text']})}\n\n"
                    continue

                if step["kind"] == "cleanup":
                    try:
                        cleanup_message = _delete_dataset_subdir(step["path"])
                        yield f"data: {json.dumps({'type': 'output', 'text': cleanup_message})}\n\n"
                    except Exception as exc:
                        yield f"data: {json.dumps({'type': 'output', 'text': f'Temizleme hatası: {exc}'})}\n\n"
                    continue

                command = step["command"]
                yield f"data: {json.dumps({'type': 'output', 'text': '$ ' + ' '.join(command)})}\n\n"
                process = subprocess.Popen(
                    command,
                    cwd=str(DATASET_APP_DIR),
                    env=_subprocess_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                )
                with _dataset_gui_lock:
                    _dataset_gui_process = process

                assert process.stdout is not None
                for line in iter(process.stdout.readline, ""):
                    text = line.rstrip()
                    yield f"data: {json.dumps({'type': 'output', 'text': text})}\n\n"
                    hint = _dataset_tool_hint(text)
                    if hint:
                        yield f"data: {json.dumps({'type': 'output', 'text': f'İpucu: {hint}'})}\n\n"

                process.wait()
                with _dataset_gui_lock:
                    if _dataset_gui_process is process:
                        _dataset_gui_process = None

                if process.returncode != 0:
                    yield f"data: {json.dumps({'type': 'done', 'exit_code': process.returncode})}\n\n"
                    return

            yield f"data: {json.dumps({'type': 'output', 'text': '\\n[TÜM DATASET GUI ADIMLARI TAMAMLANDI]'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'exit_code': 0})}\n\n"
        except Exception as exc:
            with _dataset_gui_lock:
                _dataset_gui_process = None
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/dataset-tools/run", methods=["POST"])
def api_dataset_tools_run():
    """Run a whitelisted internal dataset-tool command and stream logs."""
    body = request.get_json(force=True)
    command_id = str(body.get("id", ""))
    command_map = _dataset_tool_command_map()
    command = command_map.get(command_id)
    if command is None:
        return jsonify({"error": "Unknown dataset tool command"}), 400
    if command.get("requires_winget") and not shutil.which("winget"):
        return jsonify({"error": "Winget bulunamadı. Bu sistem bağımlılığı otomatik kurulamıyor."}), 400
    if not DATASET_APP_DIR.exists():
        return jsonify({"error": f"Dataset tool directory not found: {DATASET_APP_DIR}"}), 404

    def generate():
        display_command = command["display_command"]
        yield f"data: {json.dumps({'type': 'output', 'text': f'$ {display_command}'})}\n\n"
        try:
            if command.get("detached"):
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                subprocess.Popen(
                    command["command"],
                    cwd=str(DATASET_APP_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    shell=False,
                    env=_subprocess_env(),
                    creationflags=creationflags,
                    close_fds=True,
                )
                message = "Komut ayrı pencere/process olarak başlatıldı. Pencere açılmazsa Dataset Tool durum kartlarını ve bağımlılıkları kontrol et."
                yield f"data: {json.dumps({'type': 'output', 'text': message})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'exit_code': 0})}\n\n"
                return

            process = subprocess.Popen(
                command["command"],
                cwd=str(DATASET_APP_DIR),
                env=_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
            )
            assert process.stdout is not None
            output_lines: list[str] = []
            for line in iter(process.stdout.readline, ""):
                text = line.rstrip()
                output_lines.append(text)
                yield f"data: {json.dumps({'type': 'output', 'text': text})}\n\n"
                hint = _dataset_tool_hint(text)
                if hint:
                    yield f"data: {json.dumps({'type': 'output', 'text': f'İpucu: {hint}'})}\n\n"
            process.wait()
            exit_code = process.returncode
            if exit_code != 0 and _is_already_installed_output(command_id, output_lines):
                exit_code = 0
                message = "Kurulum aracı paket zaten kurulu dedi; bu durum hata değil, hazır kabul edildi."
                yield f"data: {json.dumps({'type': 'output', 'text': message})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'exit_code': exit_code})}\n\n"
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
    interpolation_ok = _data_config.one_engine_path.exists() and _data_config.two_engine_path.exists()
    xgb_report = _report_paths("xgboost")["row_level"].exists()
    ft_report = _report_paths("ft_transformer")["row_level"].exists()
    data_ok = _data_config.processed_path.exists()
    return jsonify({
        "data_ready": data_ok,
        "interpolation_ready": interpolation_ok,
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
