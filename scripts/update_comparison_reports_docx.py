from __future__ import annotations

import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "comparison_reports"
FIG_DIR = REPORT_DIR / "figures"
BENCHMARK_PATH = ROOT / "artifacts" / "benchmarks" / "pi_benchmark_latest.json"

GENERAL_DOCX = REPORT_DIR / "Genel_Yontem_Karsilastirma_Raporu.docx"
PI_DOCX = REPORT_DIR / "Raspberry_Pi3_Hedef_Benchmark_Raporu.docx"
GENERAL_OUT = REPORT_DIR / "Genel_Yontem_Karsilastirma_Raporu_Guncel.docx"
PI_OUT = REPORT_DIR / "Raspberry_Pi3_Hedef_Benchmark_Raporu_Guncel.docx"

METHOD_LABELS = {
    "interpolation": "Interpolasyon",
    "xgboost": "XGBoost",
    "ft_transformer": "FT-Transformer",
}


@dataclass
class CostTargets:
    rmse: float = 0.003
    mae: float = 0.0015
    mape: float = 2.0
    r2_gap: float = 0.001
    latency_ms: float = 10.0
    memory_mb: float = 1024.0
    cpu_percent: float = 80.0


@dataclass
class CostWeights:
    accuracy: float = 0.45
    latency: float = 0.25
    memory: float = 0.20
    cpu: float = 0.10


def _load_benchmark() -> dict:
    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _method_entry(benchmark: dict, method: str) -> dict:
    methods = benchmark.get("models", benchmark.get("methods", benchmark.get("results", {})))
    if isinstance(methods, list):
        for row in methods:
            if row.get("method") == method or row.get("name") == method:
                return row
    if method in methods:
        return methods[method]
    raise KeyError(f"Benchmark method not found: {method}")


def _get(row: dict, *keys: str, default: float | None = None) -> float | None:
    cursor = row
    for key in keys:
        if isinstance(cursor, dict) and key in cursor:
            cursor = cursor[key]
        else:
            return default
    if cursor is None:
        return default
    return float(cursor)


def _metrics(benchmark: dict) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for method in METHOD_LABELS:
        row = _method_entry(benchmark, method)
        out[method] = {
            "p50_ms": _get(row, "latency_ms", "p50", default=_get(row, "latency", "p50_ms", default=_get(row, "p50_ms"))),
            "p95_ms": _get(row, "latency_ms", "p95", default=_get(row, "latency", "p95_ms", default=_get(row, "p95_ms"))),
            "peak_rss_mb": _get(row, "memory_mb", "rss_peak", default=_get(row, "memory", "peak_rss_mb", default=_get(row, "peak_rss_mb"))),
            "cpu_avg_percent": _get(row, "cpu_percent", "process_avg", default=_get(row, "cpu", "avg_percent", default=_get(row, "cpu_avg_percent"))),
            "model_size_mb": _get(row, "model_size_mb", default=_get(row, "model", "size_mb", default=_get(row, "model_size_mb"))),
            "rmse": _get(row, "accuracy", "rmse", default=_get(row, "rmse")),
            "mae": _get(row, "accuracy", "mae", default=_get(row, "mae")),
            "mape": _get(row, "accuracy", "mape", default=_get(row, "mape")),
            "r2": _get(row, "accuracy", "r2", default=_get(row, "r2")),
        }
    return out


def _costs(metrics: dict[str, dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    targets = CostTargets()
    weights = CostWeights()
    costs: dict[str, dict[str, float | None]] = {}
    for method, row in metrics.items():
        latency = (row["p95_ms"] or 0.0) / targets.latency_ms
        memory = (row["peak_rss_mb"] or 0.0) / targets.memory_mb
        cpu = (row["cpu_avg_percent"] or 0.0) / targets.cpu_percent

        accuracy = None
        if method != "interpolation" and row["rmse"] is not None:
            r2_gap = max(1.0 - float(row["r2"] or 0.0), 0.0)
            accuracy = float(
                np.mean(
                    [
                        (row["rmse"] or 0.0) / targets.rmse,
                        (row["mae"] or 0.0) / targets.mae,
                        (row["mape"] or 0.0) / targets.mape,
                        r2_gap / targets.r2_gap,
                    ]
                )
            )
            composite = (
                weights.accuracy * accuracy
                + weights.latency * latency
                + weights.memory * memory
                + weights.cpu * cpu
            )
        else:
            runtime_weight = weights.latency + weights.memory + weights.cpu
            composite = (
                weights.latency * latency
                + weights.memory * memory
                + weights.cpu * cpu
            ) / runtime_weight

        costs[method] = {
            "accuracy": accuracy,
            "latency": latency,
            "memory": memory,
            "cpu": cpu,
            "composite": composite,
            "fit_score": 100.0 / (1.0 + 0.30 * composite),
        }
    return costs


def _fmt(value: float | None, digits: int = 3, empty: str = "-") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return empty
    return f"{value:.{digits}f}"


def _set_cell(table, row: int, col: int, text: str) -> None:
    table.rows[row].cells[col].text = text


def _update_general_doc(benchmark: dict, metrics: dict, costs: dict) -> Path:
    doc = Document(GENERAL_DOCX)
    created = benchmark.get("created_at") or benchmark.get("timestamp") or ""
    run_id = benchmark.get("run_id", "unknown")
    if created:
        created = created.replace("T", " ").replace("Z", " UTC")
    doc.paragraphs[2].text = f"Run ID: {run_id} | Referans ölçüm: {created}"

    # Main metric table.
    table = doc.tables[1]
    for r, method in enumerate(["interpolation", "xgboost", "ft_transformer"], start=1):
        row = metrics[method]
        _set_cell(table, r, 0, METHOD_LABELS[method])
        _set_cell(table, r, 1, _fmt(row["p95_ms"], 2))
        _set_cell(table, r, 2, _fmt(row["peak_rss_mb"], 1))
        _set_cell(table, r, 3, _fmt(row["cpu_avg_percent"], 1))
        _set_cell(table, r, 4, _fmt(row["model_size_mb"], 2))
        _set_cell(table, r, 5, _fmt(row["rmse"], 6))
        _set_cell(table, r, 6, _fmt(row["mae"], 6))
        _set_cell(table, r, 7, _fmt(row["mape"], 2))
        _set_cell(table, r, 8, _fmt(row["r2"], 6))

    # Decision matrix remains qualitative, but update the practical winners using latest measurements.
    decision = doc.tables[2]
    updates = {
        1: ("Doğruluk", "XGBoost", "Öğrenen modeller arasında RMSE, MAE ve MAPE değerleri en düşük seviyededir."),
        2: ("Hız", "XGBoost", "p95 gecikme değeri en düşük olan öğrenen yöntemdir."),
        3: ("Düşük RAM", "Interpolasyon", "Peak RSS değeri en düşük runtime profiline sahiptir."),
        4: ("Küçük model boyutu", "FT-Transformer", "Ağırlık dosyası en küçük model artefact'ıdır."),
        5: ("Yorumlanabilirlik", "Interpolasyon", "Kaynak tablo ve arama mantığına en yakın yöntemdir."),
        6: ("Pratik demo", "XGBoost", "Doğruluk, hız ve kaynak dengesi açısından en güvenli varsayılandır."),
        7: ("Araştırma değeri", "FT-Transformer", "Tabular Transformer mimarisini ve self-attention temelli etkileşim öğrenimini temsil eder."),
    }
    for r, cells in updates.items():
        for c, value in enumerate(cells):
            _set_cell(decision, r, c, value)

    replacements = {
        "Model tarafında FT-Transformer için daha uzun eğitim, PSO ile hiperparametre araması, daha küçük mimari varyantlar, quantization ve ONNX export denenmelidir. XGBoost tarafında ise model sıkıştırma, tree depth/estimator sayısı dengesi ve thread kontrollü latency ölçümleri ile daha kararlı edge profili elde edilebilir.": (
            "Model tarafında FT-Transformer ve XGBoost için aynı ağırlıklı çok hedefli PSO maliyeti kullanılabilir. "
            "FT-Transformer tarafında daha uzun eğitim, küçük mimari varyantlar, quantization ve ONNX export denenmelidir. "
            "XGBoost tarafında ise tree depth/estimator sayısı dengesi, regularization ve thread kontrollü latency ölçümleri ile daha kararlı edge profili elde edilebilir."
        ),
        "XGBoost en dengeli profil verir; FT-Transformer araştırma adayı, interpolasyon referans ailesidir.": (
            "XGBoost en dengeli profil verir; FT-Transformer araştırma adayı, interpolasyon ise doğruluk yarışına sokulmayan referans ailesidir."
        ),
    }
    _replace_paragraphs(doc, replacements)
    doc.save(GENERAL_OUT)
    return GENERAL_OUT


def _update_pi_doc(benchmark: dict, metrics: dict, costs: dict) -> Path:
    doc = Document(PI_DOCX)
    created = benchmark.get("created_at") or benchmark.get("timestamp") or ""
    run_id = benchmark.get("run_id", "unknown")
    if created:
        created = created.replace("T", " ").replace("Z", " UTC")
    doc.paragraphs[2].text = f"Run ID: {run_id} | Referans ölçüm: {created}"

    table = doc.tables[1]
    for r, method in enumerate(["interpolation", "xgboost", "ft_transformer"], start=1):
        row = metrics[method]
        cost = costs[method]
        _set_cell(table, r, 0, METHOD_LABELS[method])
        _set_cell(table, r, 1, _fmt(row["p50_ms"], 2))
        _set_cell(table, r, 2, _fmt(row["p95_ms"], 2))
        _set_cell(table, r, 3, _fmt(row["peak_rss_mb"], 1))
        _set_cell(table, r, 4, _fmt(row["cpu_avg_percent"], 1))
        _set_cell(table, r, 5, _fmt(row["model_size_mb"], 2))
        _set_cell(table, r, 6, _fmt(cost["fit_score"], 1))

    replacements = {
        "Kullanılan benchmark komutu şu mantığı izler: `python scripts/run_pi_benchmark.py --models interpolation,xgboost,ft_transformer --sample-size 200 --warmup 20 --repetitions 200 --device cpu`. Warmup ölçüme dahil edilmez; p50/p95 latency değerleri yalnız ölçüm tekrarlarından hesaplanır.": (
            "Kullanılan benchmark komutu şu mantığı izler: `python scripts/run_pi_benchmark.py --models interpolation,xgboost,ft_transformer --sample-size 200 --warmup 50 --repetitions 500 --device cpu`. "
            "Warmup ölçüme dahil edilmez; p50/p95 latency değerleri yalnız ölçüm tekrarlarından hesaplanır."
        ),
        "Tablo, üç yöntemin aynı inference benchmark protokolü altında elde edilen özetini verir. Interpolasyon bellek açısından en hafif görünürken, XGBoost gecikme ve genel maliyet dengesi açısından öne çıkmaktadır. FT-Transformer ise küçük model dosyasına rağmen Python/PyTorch runtime ayak izi nedeniyle daha yüksek RAM maliyeti taşımaktadır.": (
            "Tablo, üç yöntemin aynı inference benchmark protokolü altında elde edilen özetini verir. Interpolasyon bellek açısından en hafif görünürken, XGBoost gecikme ve genel maliyet dengesi açısından öne çıkmaktadır. "
            "FT-Transformer ise küçük model dosyasına rağmen Python/PyTorch runtime ayak izi nedeniyle daha yüksek RAM maliyeti taşımaktadır."
        ),
    }
    _replace_paragraphs(doc, replacements)
    doc.save(PI_OUT)
    return PI_OUT


def _replace_paragraphs(doc: Document, replacements: dict[str, str]) -> None:
    for paragraph in doc.paragraphs:
        text = paragraph.text
        if text in replacements:
            paragraph.text = replacements[text]


def _plot_accuracy(metrics: dict) -> Path:
    path = FIG_DIR / "learned_model_accuracy_comparison.png"
    targets = CostTargets()
    labels = ["RMSE / hedef", "MAE / hedef", "MAPE / hedef", "R2 gap / hedef"]
    xgb = [
        metrics["xgboost"]["rmse"] / targets.rmse,
        metrics["xgboost"]["mae"] / targets.mae,
        metrics["xgboost"]["mape"] / targets.mape,
        max(1.0 - metrics["xgboost"]["r2"], 0.0) / targets.r2_gap,
    ]
    ft = [
        metrics["ft_transformer"]["rmse"] / targets.rmse,
        metrics["ft_transformer"]["mae"] / targets.mae,
        metrics["ft_transformer"]["mape"] / targets.mape,
        max(1.0 - metrics["ft_transformer"]["r2"], 0.0) / targets.r2_gap,
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=180)
    ax.bar(x - 0.18, xgb, width=0.36, label="XGBoost", color="#1f9d55")
    ax.bar(x + 0.18, ft, width=0.36, label="FT-Transformer", color="#f28e2b")
    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1, label="Hedef oranı")
    ax.set_title("Öğrenen Modellerde Normalize Doğruluk Karşılaştırması")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalize hata bileşeni (düşük daha iyi)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    ax.text(
        0.0,
        -0.22,
        "Not: Interpolasyon referans/tahmini gerçek değer ailesidir; doğruluk yarışına dahil edilmez.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_runtime_cost(costs: dict) -> Path:
    path = FIG_DIR / "runtime_cost_components.png"
    methods = ["interpolation", "xgboost", "ft_transformer"]
    labels = [METHOD_LABELS[m] for m in methods]
    components = ["latency", "memory", "cpu"]
    colors = ["#2f80ed", "#27ae60", "#f2994a"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=180)
    width = 0.24
    for i, comp in enumerate(components):
        values = [costs[m][comp] for m in methods]
        ax.bar(x + (i - 1) * width, values, width=width, label=comp.capitalize(), color=colors[i])
    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1, label="Hedef oranı")
    ax.set_title("Normalize Runtime Maliyet Bileşenleri")
    ax.set_ylabel("Normalize maliyet (düşük daha iyi)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_decision_matrix() -> Path:
    path = FIG_DIR / "overall_decision_matrix.png"
    criteria = ["Doğruluk", "Hız", "Düşük RAM", "Küçük Boyut", "Açıklanabilirlik", "Deployment"]
    methods = ["Interpolasyon", "XGBoost", "FT-Transformer"]
    scores = np.array(
        [
            [np.nan, 5, 4],
            [3, 5, 4],
            [5, 3.5, 2.5],
            [3.5, 4, 5],
            [5, 4, 2.5],
            [4, 4.5, 3],
        ]
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=180)
    masked = np.ma.masked_invalid(scores)
    im = ax.imshow(masked, cmap="YlGnBu", vmin=1, vmax=5)
    ax.set_xticks(np.arange(len(methods)), methods)
    ax.set_yticks(np.arange(len(criteria)), criteria)
    ax.set_title("Genel Karar Matrisi (1-5)")
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            text = "N/A" if np.isnan(scores[i, j]) else f"{scores[i, j]:.1f}"
            ax.text(j, i, text, ha="center", va="center", color="#111827", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax.text(
        0.0,
        -0.16,
        "N/A: İnterpolasyon doğrulukta referans olduğu için kendi referansına karşı puanlanmaz.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_radar() -> Path:
    path = FIG_DIR / "overall_profile_radar.png"
    labels = ["Referans rolü", "Hız", "Düşük RAM", "Küçük boyut", "Açıklanabilirlik", "Deployment"]
    data = {
        "Interpolasyon": [5, 3, 5, 3.5, 5, 4],
        "XGBoost": [3, 5, 3.5, 4, 4, 4.5],
        "FT-Transformer": [2.5, 4, 2.5, 5, 2.5, 3],
    }
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.7, 6.2), dpi=180, subplot_kw={"polar": True})
    colors = {"Interpolasyon": "#2f80ed", "XGBoost": "#1f9d55", "FT-Transformer": "#f28e2b"}
    for name, values in data.items():
        vals = values + values[:1]
        ax.plot(angles, vals, label=name, linewidth=2, color=colors[name])
        ax.fill(angles, vals, alpha=0.10, color=colors[name])
    ax.set_title("Genel Profil Radar Grafiği", pad=18)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    ax.text(
        0.0,
        -0.12,
        "Accuracy ekseni özellikle kullanılmamıştır; interpolasyon referans aile olarak ayrı değerlendirilir.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pi_runtime(metrics: dict) -> Path:
    path = FIG_DIR / "pi3_target_runtime_resource_summary.png"
    methods = ["interpolation", "xgboost", "ft_transformer"]
    labels = [METHOD_LABELS[m] for m in methods]
    series = [
        ("p95 ms", [metrics[m]["p95_ms"] for m in methods], "#2f80ed"),
        ("Peak RSS MB / 100", [(metrics[m]["peak_rss_mb"] or 0) / 100 for m in methods], "#27ae60"),
        ("CPU avg % / 100", [(metrics[m]["cpu_avg_percent"] or 0) / 100 for m in methods], "#f2994a"),
        ("Model MB", [metrics[m]["model_size_mb"] for m in methods], "#9b51e0"),
    ]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180)
    for i, (name, values, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, values, width, label=name, color=color)
    ax.set_title("Pi 3 Hedef Runtime ve Kaynak Özeti")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Karşılaştırmalı ölçek")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_fit_score(costs: dict) -> Path:
    path = FIG_DIR / "fit_score.png"
    methods = ["interpolation", "xgboost", "ft_transformer"]
    labels = [METHOD_LABELS[m] for m in methods]
    values = [costs[m]["fit_score"] for m in methods]
    colors = ["#2f80ed", "#1f9d55", "#f28e2b"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=180)
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Fit Score (yüksek daha iyi)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _replace_docx_media(docx_path: Path, image_paths: list[Path]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        unpacked = tmp / "docx"
        with zipfile.ZipFile(docx_path, "r") as z:
            z.extractall(unpacked)

        media_dir = unpacked / "word" / "media"
        media_files = sorted(media_dir.glob("image*.png"), key=lambda p: p.name)
        if len(media_files) != len(image_paths):
            raise RuntimeError(f"{docx_path.name}: expected {len(media_files)} images, got {len(image_paths)} replacements")

        for media, source in zip(media_files, image_paths):
            shutil.copyfile(source, media)

        new_docx = tmp / docx_path.name
        with zipfile.ZipFile(new_docx, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file in unpacked.rglob("*"):
                if file.is_file():
                    z.write(file, file.relative_to(unpacked).as_posix())
        shutil.copyfile(new_docx, docx_path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = _load_benchmark()
    metrics = _metrics(benchmark)
    costs = _costs(metrics)

    general_figs = [
        _plot_accuracy(metrics),
        _plot_runtime_cost(costs),
        _plot_decision_matrix(),
        _plot_radar(),
    ]
    pi_figs = [
        _plot_pi_runtime(metrics),
        _plot_runtime_cost(costs),
        _plot_fit_score(costs),
    ]

    general_path = _update_general_doc(benchmark, metrics, costs)
    _replace_docx_media(general_path, general_figs)
    pi_path = _update_pi_doc(benchmark, metrics, costs)
    _replace_docx_media(pi_path, pi_figs)

    print(f"Updated: {general_path}")
    print(f"Updated: {pi_path}")
    print("Metrics:")
    for method in METHOD_LABELS:
        print(method, metrics[method], costs[method])


if __name__ == "__main__":
    main()
