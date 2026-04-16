from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def _base_figure(width: float = 12, height: float = 4):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _add_box(ax, xy, wh, text, fc="#eef4ff", ec="#4f7bd9", fontsize=11, rounded=True):
    x, y = xy
    w, h = wh
    if rounded:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
        )
    else:
        patch = Rectangle((x, y), w, h, linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return patch


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.8,
            color="#2f3e5e",
        )
    )


def build_problem_motivation_figure(output_path: Path) -> Path:
    fig, ax = _base_figure(12, 4.2)
    _add_box(
        ax,
        (0.04, 0.2),
        (0.23, 0.6),
        "Performans tablolari\n\n10,000 ft\n15,000 ft\n20,000 ft\n\nYalnizca belirli irtifa seviyeleri var",
        fc="#f7fbff",
        ec="#6e8cb6",
    )
    _add_box(
        ax,
        (0.38, 0.22),
        (0.22, 0.56),
        "Ara deger problemi\n\n11,000 ft gibi bir nokta\n tabloda dogrudan yer almaz\n\nKlasik cozum:\nlookup table + interpolasyon",
        fc="#fff7eb",
        ec="#d49b38",
    )
    _add_box(
        ax,
        (0.72, 0.2),
        (0.23, 0.6),
        "Onerilen yaklasim\n\naltitude\n gross_weight\n drag_index\n mach\n fuel_flow\n engine_type\n\nFT-Transformer -> specific_range",
        fc="#eefaf1",
        ec="#3a9d5d",
    )
    _arrow(ax, (0.27, 0.5), (0.38, 0.5))
    _arrow(ax, (0.60, 0.5), (0.72, 0.5))
    ax.text(
        0.5,
        0.92,
        "Problem Motivasyonu: Ayrik tablolardan surekli tahmin uzayina gecis",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_ft_pipeline_figure(output_path: Path) -> Path:
    fig, ax = _base_figure(13, 4.6)
    _add_box(ax, (0.03, 0.18), (0.16, 0.62), "Girdi ozellikleri\n\naltitude\ngross_weight\ndrag_index\nmach\nfuel_flow\nengine_type", fc="#f7fbff", ec="#6e8cb6")
    _add_box(ax, (0.24, 0.29), (0.15, 0.40), "Feature\nTokenizer", fc="#eef4ff", ec="#4f7bd9", fontsize=12)
    _add_box(ax, (0.45, 0.18), (0.13, 0.62), "Tokenlar\n+\n[CLS]", fc="#ffffff", ec="#8ea1b8", rounded=False)
    _add_box(ax, (0.64, 0.24), (0.16, 0.52), "Transformer\nEncoder\n\nMHSA + FFN\nAdd & Norm", fc="#eef4ff", ec="#4f7bd9", fontsize=12)
    _add_box(ax, (0.86, 0.32), (0.11, 0.36), "Regresyon\nHead", fc="#eefaf1", ec="#3a9d5d", fontsize=12)
    _arrow(ax, (0.19, 0.49), (0.24, 0.49))
    _arrow(ax, (0.39, 0.49), (0.45, 0.49))
    _arrow(ax, (0.58, 0.49), (0.64, 0.49))
    _arrow(ax, (0.80, 0.49), (0.86, 0.49))
    ax.text(0.50, 0.92, "FT-Transformer Veri Akisi", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.50, 0.08, "Tum ozellikler ortak token uzayina tasinir; [CLS] tokeni nihai tahmin icin kullanilir.", ha="center", va="center", fontsize=10.5, color="#334155")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_encoder_block_figure(output_path: Path) -> Path:
    fig, ax = _base_figure(13, 4.4)
    _add_box(ax, (0.03, 0.33), (0.12, 0.34), "Input\nTokens", fc="#f7fbff", ec="#6e8cb6")
    _add_box(ax, (0.20, 0.33), (0.10, 0.34), "Layer\nNorm", fc="#ffffff", ec="#8ea1b8")
    _add_box(ax, (0.35, 0.23), (0.16, 0.54), "MHSA\n\nMulti-Head\nSelf-Attention", fc="#eef4ff", ec="#4f7bd9")
    _add_box(ax, (0.56, 0.33), (0.10, 0.34), "Add &\nNorm", fc="#ffffff", ec="#8ea1b8")
    _add_box(ax, (0.71, 0.23), (0.16, 0.54), "FFN\n\nLinear -> ReLU\n-> Linear", fc="#fff7eb", ec="#d49b38")
    _add_box(ax, (0.91, 0.33), (0.08, 0.34), "Add &\nNorm", fc="#ffffff", ec="#8ea1b8")
    _arrow(ax, (0.15, 0.50), (0.20, 0.50))
    _arrow(ax, (0.30, 0.50), (0.35, 0.50))
    _arrow(ax, (0.51, 0.50), (0.56, 0.50))
    _arrow(ax, (0.66, 0.50), (0.71, 0.50))
    _arrow(ax, (0.87, 0.50), (0.91, 0.50))
    ax.plot([0.30, 0.56], [0.80, 0.80], color="#6b7280", linewidth=1.4, linestyle="--")
    ax.plot([0.30, 0.30], [0.80, 0.60], color="#6b7280", linewidth=1.4, linestyle="--")
    ax.plot([0.56, 0.56], [0.80, 0.60], color="#6b7280", linewidth=1.4, linestyle="--")
    ax.text(0.43, 0.84, "Residual baglanti", fontsize=9.5, ha="center", color="#475569")
    ax.plot([0.66, 0.91], [0.14, 0.14], color="#6b7280", linewidth=1.4, linestyle="--")
    ax.plot([0.66, 0.14], [0.14, 0.40], color="#6b7280", linewidth=0)
    ax.plot([0.66, 0.66], [0.14, 0.33], color="#6b7280", linewidth=1.4, linestyle="--")
    ax.plot([0.91, 0.91], [0.14, 0.33], color="#6b7280", linewidth=1.4, linestyle="--")
    ax.text(0.785, 0.05, "Residual baglanti", fontsize=9.5, ha="center", color="#475569")
    ax.text(0.50, 0.93, "Transformer Encoder Blo\u011fu", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.50, 0.10, "Pre-Norm akisinda tokenlar once normalize edilir; MHSA ve FFN sonrasinda residual toplama yapilir.", ha="center", va="center", fontsize=10.2, color="#334155")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_comparison_protocol_figure(output_path: Path) -> Path:
    fig, ax = _base_figure(13, 4.4)
    _add_box(ax, (0.04, 0.26), (0.16, 0.48), "Ham veri\n\nOne Engine\nTwo Engine", fc="#f7fbff", ec="#6e8cb6")
    _add_box(ax, (0.27, 0.26), (0.17, 0.48), "On isleme\nve Split\n\nTrain / Val / Test", fc="#eef4ff", ec="#4f7bd9")
    _add_box(ax, (0.54, 0.58), (0.15, 0.20), "XGBoost", fc="#eefaf1", ec="#3a9d5d")
    _add_box(ax, (0.54, 0.22), (0.15, 0.20), "FT-\nTransformer", fc="#eefaf1", ec="#3a9d5d")
    _add_box(ax, (0.79, 0.26), (0.17, 0.48), "Ayni metrikler\n\nRMSE\nMAE\nR2\nMAPE\nLatency\nSize", fc="#fff7eb", ec="#d49b38")
    _arrow(ax, (0.20, 0.50), (0.27, 0.50))
    _arrow(ax, (0.44, 0.62), (0.54, 0.68))
    _arrow(ax, (0.44, 0.38), (0.54, 0.32))
    _arrow(ax, (0.69, 0.68), (0.79, 0.56))
    _arrow(ax, (0.69, 0.32), (0.79, 0.44))
    ax.text(0.50, 0.92, "Adil Karsilastirma Protokol\u00fc", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.50, 0.08, "Iki model de ayni veri bolmesi, ayni on isleme ve ayni degerlendirme proseduru ile kiyaslanir.", ha="center", va="center", fontsize=10.2, color="#334155")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_altitude_distribution_figure(root: Path, output_path: Path) -> Path:
    dataset_path = root / "data" / "processed" / "combined_specific_range.csv"
    df = pd.read_csv(dataset_path)
    counts = (
        df.groupby(["altitude", "engine_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    altitudes = counts.index.astype(int).tolist()
    one_engine = counts.get("one_engine", pd.Series(index=counts.index, data=0)).tolist()
    two_engine = counts.get("two_engine", pd.Series(index=counts.index, data=0)).tolist()

    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=180)
    ax.bar(altitudes, one_engine, width=3500, label="one_engine", color="#84a9ff", edgecolor="#4f7bd9")
    ax.bar(altitudes, two_engine, width=3500, bottom=one_engine, label="two_engine", color="#97d8a5", edgecolor="#3a9d5d")

    ax.set_title("Irtifa seviyelerine gore ornek dagilimi", fontsize=14, fontweight="bold")
    ax.set_xlabel("Altitude (ft)")
    ax.set_ylabel("Ornek sayisi")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_xticks(altitudes)
    ax.set_xticklabels([f"{alt:,}" for alt in altitudes], rotation=30, ha="right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def ensure_report_figures(root: Path) -> dict[str, Path]:
    output_dir = root / "artifacts" / "report_figures"
    figures = {
        "motivation": build_problem_motivation_figure(output_dir / "report_problem_motivation.png"),
        "ft_pipeline": build_ft_pipeline_figure(output_dir / "report_ft_pipeline.png"),
        "encoder_block": build_encoder_block_figure(output_dir / "report_encoder_block.png"),
        "comparison_protocol": build_comparison_protocol_figure(output_dir / "report_comparison_protocol.png"),
        "altitude_distribution": build_altitude_distribution_figure(root, output_dir / "report_altitude_distribution.png"),
    }
    return figures
