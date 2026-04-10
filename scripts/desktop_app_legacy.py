from __future__ import annotations

import os
import sys
from pathlib import Path
from tkinter import StringVar, Tk, messagebox
from tkinter import ttk

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nomogram_report import generate_nomogram_report
from src.inference.predictors import (
    FTTransformerPredictor,
    XGBoostPredictor,
    build_single_row_frame,
    build_test_scenarios,
    load_reference_dataset,
)
from src.utils.config import DataConfig


def report_paths(model_key: str) -> dict[str, Path]:
    data_config = DataConfig()
    base = data_config.xgboost_artifact_dir if model_key == "xgboost" else data_config.ft_transformer_artifact_dir
    report_dir = base / "reports"
    return {
        "row_level": report_dir / f"{model_key}_row_level_comparison.csv",
        "slice_summary": report_dir / f"{model_key}_slice_summary.csv",
        "overall_summary": report_dir / f"{model_key}_overall_summary.csv",
        "slice_plot": report_dir / f"{model_key}_slice_predictions.png",
        "summary_plot": report_dir / f"{model_key}_slice_summary.png",
    }


class SpecificRangeDesktopApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Specific Range Tahmin ve Toplu Karsilastirma")
        self.root.geometry("1400x820")

        self.xgb_predictor: XGBoostPredictor | None = None
        self.ft_predictor: FTTransformerPredictor | None = None
        self.reference_df = load_reference_dataset()
        self.scenarios = build_test_scenarios(pd.DataFrame([{
            "altitude": 11000.0,
            "gross_weight": 30000.0,
            "drag_index": 0.0,
            "mach": 0.22,
            "fuel_flow": 5000.0,
            "engine_type": "one_engine",
            "specific_range": 0.0,
        }]))

        self.report_model_var = StringVar(value="xgboost")
        self.ft_device_var = StringVar(value="cpu")
        self.status_var = StringVar(value="Hazir.")
        self.nomogram_engine_type_var = StringVar(value="one_engine")
        self.nomogram_altitude_var = StringVar(value="5000")
        self.nomogram_weight_var = StringVar(value="50000")
        self.nomogram_path_var = StringVar(value="-")
        self.slice_plot_var = StringVar(value="-")
        self.summary_plot_var = StringVar(value="-")

        self.altitude_var = StringVar(value="11000")
        self.gross_weight_var = StringVar(value="30000")
        self.drag_index_var = StringVar(value="0")
        self.mach_var = StringVar(value="0.22")
        self.fuel_flow_var = StringVar(value="5000")
        self.engine_type_var = StringVar(value="one_engine")

        self.xgb_result_var = StringVar(value="-")
        self.ft_result_var = StringVar(value="-")

        self._build_layout()
        self._load_report()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.LabelFrame(outer, text="Hazir Rapor Goruntuleme", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Rapor modeli").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(top, textvariable=self.report_model_var, values=["xgboost", "ft_transformer"], state="readonly", width=18).grid(
            row=0, column=1, sticky="w", padx=4, pady=4
        )
        ttk.Button(top, text="Raporu Yukle", command=self._load_report).grid(row=0, column=2, sticky="w", padx=4, pady=4)

        summary_frame = ttk.LabelFrame(outer, text="Toplu Ozet", padding=8)
        summary_frame.pack(fill="x", pady=(10, 0))
        self.summary_label = ttk.Label(summary_frame, text="Rapor yuklenmedi.")
        self.summary_label.pack(anchor="w")

        plot_frame = ttk.LabelFrame(outer, text="Grafik Dosyalari", padding=8)
        plot_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(plot_frame, text="Gercek vs Tahmin").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(plot_frame, textvariable=self.slice_plot_var).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(plot_frame, text="Ac", command=lambda: self._open_file(self.slice_plot_var.get())).grid(
            row=0, column=2, sticky="w", padx=4, pady=4
        )
        ttk.Label(plot_frame, text="Slice Ozet Hata").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(plot_frame, textvariable=self.summary_plot_var).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(plot_frame, text="Ac", command=lambda: self._open_file(self.summary_plot_var.get())).grid(
            row=1, column=2, sticky="w", padx=4, pady=4
        )

        nomogram_frame = ttk.LabelFrame(outer, text="Taslak Nomogram Karsilastirmasi", padding=8)
        nomogram_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(nomogram_frame, text="engine_type").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            nomogram_frame,
            textvariable=self.nomogram_engine_type_var,
            values=["one_engine", "two_engine"],
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(nomogram_frame, text="altitude").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(nomogram_frame, textvariable=self.nomogram_altitude_var, width=12).grid(row=0, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(nomogram_frame, text="gross_weight").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        ttk.Entry(nomogram_frame, textvariable=self.nomogram_weight_var, width=12).grid(row=0, column=5, sticky="w", padx=4, pady=4)
        ttk.Button(nomogram_frame, text="Nomogram Uret", command=self._generate_nomogram).grid(row=0, column=6, sticky="w", padx=4, pady=4)
        ttk.Label(nomogram_frame, textvariable=self.nomogram_path_var).grid(row=1, column=0, columnspan=6, sticky="w", padx=4, pady=4)
        ttk.Button(nomogram_frame, text="Ac", command=lambda: self._open_file(self.nomogram_path_var.get())).grid(
            row=1, column=6, sticky="w", padx=4, pady=4
        )

        table_frame = ttk.LabelFrame(outer, text="Tum Satirlarin Karsilastirmasi", padding=8)
        table_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.report_tree = ttk.Treeview(table_frame, show="headings", height=18)
        self.report_tree.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(table_frame, orient="vertical", command=self.report_tree.yview).pack(side="right", fill="y")
        self.report_tree.configure(yscrollcommand=lambda a, b: None)

        single = ttk.LabelFrame(outer, text="Tekil Tahmin", padding=10)
        single.pack(fill="x", pady=(10, 0))

        specs = [
            ("Altitude (ft)", self.altitude_var),
            ("Gross Weight (lb)", self.gross_weight_var),
            ("Drag Index", self.drag_index_var),
            ("Mach", self.mach_var),
            ("Fuel Flow (lb / h)", self.fuel_flow_var),
        ]
        for idx, (label, variable) in enumerate(specs):
            ttk.Label(single, text=label).grid(row=0, column=idx * 2, sticky="w", padx=4, pady=4)
            ttk.Entry(single, textvariable=variable, width=12).grid(row=0, column=idx * 2 + 1, sticky="w", padx=4, pady=4)

        ttk.Label(single, text="Engine Type").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(single, textvariable=self.engine_type_var, values=["one_engine", "two_engine"], state="readonly", width=14).grid(
            row=1, column=1, sticky="w", padx=4, pady=4
        )
        ttk.Label(single, text="FT cihaz").grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Combobox(single, textvariable=self.ft_device_var, values=["cpu", "cuda"], state="readonly", width=12).grid(
            row=1, column=3, sticky="w", padx=4, pady=4
        )
        ttk.Button(single, text="Tekil Tahmini Hesapla", command=self._predict_single).grid(row=1, column=4, sticky="w", padx=4, pady=4)
        ttk.Label(single, text="XGBoost").grid(row=1, column=5, sticky="w", padx=4, pady=4)
        ttk.Label(single, textvariable=self.xgb_result_var).grid(row=1, column=6, sticky="w", padx=4, pady=4)
        ttk.Label(single, text="FT-Transformer").grid(row=1, column=7, sticky="w", padx=4, pady=4)
        ttk.Label(single, textvariable=self.ft_result_var).grid(row=1, column=8, sticky="w", padx=4, pady=4)

        ttk.Label(outer, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(8, 0))

    def _set_tree_data(self, df: pd.DataFrame) -> None:
        self.report_tree.delete(*self.report_tree.get_children())
        self.report_tree["columns"] = list(df.columns)
        for column in df.columns:
            self.report_tree.heading(column, text=column)
            self.report_tree.column(column, width=120, anchor="center")
        for _, row in df.iterrows():
            self.report_tree.insert("", "end", values=list(row))

    def _open_file(self, path_str: str) -> None:
        if not path_str or path_str == "-" or not Path(path_str).exists():
            messagebox.showinfo("Bilgi", "Acilacak dosya bulunamadi.")
            return
        os.startfile(path_str)

    def _load_report(self) -> None:
        paths = report_paths(self.report_model_var.get())
        if not paths["row_level"].exists():
            self.summary_label.config(
                text=(
                    "Rapor bulunamadi. Once su komutu calistir:\n"
                    f"python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model {self.report_model_var.get()}"
                )
            )
            self._set_tree_data(pd.DataFrame())
            self.slice_plot_var.set("-")
            self.summary_plot_var.set("-")
            self.status_var.set("Rapor bulunamadi.")
            return

        row_df = pd.read_csv(paths["row_level"])
        summary_df = pd.read_csv(paths["overall_summary"])
        row = summary_df.iloc[0]
        self.summary_label.config(
            text=(
                f"rows={int(row['rows'])} | mae={row['mae']:.6f} | rmse={row['rmse']:.6f} | "
                f"mape={row['mape']:.4f} | r2={row['r2']:.6f}"
            )
        )
        self._set_tree_data(row_df)
        self.slice_plot_var.set(str(paths["slice_plot"]) if paths["slice_plot"].exists() else "-")
        self.summary_plot_var.set(str(paths["summary_plot"]) if paths["summary_plot"].exists() else "-")
        self.status_var.set(f"Rapor yuklendi: {paths['row_level']}")

    def _predict_single(self) -> None:
        try:
            frame = build_single_row_frame(
                altitude=float(self.altitude_var.get()),
                gross_weight=float(self.gross_weight_var.get()),
                drag_index=float(self.drag_index_var.get()),
                mach=float(self.mach_var.get()),
                fuel_flow=float(self.fuel_flow_var.get()),
                engine_type=self.engine_type_var.get(),
            )
            if self.xgb_predictor is None:
                self.xgb_predictor = XGBoostPredictor.from_artifacts()
            self.xgb_result_var.set(f"{self.xgb_predictor.predict_from_frame(frame):.6f}")
            self.ft_predictor = FTTransformerPredictor.from_artifacts(device=self.ft_device_var.get())
            self.ft_result_var.set(f"{self.ft_predictor.predict_from_frame(frame):.6f}")
            self.status_var.set("Tekil tahmin tamamlandi.")
        except Exception as exc:
            messagebox.showerror("Hata", str(exc))
            self.status_var.set("Tekil tahmin basarisiz.")

    def _generate_nomogram(self) -> None:
        try:
            model_key = self.report_model_var.get()
            if model_key == "xgboost":
                if self.xgb_predictor is None:
                    self.xgb_predictor = XGBoostPredictor.from_artifacts()
                batch_predict_fn = self.xgb_predictor.predict_many_from_frame
                output_dir = DataConfig().xgboost_artifact_dir / "nomogram_reports"
            else:
                self.ft_predictor = FTTransformerPredictor.from_artifacts(device=self.ft_device_var.get())
                batch_predict_fn = self.ft_predictor.predict_many_from_frame
                output_dir = DataConfig().ft_transformer_artifact_dir / "nomogram_reports"

            result = generate_nomogram_report(
                self.reference_df,
                model_name=model_key,
                batch_predict_fn=batch_predict_fn,
                output_dir=output_dir,
                engine_type=self.nomogram_engine_type_var.get(),
                altitude=float(self.nomogram_altitude_var.get()),
                gross_weight=float(self.nomogram_weight_var.get()),
            )
            self.nomogram_path_var.set(str(result.nomogram_png))
            self.status_var.set("Taslak nomogram olusturuldu.")
        except Exception as exc:
            messagebox.showerror("Hata", str(exc))
            self.status_var.set("Taslak nomogram olusturulamadi.")


def main() -> None:
    root = Tk()
    app = SpecificRangeDesktopApp(root)
    root.minsize(1280, 760)
    root.mainloop()


if __name__ == "__main__":
    main()
