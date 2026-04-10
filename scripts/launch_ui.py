from __future__ import annotations

import base64
import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from src.utils.config import DataConfig


st.set_page_config(page_title="Specific Range Tahmin ve Raporlama", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_xgboost_predictor() -> XGBoostPredictor:
    return XGBoostPredictor.from_artifacts()


@st.cache_resource
def load_ft_predictor(device: str) -> FTTransformerPredictor:
    return FTTransformerPredictor.from_artifacts(device=device)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return add_reference_row_id(load_reference_dataset())


@st.cache_data
def load_test_scenarios():
    return build_test_scenarios(load_reference_dataset())


def _report_paths(model_key: str) -> dict[str, Path]:
    data_config = DataConfig()
    base = data_config.xgboost_artifact_dir if model_key == "xgboost" else data_config.ft_transformer_artifact_dir
    report_dir = base / "reports"
    safe_name = model_key
    return {
        "row_level": report_dir / f"{safe_name}_row_level_comparison.csv",
        "slice_summary": report_dir / f"{safe_name}_slice_summary.csv",
        "overall_summary": report_dir / f"{safe_name}_overall_summary.csv",
        "excel_report": report_dir / f"{safe_name}_table_report.xlsx",
        "slice_plot": report_dir / f"{safe_name}_slice_predictions.png",
        "summary_plot": report_dir / f"{safe_name}_slice_summary.png",
    }


def _nomogram_dir(model_key: str) -> Path:
    data_config = DataConfig()
    base = data_config.xgboost_artifact_dir if model_key == "xgboost" else data_config.ft_transformer_artifact_dir
    return base / "nomogram_reports"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_executable() -> str:
    return sys.executable


def _format_shell_command(parts: list[str]) -> str:
    return " ".join(parts)


def _run_setup_command(command_parts: list[str], label: str) -> tuple[bool, str]:
    completed = subprocess.run(
        command_parts,
        cwd=_workspace_root(),
        capture_output=True,
        text=True,
    )
    output_chunks = [f"$ {_format_shell_command(command_parts)}"]
    if completed.stdout:
        output_chunks.append(completed.stdout.strip())
    if completed.stderr:
        output_chunks.append(completed.stderr.strip())
    output = "\n\n".join(chunk for chunk in output_chunks if chunk)
    return completed.returncode == 0, f"[{label}]\n{output}"


def _load_display_image(path: Path, max_width: int = 1400) -> Image.Image | None:
    if not path.exists():
        return None

    image = Image.open(path)
    if image.width <= max_width:
        return image

    ratio = max_width / float(image.width)
    resized_height = int(image.height * ratio)
    return image.resize((max_width, resized_height))


def _render_clickable_image_preview(path: Path, caption: str, key: str, max_width: int = 1400) -> None:
    if not path.exists():
        st.info("Gorsel bulunamadi.")
        return

    with path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    data_url = f"data:image/png;base64,{encoded}"
    modal_id = f"modal_{key}"
    st.markdown(
        f"""
        <style>
        #{modal_id} {{
          display: none;
          position: fixed;
          z-index: 99999;
          inset: 0;
          background: rgba(15, 23, 42, 0.92);
          padding: 1.5rem;
          box-sizing: border-box;
        }}
        #{modal_id}:target {{
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        #{modal_id} .modal-inner {{
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }}
        #{modal_id} .modal-top {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          color: white;
          font-size: 0.95rem;
        }}
        #{modal_id} .modal-close {{
          color: white;
          text-decoration: none;
          font-weight: 700;
          border: 1px solid rgba(255,255,255,0.25);
          padding: 0.35rem 0.75rem;
          border-radius: 999px;
        }}
        #{modal_id} .modal-image-wrap {{
          flex: 1;
          overflow: auto;
          background: rgba(2, 6, 23, 0.45);
          border-radius: 16px;
          padding: 1rem;
        }}
        #{modal_id} .modal-image {{
          display: block;
          max-width: none;
          width: auto;
          height: auto;
          margin: 0 auto;
        }}
        </style>
        <div style="margin-bottom: 0.5rem;">
          <a href="#{modal_id}" title="Buyuk gormek icin tikla">
            <img
              src="{data_url}"
              alt="{caption}"
              style="
                width: 100%;
                border-radius: 12px;
                border: 1px solid rgba(148, 163, 184, 0.25);
                cursor: zoom-in;
                display: block;
              "
            />
          </a>
        </div>
        <div id="{modal_id}">
          <div class="modal-inner">
            <div class="modal-top">
              <div>{caption}</div>
              <a class="modal-close" href="#">Kapat</a>
            </div>
            <div class="modal-image-wrap">
              <img class="modal-image" src="{data_url}" alt="{caption}" />
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Onizlemeye tikla: gorsel sayfa icinde buyuk onizleme olarak acilir.")


def _setup_command_specs(python_exec: str, dataset_path: str, xgb_device: str, ft_device: str) -> dict[str, dict[str, object]]:
    return {
        "Veri pipeline": {
            "command": [python_exec, "scripts/run_data_pipeline.py"],
            "eta": "yaklasik 10-30 sn",
            "artifacts": ["data/processed/combined_specific_range.csv"],
        },
        "XGBoost egit + rapor": {
            "command": [
                python_exec,
                "scripts/train_xgboost.py",
                "--dataset",
                dataset_path,
                "--device",
                xgb_device,
                "--run-table-report",
            ],
            "eta": "yaklasik 30 sn - 3 dk",
            "artifacts": [
                "artifacts/xgboost/model.json",
                "artifacts/xgboost/preprocessor.joblib",
                "artifacts/xgboost/reports/*.csv|*.xlsx|*.png",
            ],
        },
        "FT-Transformer egit + rapor": {
            "command": [
                python_exec,
                "scripts/train_ft_transformer.py",
                "--dataset",
                dataset_path,
                "--device",
                ft_device,
                "--run-table-report",
            ],
            "eta": "yaklasik 2-15 dk",
            "artifacts": [
                "artifacts/ft_transformer/model.pt",
                "artifacts/ft_transformer/preprocessor.joblib",
                "artifacts/ft_transformer/reports/*.csv|*.xlsx|*.png",
            ],
        },
        "Toplu rapor uret": {
            "command": [
                python_exec,
                "scripts/run_table_report.py",
                "--dataset",
                dataset_path,
                "--model",
                "both",
                "--ft-device",
                ft_device,
            ],
            "eta": "yaklasik 20 sn - 3 dk",
            "artifacts": [
                "artifacts/xgboost/reports/*.csv|*.xlsx|*.png",
                "artifacts/ft_transformer/reports/*.csv|*.xlsx|*.png",
            ],
        },
        "PSO calistir": {
            "command": [
                python_exec,
                "scripts/run_pso.py",
                "--dataset",
                dataset_path,
            ],
            "eta": "yaklasik 5-30 dk",
            "artifacts": ["artifacts/*/ metrics and optimization outputs"],
        },
        "Model karsilastir": {
            "command": [
                python_exec,
                "scripts/compare_models.py",
                "--dataset",
                dataset_path,
            ],
            "eta": "yaklasik 1-5 dk",
            "artifacts": ["terminal output comparison summary"],
        },
    }


def _render_report_viewer() -> None:
    st.subheader("Hazir Toplu Karsilastirma Raporu")
    st.caption(
        "Burada egitim sonrasi uretilmis satir-bazli raporlar goruntulenir. "
        "Yani Excel icindeki tum satirlar tek tek modele verilmis, tahmin edilmis ve hata hesaplanmistir."
    )

    report_model = st.sidebar.selectbox("Rapor modeli", ["xgboost", "ft_transformer"], index=0)
    paths = _report_paths(report_model)
    reference_df = load_reference_data()

    if not paths["row_level"].exists():
        st.warning(
            "Rapor bulunamadi. Once su komutlardan birini calistir:\n"
            f"`python scripts/train_{'xgboost' if report_model == 'xgboost' else 'ft_transformer'}.py --dataset data/processed/combined_specific_range.csv --device cpu --run-table-report`\n"
            f"veya\n`python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model {report_model}`"
        )
        return

    overall_df = pd.read_csv(paths["overall_summary"])
    slice_df = pd.read_csv(paths["slice_summary"])
    row_df = pd.read_csv(paths["row_level"])

    st.success(f"Rapor yuklendi: `{paths['row_level']}`")
    if paths["excel_report"].exists():
        st.caption(f"Excel raporu: `{paths['excel_report']}`")

    with st.expander("Onerilen Akis", expanded=False):
        st.write("1. `Setup` ekranindan veri pipeline ve egitim adimlarini tamamla.")
        st.write("2. `Hazir rapor` ekraninda once genel metrikleri ve slice ozetini incele.")
        st.write("3. Sonra filtrelerle belirli `engine_type` / `altitude` / hata bandlarini daralt.")
        st.write("4. Gerekirse `Tekil Tahmin (Custom Input)` ekraninda ara degerleri dene.")

    if report_model == "xgboost":
        preferred = [
            "row_id",
            "engine_type",
            "altitude",
            "gross_weight",
            "drag_index",
            "mach",
            "fuel_flow",
            "actual_specific_range",
            "xgboost_predicted",
            "xgboost_absolute_error",
            "xgboost_signed_error",
        ]
    else:
        preferred = [
            "row_id",
            "engine_type",
            "altitude",
            "gross_weight",
            "drag_index",
            "mach",
            "fuel_flow",
            "actual_specific_range",
            "ft_transformer_predicted",
            "ft_transformer_absolute_error",
            "ft_transformer_signed_error",
        ]

    available = [column for column in preferred if column in row_df.columns]

    tab_ozet, tab_satirlar, tab_grafikler, tab_nomogram = st.tabs(
        ["Ozet", "Satir Bazli Karsilastirma", "Grafikler", "Taslak Nomogram"]
    )

    with tab_ozet:
        summary_cols = st.columns(5)
        row = overall_df.iloc[0]
        summary_cols[0].metric("Rows", int(row["rows"]))
        summary_cols[1].metric("MAE", f"{row['mae']:.6f}")
        summary_cols[2].metric("RMSE", f"{row['rmse']:.6f}")
        summary_cols[3].metric("MAPE", f"{row['mape']:.4f}")
        summary_cols[4].metric("R2", f"{row['r2']:.6f}")
        st.markdown("**Slice Ozeti**")
        styled_slice = slice_df.style
        if "mae" in slice_df.columns:
            styled_slice = styled_slice.background_gradient(subset=["mae"], cmap="RdYlGn_r")
        if "rmse" in slice_df.columns:
            styled_slice = styled_slice.background_gradient(subset=["rmse"], cmap="RdYlGn_r")
        st.dataframe(styled_slice, use_container_width=True, height=460)

    with tab_satirlar:
        st.caption(
            "Asagidaki tablo, custom input ekraninda gorecegin ciktinin toplu halidir. "
            "Her satir icin gercek deger, model tahmini ve hata birlikte listelenir."
        )
        filter_cols = st.columns(4)
        engine_options = ["All"] + sorted(row_df["engine_type"].dropna().astype(str).unique().tolist())
        selected_engine = filter_cols[0].selectbox("engine_type filtresi", engine_options, key="report_engine_filter")
        altitude_values = sorted(row_df["altitude"].dropna().astype(float).unique().tolist())
        altitude_options = ["All"] + [f"{int(value)} ft" for value in altitude_values]
        selected_altitude_label = filter_cols[1].selectbox("altitude filtresi", altitude_options, key="report_alt_filter")
        error_column = "xgboost_absolute_error" if report_model == "xgboost" else "ft_transformer_absolute_error"
        if error_column in row_df.columns:
            max_error = float(row_df[error_column].fillna(0.0).max())
        else:
            max_error = 0.0
        selected_error_band = filter_cols[2].slider(
            "Max absolute error",
            min_value=0.0,
            max_value=max(max_error, 0.001),
            value=max(max_error, 0.001),
            key="report_error_filter",
        )
        sort_mode = filter_cols[3].selectbox(
            "Siralama",
            ["En buyuk hata ustte", "En kucuk hata ustte", "row_id"],
            key="report_sort_mode",
        )

        filtered_rows = row_df.copy()
        if selected_engine != "All":
            filtered_rows = filtered_rows[filtered_rows["engine_type"] == selected_engine]
        if selected_altitude_label != "All":
            selected_altitude = float(selected_altitude_label.replace(" ft", ""))
            filtered_rows = filtered_rows[filtered_rows["altitude"].astype(float) == selected_altitude]
        if error_column in filtered_rows.columns:
            filtered_rows = filtered_rows[filtered_rows[error_column].astype(float) <= float(selected_error_band)]
            if sort_mode == "En buyuk hata ustte":
                filtered_rows = filtered_rows.sort_values(error_column, ascending=False)
            elif sort_mode == "En kucuk hata ustte":
                filtered_rows = filtered_rows.sort_values(error_column, ascending=True)
            else:
                filtered_rows = filtered_rows.sort_values("row_id", ascending=True)
        else:
            filtered_rows = filtered_rows.sort_values("row_id", ascending=True)

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        detail_col1.metric("Filtrelenmis satir", len(filtered_rows))
        if error_column in filtered_rows.columns and not filtered_rows.empty:
            detail_col2.metric("Ortalama abs error", f"{filtered_rows[error_column].mean():.6f}")
            detail_col3.metric("Maks abs error", f"{filtered_rows[error_column].max():.6f}")

        preview_count = min(len(row_df), 2500)
        display_df = filtered_rows[available]
        if error_column in display_df.columns:
            styled_rows = display_df.style.background_gradient(subset=[error_column], cmap="RdYlGn_r")
        else:
            styled_rows = display_df.style
        st.dataframe(styled_rows, use_container_width=True, height=640)
        st.caption(
            f"Toplam satir: {len(row_df)}. Filtre sonrasi: {len(filtered_rows)}. "
            f"Tablo akici sekilde kaydirilabilir."
        )

        detail_candidates = filtered_rows["row_id"].astype(int).tolist()[:500]
        if detail_candidates:
            selected_row_id = st.selectbox("Detay satiri", detail_candidates, key="report_detail_row")
            selected_row = filtered_rows[filtered_rows["row_id"].astype(int) == int(selected_row_id)].iloc[0]
            st.markdown("**Secili Satir Detayi**")
            detail_payload = {column: selected_row[column] for column in available if column in selected_row.index}
            st.json(detail_payload)

        csv_bytes = filtered_rows.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Satir-bazli raporu indir (CSV)",
            data=csv_bytes,
            file_name=paths["row_level"].name,
            mime="text/csv",
        )

    with tab_grafikler:
        grafikler = []
        if paths["slice_plot"].exists():
            grafikler.append(("Gercek vs Tahmin Grafigi", paths["slice_plot"]))
        if paths["summary_plot"].exists():
            grafikler.append(("Slice Ozet Hata Grafigi", paths["summary_plot"]))

        if not grafikler:
            st.info("Grafik bulunamadi. Rapor komutunu tekrar calistirman gerekebilir.")
        else:
            columns = st.columns(2)
            for index, (baslik, yol) in enumerate(grafikler):
                with columns[index % 2]:
                    st.markdown(f"**{baslik}**")
                    st.caption(str(yol))
                    _render_clickable_image_preview(
                        yol,
                        caption=baslik,
                        key=f"preview_{report_model}_{index}",
                    )
                    with open(yol, "rb") as image_file:
                        st.download_button(
                            "Grafigi indir",
                            data=image_file.read(),
                            file_name=yol.name,
                            mime="image/png",
                            key=f"download_plot_{report_model}_{index}",
                        )

    with tab_nomogram:
        st.caption(
            "Bu alan handbook benzeri kategorik grafik mantigini taslak olarak uretir. "
            "`altitude` ve `gross_weight` birer slice, `drag_index` ise egri ailesidir."
        )
        available_engine_types = sorted(reference_df["engine_type"].dropna().astype(str).unique().tolist())
        nomo_engine_type = st.selectbox("Nomogram engine_type", available_engine_types, key="nomo_engine")
        nomo_altitudes = sorted(
            reference_df[reference_df["engine_type"] == nomo_engine_type]["altitude"].dropna().astype(float).unique().tolist()
        )
        nomo_altitude = st.selectbox(
            "Nomogram altitude",
            nomo_altitudes,
            key="nomo_altitude",
            format_func=lambda x: f"{int(x)} ft",
        )
        nomo_weights = sorted(
            reference_df[
                (reference_df["engine_type"] == nomo_engine_type)
                & (reference_df["altitude"].astype(float) == float(nomo_altitude))
            ]["gross_weight"].dropna().astype(float).unique().tolist()
        )
        if nomo_weights:
            nomo_weight = st.selectbox(
                "Nomogram gross_weight",
                nomo_weights,
                key="nomo_weight",
                format_func=lambda x: f"{int(x)} lb",
            )
        else:
            nomo_weight = None

        if st.button("Taslak Nomogram Uret", key="run_nomogram_ui") and nomo_weight is not None:
            if report_model == "xgboost":
                predictor = load_xgboost_predictor()
                batch_predict_fn = predictor.predict_many_from_frame
            else:
                predictor = load_ft_predictor(st.session_state.get("ft_device", "cpu"))
                batch_predict_fn = predictor.predict_many_from_frame

            result = generate_nomogram_report(
                reference_df,
                model_name=report_model,
                batch_predict_fn=batch_predict_fn,
                output_dir=_nomogram_dir(report_model),
                engine_type=nomo_engine_type,
                altitude=float(nomo_altitude),
                gross_weight=float(nomo_weight),
            )
            st.success(f"Taslak nomogram olusturuldu: `{result.nomogram_png}`")
            _render_clickable_image_preview(
                result.nomogram_png,
                caption="Taslak Nomogram",
                key=f"nomogram_preview_{report_model}",
                max_width=1500,
            )
            st.caption(str(result.nomogram_png))


def _render_setup_tab(ft_device: str) -> None:
    st.subheader("Setup ve Calistirma")
    st.caption(
        "Quickstart akisini bu ekrandan da yurutebilirsin. "
        "Butonlar ilgili scriptleri bu arayuzun arkasindan calistirir ve ciktilari asagida toplar."
    )

    dataset_path = str(DataConfig().processed_path)
    python_exec = _python_executable()
    xgb_device = st.selectbox("XGBoost cihazi", ["cpu", "cuda"], index=1 if ft_device == "cuda" else 0, key="setup_xgb_device")
    setup_ft_device = st.selectbox("FT-Transformer cihazi", ["cpu", "cuda"], index=1 if ft_device == "cuda" else 0, key="setup_ft_device")

    command_specs = _setup_command_specs(python_exec, dataset_path, xgb_device, setup_ft_device)

    full_sequence = [
        "Veri pipeline",
        "XGBoost egit + rapor",
        "FT-Transformer egit + rapor",
        "Toplu rapor uret",
    ]

    left_col, right_col = st.columns([1.2, 1.8], gap="large")
    with left_col:
        st.markdown("**Hazir komutlar**")
        for label, spec in command_specs.items():
            parts = spec["command"]
            st.code(_format_shell_command(parts), language="bash")
            st.caption(f"Tahmini sure: {spec['eta']}")
            with st.expander(f"Uretecegi artefactlar: {label}", expanded=False):
                for item in spec["artifacts"]:
                    st.write(f"- `{item}`")
            if st.button(f"Calistir: {label}", key=f"run_{label}"):
                with st.spinner(f"{label} calisiyor..."):
                    ok, output = _run_setup_command(parts, label)
                st.session_state.setdefault("setup_logs", [])
                st.session_state["setup_logs"].append(output)
                if ok:
                    st.success(f"{label} tamamlandi.")
                else:
                    st.error(f"{label} basarisiz oldu.")

        st.markdown("**Toplu quickstart akisi**")
        st.caption("Arayuzu yeniden acan komutlar dahil edilmedi; egitim ve raporlama akisidir.")
        if st.button("Quickstart setup'i calistir", type="primary", key="run_full_setup"):
            st.session_state.setdefault("setup_logs", [])
            progress = st.progress(0.0, text="Quickstart setup basliyor...")
            status_placeholder = st.empty()
            all_ok = True
            for index, label in enumerate(full_sequence, start=1):
                status_placeholder.info(f"{label} calisiyor...")
                with st.spinner(f"{label} calisiyor..."):
                    ok, output = _run_setup_command(command_specs[label]["command"], label)
                st.session_state["setup_logs"].append(output)
                progress.progress(index / len(full_sequence), text=f"{label} tamamlandi.")
                if not ok:
                    all_ok = False
                    st.error(f"Akis {label} adiminda durdu.")
                    break
            if all_ok:
                status_placeholder.success("Quickstart setup akisi tamamlandi.")
                st.success("Quickstart setup akisi tamamlandi.")

    with right_col:
        st.markdown("**Calisma gunlugu**")
        logs = st.session_state.get("setup_logs", [])
        if not logs:
            st.info("Henuz setup komutu calistirilmadi.")
        else:
            joined_logs = "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(reversed(logs[-8:]))
            st.text_area("Son loglar", value=joined_logs.strip(), height=620)

        st.markdown("**Notlar**")
        st.write("- Bu sekme UI'nin icinden scriptleri calistirir; uzun egitimlerde sayfa bir sure mesgul kalir.")
        st.write("- `Quickstart setup`, README'deki ardisik backend komutlarinin arayuz karsiligidir.")
        st.write("- Bu ekrandan `streamlit run ...` veya masaustu uygulamasi baslatma komutu calistirilmiyor; zaten aktif arayuzun icindesin.")


def _render_single_input(ft_device: str) -> None:
    st.subheader("Tekil Tahmin (Custom Input)")
    st.caption("Bu mod tek bir custom input veya hazir senaryo denemek icin kullanilir.")

    scenarios = load_test_scenarios()
    reference_df = load_reference_data()
    scenario_names = [scenario["name"] for scenario in scenarios]
    selected_scenario_name = st.sidebar.selectbox("Hazir test senaryosu", scenario_names, index=0)
    selected_scenario = next(scenario for scenario in scenarios if scenario["name"] == selected_scenario_name)

    col1, col2, col3 = st.columns(3)
    altitude = col1.number_input("Altitude (ft)", min_value=0.0, value=float(selected_scenario["altitude"]), step=500.0)
    gross_weight = col2.number_input("Gross Weight (lb)", min_value=0.0, value=float(selected_scenario["gross_weight"]), step=500.0)
    drag_index = col3.number_input("Drag Index", min_value=0.0, value=float(selected_scenario["drag_index"]), step=1.0)

    col4, col5, col6 = st.columns(3)
    mach = col4.number_input("Mach", min_value=0.0, max_value=2.0, value=float(selected_scenario["mach"]), step=0.01, format="%.4f")
    fuel_flow = col5.number_input("Fuel Flow (lb / h)", min_value=0.0, value=float(selected_scenario["fuel_flow"]), step=100.0)
    engine_type = col6.selectbox("Engine Type", ["one_engine", "two_engine"], index=["one_engine", "two_engine"].index(str(selected_scenario["engine_type"])))

    frame = build_single_row_frame(
        altitude=altitude,
        gross_weight=gross_weight,
        drag_index=drag_index,
        mach=mach,
        fuel_flow=fuel_flow,
        engine_type=engine_type,
    )
    st.dataframe(frame.drop(columns=["specific_range"]), use_container_width=True)
    mode = st.radio("Tahmin modu", ["Compare both", "XGBoost only", "FT-Transformer only"], horizontal=True)

    if st.button("Tahmini Hesapla", type="primary"):
        xgb_value: float | None = None
        ft_value: float | None = None
        result_cols = st.columns(2)
        if mode in {"Compare both", "XGBoost only"}:
            xgb_value = load_xgboost_predictor().predict_from_frame(frame)
            with result_cols[0]:
                st.markdown("### XGBoost")
                st.metric("specific_range", f"{xgb_value:.6f}")
        if mode in {"Compare both", "FT-Transformer only"}:
            ft_value = load_ft_predictor(ft_device).predict_from_frame(frame)
            target_col = 1 if mode == "Compare both" else 0
            with result_cols[target_col]:
                st.markdown("### FT-Transformer")
                st.metric("specific_range", f"{ft_value:.6f}")

        if xgb_value is not None and ft_value is not None:
            diff_cols = st.columns(3)
            diff_cols[0].metric("Model farki", f"{ft_value - xgb_value:.6f}")
            diff_cols[1].metric("XGBoost", f"{xgb_value:.6f}")
            diff_cols[2].metric("FT-Transformer", f"{ft_value:.6f}")

        st.subheader("Gercek Veriyle Kiyas")
        exact_matches = find_exact_match(reference_df, frame)
        if not exact_matches.empty:
            actual_value = float(exact_matches.iloc[0]["specific_range"])
            st.success(f"Birebir eslesen gercek satir bulundu. actual_specific_range = {actual_value:.6f}")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Actual", f"{actual_value:.6f}")
            if xgb_value is not None:
                metric_cols[1].metric("XGBoost absolute error", f"{abs(xgb_value - actual_value):.6f}")
            if ft_value is not None:
                metric_cols[2].metric("FT absolute error", f"{abs(ft_value - actual_value):.6f}")
            st.dataframe(exact_matches, use_container_width=True, height=180)
        else:
            st.info("Birebir eslesen gercek satir bulunamadi. En yakin gercek satirlar gosteriliyor.")
            nearest = find_nearest_reference_rows(reference_df, frame, top_k=5)
            if not nearest.empty:
                nearest_actual = float(nearest.iloc[0]["specific_range"])
                metric_cols = st.columns(3)
                metric_cols[0].metric("Nearest actual", f"{nearest_actual:.6f}")
                if xgb_value is not None:
                    metric_cols[1].metric("XGBoost abs diff to nearest", f"{abs(xgb_value - nearest_actual):.6f}")
                if ft_value is not None:
                    metric_cols[2].metric("FT abs diff to nearest", f"{abs(ft_value - nearest_actual):.6f}")
                visible_columns = [
                    column
                    for column in [
                        "row_id",
                        "engine_type",
                        "altitude",
                        "gross_weight",
                        "drag_index",
                        "mach",
                        "fuel_flow",
                        "specific_range",
                        "distance",
                    ]
                    if column in nearest.columns
                ]
                st.dataframe(nearest[visible_columns], use_container_width=True, height=260)


def main() -> None:
    data_config = DataConfig()
    _ = load_reference_data()
    st.title("Specific Range Tahmin ve Toplu Karsilastirma")
    st.caption("Arayuz metinleri Turkce tutuldu. Tablo kolon isimleri raporlar ile ayni kalsin diye Ingilizce birakildi.")

    with st.sidebar:
        st.header("Artifact Dizini")
        st.write(f"XGBoost: `{data_config.xgboost_artifact_dir}`")
        st.write(f"FT-Transformer: `{data_config.ft_transformer_artifact_dir}`")
        ft_device = st.selectbox("FT-Transformer cihazi", ["cpu", "cuda"], index=0)
        ekran = st.radio("Ekran", ["Hazir rapor", "Tekil tahmin", "Setup"], index=0)
        st.info(
            "Toplu rapor icin once su komutu calistir:\n"
            "`python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model both --ft-device cpu`"
        )

    if ekran == "Hazir rapor":
        _render_report_viewer()
    elif ekran == "Setup":
        _render_setup_tab(ft_device)
    else:
        _render_single_input(ft_device)


if __name__ == "__main__":
    main()
