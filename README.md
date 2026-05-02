# Aircraft Specific Range Modeling

This repository contains a modular research codebase for predicting `specific_range` from aircraft performance tables. The current focus is tabular modeling with:

- `FT-Transformer` as the main model,
- `XGBoost` as the baseline,
- classic table interpolation as the deterministic reference family,
- `PSO` for hyperparameter optimization,
- deployment-aware hooks for later ONNX / TensorRT / Jetson benchmarking.

## Data Expectations

The project expects these workbooks at the repository root:

- `One_Engine_Data.xlsx`
- `Two_Engine_Data.xlsx`

Observed workbook structure in the provided files:

- each sheet corresponds to an altitude level such as `Sea Level`, `5,000 Feet`, `10,000 Feet`, ...,
- columns are variants of:
  - `Altitude (ft)`
  - `Gross Weight (lb)`
  - `Drag Index`
  - `Mach Number (Ma)`
  - `Specific Range (NM)`
  - `Fuel Flow (lb / h)`
- decimal separators may be either `,` or `.`,
- header spelling and spacing may vary slightly across sheets and workbooks.

The loader standardizes these to the canonical schema:

- `altitude`
- `gross_weight`
- `drag_index`
- `mach`
- `fuel_flow`
- `engine_type`
- `specific_range`

## Project Layout

```text
src/
  interpolation/
    specific_range.py
  data/
    load_data.py
    preprocess.py
    split.py
  models/
    ft_transformer.py
    xgboost_baseline.py
  optimization/
    objective.py
    pso_search.py
  evaluation/
    benchmark.py
    metrics.py
  utils/
    config.py
    seed.py
scripts/
  web_app/
    server.py
    templates/
    static/
  run_data_pipeline.py
  train_ft_transformer.py
  train_xgboost.py
  run_pso.py
  compare_models.py
tools/
  dataset_builder/
    2-check_env_sys.py
    3-synthetic_production.py
    5-segment_curves.py
    train_unet.py
    synthetic_data_gui.py
```

Generated outputs are intentionally not tracked in Git:

- `artifacts/` for trained model files and table reports,
- `data/processed/` for the cleaned combined CSV,
- `report_outputs/` for draft document/report generation artifacts,
- `external_apps/` for archived/reference apps that are not part of the standalone runtime.

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quickstart

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/run_data_pipeline.py
python scripts/train_xgboost.py --dataset data/processed/combined_specific_range.csv --device cuda --run-table-report
python scripts/train_ft_transformer.py --dataset data/processed/combined_specific_range.csv --device cuda --run-table-report
python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model both --ft-device cuda
python scripts/web_app/server.py
```

GPU yoksa FT-Transformer komutunda `--device cpu`, toplu raporda da `--ft-device cpu` kullanin.

Fresh clone note:

- `One_Engine_Data.xlsx` and `Two_Engine_Data.xlsx` are tracked with the repository, so `python scripts/run_data_pipeline.py` can rebuild `data/processed/combined_specific_range.csv`.
- Trained model artifacts are not tracked. The Flask UI opens without them, but prediction/comparison/model report panels become fully usable after running the XGBoost and FT-Transformer training commands above.
- Dataset Builder demo code and selected sample assets live under `tools/dataset_builder/`; generated datasets, segmentation outputs and checkpoints remain ignored.

## Typical Workflow

1. Build the cleaned merged dataset:

```bash
python scripts/run_data_pipeline.py
```

2. Train the XGBoost baseline:

```bash
python scripts/train_xgboost.py --dataset data/processed/combined_specific_range.csv
```

3. Train the FT-Transformer:

```bash
python scripts/train_ft_transformer.py --dataset data/processed/combined_specific_range.csv --device cpu
```

GPU varsa:

```bash
python scripts/train_ft_transformer.py --dataset data/processed/combined_specific_range.csv --device cuda
python scripts/train_xgboost.py --dataset data/processed/combined_specific_range.csv --device cuda
```

4. Run PSO for FT-Transformer hyperparameters:

```bash
python scripts/run_pso.py --dataset data/processed/combined_specific_range.csv --iterations 5 --population 6
```

5. Compare both models:

```bash
python scripts/compare_models.py --dataset data/processed/combined_specific_range.csv --run-pso
```

6. Launch the main web UI:

```bash
python scripts/web_app/server.py
```

Windows kisayolu:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_flask_ui.ps1
```

veya dosyaya cift tiklamak icin:

```bat
scripts\start_flask_ui.bat
```

Bu arayuz Flask tabanlidir ve tarayicida `http://localhost:5000` adresinde acilir.
Icerdigi ana sekmeler:

- `Genel Bakis`
- `Karsilastirma`
- `Maliyet`
- `Tekil Tahmin`
- `Nomogram`
- `Veri Uretimi`
- `Setup`
- `Bilgi`

`Setup` sekmesi veri pipeline, egitim ve toplu raporlama adimlarini arayuz icinden calistirabilir.
`Veri Uretimi` sekmesi ise proje icine alinmis `tools/dataset_builder/` altindaki grafik segmentasyon ve sentetik veri araclarini launcher panel olarak acar; ana tahmin yontemi degildir.
`Maliyet` sekmesi XGBoost ve FT-Transformer icin tahmini dogruluk / gecikme / bellek odunlesimini gosterir; interpolasyon referans aile oldugu icin bu maliyet yarismasina dahil edilmez.
`Bilgi` sekmesi README notlarini, yontem rollerini ve sekme aciklamalarini arayuz icinden okunabilir hale getirir.

Ana Flask arayuzu artik uc ana yontemi tek panelde gosterir:

- `Interpolasyon`: deterministik tablo temelli referans aile. Varsayilan alt yontem `Cubic Spline`; tekil tahminde `Piecewise Linear` ve `Newton Divided Difference` da secilebilir.
- `XGBoost`: agac tabanli guclu baseline.
- `FT-Transformer`: projenin ana tabular transformer modeli.

Interpolasyon yontemi `altitude`, `gross_weight`, `drag_index` ve `mach` eksenlerini kullanir. `fuel_flow`, ML modellerinin girdisinde kalir; klasik interpolasyon tarafinda tablo ekseni olmadigi icin kullanilmaz.

7. Launch the legacy Streamlit UI if needed:

```bash
streamlit run scripts/launch_ui.py
```

Bu arayuz artik ikincil/legacy durumdadir; ana gelistirme akisi Flask web UI uzerinden devam etmektedir.

8. Launch the modern Qt desktop app if you want a native window:

```bash
python scripts/desktop_app_qt.py
```

Qt masaustu uygulamasinda da `Setup` sekmesi bulunur; veri pipeline, egitim ve toplu raporlama adimlari log ekranli bir panelden calistirilabilir. Ancak su an ana odak Flask tabanli web panelidir.

9. Generate full-table comparison reports from trained artifacts:

```bash
python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model both --ft-device cpu
```

10. Generate a draft nomogram-style comparison plot for one categorical slice:

```bash
python scripts/run_nomogram_report.py --dataset data/processed/combined_specific_range.csv --model xgboost --engine-type one_engine --altitude 5000 --gross-weight 50000
```

## Notes on Preprocessing

- train/validation/test split is performed before fitting transforms,
- imputers, scaler, and categorical encoder are fit on the training split only,
- numerical and categorical paths are kept separate so both tree models and FT-Transformer can consume the same dataset fairly,
- optional outlier clipping is implemented in the preprocessing module and can be enabled via config.

## UI Scope

The main UI is the Flask-based web app under `scripts/web_app/`.

It is designed for the workflow described by the project lead:

- users manually enter `altitude`, `gross_weight`, `drag_index`, `mach`, `fuel_flow`, and `engine_type`,
- the app can predict for intermediate values such as `11000 ft`, even when the original tables only contain `10000 ft` and `15000 ft`,
- predictions can be generated with Interpolation, XGBoost, FT-Transformer, or all three side by side.
- report metrics, slice plots, row-level comparison tables, nomogram generation, and setup commands are available from the same interface,
- the `Maliyet` tab includes a cost-function simulator for approximate accuracy / latency / memory trade-off exploration.
- interpolation is shown as a deterministic reference family; XGBoost and FT-Transformer are learned regressors trained from the processed table.
- the `Veri Uretimi` tab can launch the internal Dataset Builder GUI and whitelisted demo scripts with live logs.

Legacy interfaces are still present:

- `scripts/launch_ui.py` for Streamlit,
- `scripts/desktop_app_qt.py` for a native Qt window.

The old `tkinter` desktop version is kept only for reference and is not the primary interface anymore.

## Project Tools

This repository is designed to run as a standalone project. Runtime logic lives under `src/`, `scripts/`, and `tools/`.

- `src/interpolation/` contains the project-internal interpolation service for Linear, Spline, and Newton-style table interpolation.
- `tools/dataset_builder/` contains the graph segmentation, synthetic data generation, U-Net training, and Excel export tooling used by the Flask `Veri Uretimi` tab.

## Dataset Builder Tool

The Flask `Veri Uretimi` tab exposes `tools/dataset_builder/` as a controlled launcher:

- `Dataset Python Bagimliliklarini Kur`: installs Python packages such as `opencv-python`, `pdf2image`, `pytesseract`, `tqdm`, and `scikit-image` into the main project `.venv`.
- `Poppler Kur`: installs the Poppler system binary with `winget` when available.
- `Tesseract OCR Kur`: installs the Tesseract OCR system binary with `winget` when available.
- `Ortam Kontrolu`: checks dataset-tool Python dependencies.
- `Dataset GUI Ac`: starts `synthetic_data_gui.py` in a separate desktop process.
- `Sentetik Grafik Uret`: runs a small sequential `3-synthetic_production.py` demo command and creates `dataset_production/`.
- `U-Net Egit`: runs a short CPU demo for `train_unet.py`.
- `Segmentasyon / Excel Export`: runs `5-segment_curves.py` on the small `demo_graphs/` subset with the bundled model path when available.

For safety, the UI cannot run arbitrary commands; it can only call backend-defined command IDs through `/api/dataset-tools/run`.

Python package dependencies are listed in `requirements.txt`. PDF/OCR workflows can also require system binaries:

- `Poppler`: needed by `pdf2image` for PDF-to-image conversion.
- `Tesseract`: needed by `pytesseract` for OCR.

The Flask Dataset Tool status cards show both Python package status and system binary status. Missing Poppler/Tesseract does not break the main prediction UI; it only affects OCR/PDF-specific dataset workflows.
If a missing dependency has an install command, the status card shows a `Kur` button. Python packages are installed into the project `.venv`; Poppler/Tesseract are Windows system tools and may require restarting the Flask app or terminal so PATH changes are picked up.

## Full-Table Reports

If you want direct model-vs-table comparison without relying on the UI, use the reporting flow.

Training-time generation:

```bash
python scripts/train_xgboost.py --dataset data/processed/combined_specific_range.csv --device cpu --run-table-report
python scripts/train_ft_transformer.py --dataset data/processed/combined_specific_range.csv --device cpu --run-table-report
```

Post-training generation from saved artifacts:

```bash
python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model both --ft-device cpu
```

Generated outputs are saved under:

- `artifacts/xgboost/reports/`
- `artifacts/ft_transformer/reports/`

Each report directory contains:

- row-level comparison CSV
- slice summary CSV
- overall summary CSV
- actual-vs-predicted PNG plot
- slice-level error summary PNG plot

## Draft Nomogram Comparison

For handbook-style graph comparison, a draft nomogram report flow is included.

It currently treats:

- `altitude` as a categorical slice,
- `gross_weight` as a categorical slice,
- `drag_index` as the curve family,
- `mach` as the x-axis,
- `specific_range` as the y-axis,
- `fuel_flow` as a contextual curve label.

This is intentionally a draft comparison layer. It preserves the logic of the original chart family, but it is not yet a pixel-perfect recreation of the scanned manual graphs.

## PSO Objective

The optimization objective follows the planned multi-objective scalarization:

`J(theta) = w1 * RMSE / RMSE_ref + w2 * T_inf / T_ref + w3 * S / S_ref`

At the moment:

- validation `RMSE` is fully operational,
- inference latency and model size are wired through benchmark interfaces,
- ONNX export and TensorRT build functions are intentionally left as extension stubs until the target deployment environment is available.

## Assumptions

- Sheet names carry altitude information when row values are missing or textual.
- `Sea Level` is interpreted as `0 ft`.
- `engine_type` is derived from workbook identity: `one_engine` or `two_engine`.
- The target column is `specific_range`.
- Current scripts do not fabricate benchmark numbers; placeholder latency/size values are clearly labeled in the benchmark stub.
- The UI expects trained artifacts to exist under `artifacts/xgboost` and `artifacts/ft_transformer`.

## Extension Points

- Replace benchmark stubs with real ONNX export, TensorRT engine build, and Jetson latency measurement.
- Add more interpolation/reporting diagnostics for direct comparison against modern ML models.
- Extend PSO to XGBoost using the same objective function for a fully standardized comparison protocol.
