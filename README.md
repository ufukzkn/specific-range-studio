# Aircraft Specific Range Modeling

This repository contains a modular research codebase for predicting `specific_range` from aircraft performance tables. The current focus is tabular modeling with:

- `FT-Transformer` as the main model,
- `XGBoost` as the baseline,
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
```

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

Bu arayuz Flask tabanlidir ve tarayicida `http://localhost:5000` adresinde acilir.
Icerdigi ana sekmeler:

- `Genel Bakis`
- `Karsilastirma`
- `Tekil Tahmin`
- `Nomogram`
- `Setup`

`Setup` sekmesi veri pipeline, egitim ve toplu raporlama adimlarini arayuz icinden calistirabilir.

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
- predictions can be generated with XGBoost, FT-Transformer, or both side by side.
- report metrics, slice plots, row-level comparison tables, nomogram generation, and setup commands are available from the same interface,
- the comparison tab includes a cost-function simulator for approximate accuracy / latency / memory trade-off exploration.

Legacy interfaces are still present:

- `scripts/launch_ui.py` for Streamlit,
- `scripts/desktop_app_qt.py` for a native Qt window.

The old `tkinter` desktop version is kept only for reference and is not the primary interface anymore.

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
- Add interpolation or lookup-table baselines for direct comparison against modern ML models.
- Extend PSO to XGBoost using the same objective function for a fully standardized comparison protocol.
