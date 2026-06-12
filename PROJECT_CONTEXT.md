# Specific Range Studio - Project Context

This repository contains the final project codebase for modeling and comparing a fighter-aircraft specific range dataset with three method families:

- **Interpolation**: deterministic table/reference family. Runtime default is `Cubic Spline`.
- **XGBoost**: practical learned baseline and the default robust demo model.
- **FT-Transformer**: main academic tabular-transformer architecture candidate.

The current implementation is a Flask application plus research scripts. It is intended to support single prediction, method comparison, cost/benchmark review, nomogram/report figures, PSO optimization, and dataset-builder demos.

## Current Design Decisions

- Flask is the production/demo UI backend. FastAPI was discussed but intentionally postponed.
- Interpolation is implemented inside `src/interpolation/`; the app does **not** call `external_apps` at runtime.
- Runtime default interpolation method is `spline`, displayed as **Cubic Spline**.
- Interpolation is used as a deterministic reference family in custom-input scenarios; it should not be scored as an ordinary learned-model accuracy competitor against itself.
- XGBoost and FT-Transformer are compared with RMSE, MAE, MAPE, R2 and runtime/edge metrics.
- Cost/fit calculations are based on benchmark JSON files when available, not purely estimated sliders.
- PSO is implemented as a weighted scalarized multi-objective search:
  `J(theta) = w_rmse * RMSE/RMSE_ref + w_latency * T_inf/T_ref + w_size * S/S_ref`.
- PSO is available for both `ft_transformer` and `xgboost`.
- Dataset Builder lives under `tools/dataset_builder/` as a standalone internal tool panel, not as an external app dependency.

## Important Files

- `scripts/web_app/server.py`: Flask routes and API endpoints.
- `scripts/web_app/templates/index.html`: single-page UI shell.
- `scripts/web_app/static/js/app.js`: main browser behavior.
- `src/interpolation/specific_range.py`: cubic/linear/Newton interpolation service.
- `src/inference/predictors.py`: model loading and prediction helpers.
- `src/models/xgboost_baseline.py`: XGBoost training.
- `src/models/ft_transformer.py`: FT-Transformer model implementation.
- `src/optimization/objective.py`: scalarized objective for FT and XGBoost.
- `src/optimization/pso_search.py`: PSO search spaces and loop.
- `scripts/run_pso.py`: CLI for multi-objective PSO.
- `scripts/run_pi_benchmark.py`: benchmark runner producing cost inputs.
- `docs/methods/xgboost.md`: XGBoost explanation for UI info tab.
- `docs/methods/ft_transformer.md`: FT-Transformer explanation for UI info tab.
- `src/interpolation/README.md`: interpolation method notes.
- `HANDOFF.md`: step-by-step continuation instructions.
- `CODEX_CONTINUATION_PROMPT.md`: prompt to paste into a new Codex session.

## Data and Artifact Expectations

Tracked in Git:

- `One_Engine_Data.xlsx`
- `Two_Engine_Data.xlsx`
- source code
- method docs
- original research/report PDFs in `Raporlar/`
- selected docs under `docs/`

Included on the `codex/handoff-package` branch for handoff:

- `data/processed/combined_specific_range.csv`
- `artifacts/xgboost/`
- `artifacts/ft_transformer/`
- `artifacts/benchmarks/`
- `artifacts/pso/`
- `report_outputs/`
- `raporlar__/`
- generated comparison/final report documents under `docs/`

Still excluded:

- `.venv/`
- `external_apps/`
- `handoff_payload/`
- cache/build output folders

If GitHub transfer becomes too heavy or if a generated artifact is missing, use the fallback package script:

```powershell
.\scripts\package_handoff_artifacts.ps1
```

It creates a portable zip under `handoff_payload/`.

On another machine, clone this branch and run:

```powershell
.\scripts\restore_handoff_artifacts.ps1 -ZipPath .\handoff_payload\specific_range_handoff_artifacts.zip
```

Normally the branch itself should now contain the key runtime and report context. The zip is only a fallback for machine-local regenerated artifacts.

## Recommended Startup

```powershell
git clone -b codex/handoff-package https://github.com/ufukzkn/specific-range-studio.git
cd specific-range-studio
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
.\scripts\restore_handoff_artifacts.ps1 -ZipPath <path-to-specific_range_handoff_artifacts.zip>
python scripts/web_app/server.py
```

Open the local Flask URL shown by the server, usually:

```text
http://127.0.0.1:5000
```

## Current Project Narrative

The project demonstrates that a single method is not best for every requirement:

- **Cubic Spline interpolation** is the strongest traditional/reference method and best for deterministic table-like behavior.
- **XGBoost** is currently the most practical learned model for low-latency demo prediction and strong tabular accuracy.
- **FT-Transformer** is the academic/architectural candidate for studying feature tokenization and attention-based interaction learning.

The UI and reports should keep these roles distinct. Do not make interpolation "win" learned-model accuracy by comparing it against itself.
