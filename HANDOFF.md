# Handoff Instructions

This file is for continuing the project on another computer or with another teammate.

## 1. Clone the Branch

```powershell
git clone -b codex/handoff-package https://github.com/ufukzkn/specific-range-studio.git
cd specific-range-studio
```

If the repository already exists:

```powershell
git fetch origin
git switch codex/handoff-package
git pull
```

## 2. Create Python Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Linux/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Check Runtime Artifacts

On this handoff branch, the important generated context has also been committed so the next machine can start with fewer missing pieces:

- processed CSV
- trained model artifacts
- benchmark JSON/CSV files
- PSO history/result files
- generated report figures
- final/report DOCX files
- `raporlar__/` review/reference reports

Check that these exist:

```powershell
Test-Path data/processed/combined_specific_range.csv
Test-Path artifacts/xgboost/model.json
Test-Path artifacts/ft_transformer/model.pt
Test-Path artifacts/benchmarks/pi_benchmark_latest.json
Test-Path docs/final_reports/LIFT_UP_TASLAK_duzeltilmis.docx
```

If a teammate receives a separate `specific_range_handoff_artifacts.zip`, restore it:

```powershell
.\scripts\restore_handoff_artifacts.ps1 -ZipPath C:\path\to\specific_range_handoff_artifacts.zip
```

If files are missing and no zip is available, rebuild the minimum:

```powershell
python scripts/run_data_pipeline.py
python scripts/train_xgboost.py
python scripts/train_ft_transformer.py
python scripts/run_pi_benchmark.py --models interpolation,xgboost,ft_transformer --sample-size 200 --warmup 50 --repetitions 500 --device cpu
```

## 4. Run the Web App

```powershell
python scripts/web_app/server.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 5. Important UI Tabs

- **Genel Bakış**: project summary and available models.
- **Karşılaştırma**: row/slice comparison of interpolation, XGBoost and FT-Transformer.
- **Tekil Tahmin**: user-provided single prediction.
- **Maliyet**: measured benchmark/cost and PSO results.
- **Nomogram**: report/nomogram-style outputs.
- **Veri Üretimi**: dataset-builder tool launcher and GUI-like panel.
- **Bilgi**: README/method explanations.
- **Setup**: setup and command helpers.

## 6. PSO Commands

XGBoost:

```powershell
python scripts/run_pso.py --model xgboost --dataset data/processed/combined_specific_range.csv --device cpu --population 6 --iterations 5 --latency-repetitions 50 --latency-warmup 10 --w-rmse 0.70 --w-latency 0.20 --w-size 0.10 --output artifacts/pso
```

FT-Transformer:

```powershell
python scripts/run_pso.py --model ft_transformer --dataset data/processed/combined_specific_range.csv --device cuda --population 4 --iterations 3 --ft-epochs 20 --ft-patience 5 --latency-repetitions 10 --latency-warmup 2 --w-rmse 0.70 --w-latency 0.20 --w-size 0.10 --output artifacts/pso
```

Use `--device cpu` if CUDA is unavailable.

## 7. Known Framing Rules

- Interpolation is not a learned model and should not be scored against its own reference output.
- In custom input, cubic spline interpolation acts as the deterministic reference/tahmini gerçek value.
- XGBoost and FT-Transformer compete on learned-model accuracy.
- Runtime and memory can include all three methods.
- Cost simulator should prefer measured benchmark files under `artifacts/benchmarks/`.
- If benchmark files are missing, the UI should show a missing-benchmark warning instead of pretending measured values exist.

## 8. If Codex Continues the Work

Paste the content of `CODEX_CONTINUATION_PROMPT.md` into the new Codex chat/session after cloning and restoring artifacts.
