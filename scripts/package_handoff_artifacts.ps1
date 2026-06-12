param(
    [string]$OutputDir = "handoff_payload",
    [string]$ZipName = "specific_range_handoff_artifacts.zip"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputPath = Join-Path $Root $OutputDir
$StagePath = Join-Path $OutputPath "specific_range_handoff_artifacts"
$ZipPath = Join-Path $OutputPath $ZipName

if (Test-Path $StagePath) {
    Remove-Item -LiteralPath $StagePath -Recurse -Force
}
New-Item -ItemType Directory -Path $StagePath | Out-Null
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null

$Items = @(
    "data/processed/combined_specific_range.csv",
    "artifacts/xgboost/model.json",
    "artifacts/xgboost/config.json",
    "artifacts/xgboost/metrics.json",
    "artifacts/xgboost/preprocessor.joblib",
    "artifacts/ft_transformer/model.pt",
    "artifacts/ft_transformer/model_numpy.npz",
    "artifacts/ft_transformer/metrics.json",
    "artifacts/ft_transformer/preprocessor.joblib",
    "artifacts/benchmarks/pi_benchmark_latest.json",
    "artifacts/benchmarks/pi_benchmark_history.csv",
    "artifacts/pso/xgboost_pso_latest.json",
    "artifacts/pso/xgboost_pso_history_latest.csv",
    "artifacts/pso/ft_transformer_pso_latest.json",
    "artifacts/pso/ft_transformer_pso_history_latest.csv",
    "artifacts/pso/pso_latest.json",
    "artifacts/pso/pso_history_latest.csv"
)

foreach ($Item in $Items) {
    $Source = Join-Path $Root $Item
    if (Test-Path $Source) {
        $Destination = Join-Path $StagePath $Item
        New-Item -ItemType Directory -Path (Split-Path $Destination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
        Write-Host "included $Item"
    } else {
        Write-Host "missing  $Item"
    }
}

$Manifest = Join-Path $StagePath "ARTIFACT_MANIFEST.txt"
@(
    "Specific Range Studio handoff artifact bundle",
    "Created: $(Get-Date -Format o)",
    "",
    "Restore with:",
    ".\scripts\restore_handoff_artifacts.ps1 -ZipPath <this zip>",
    "",
    "This bundle intentionally contains only runtime-critical generated files.",
    "It excludes heavy report outputs, external_apps, .venv and temporary folders."
) | Set-Content -LiteralPath $Manifest -Encoding UTF8

if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Compress-Archive -Path (Join-Path $StagePath "*") -DestinationPath $ZipPath -Force
Write-Host "created $ZipPath"

