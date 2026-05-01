$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Server = Join-Path $ProjectRoot "scripts\web_app\server.py"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Virtual environment not found. Create it first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\activate"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}

Write-Host "Starting Specific Range Studio..." -ForegroundColor Cyan
Write-Host "URL: http://localhost:5000" -ForegroundColor Green
& $Python -u $Server
