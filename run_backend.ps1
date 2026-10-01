# # CareTaker.ai - Start Backend Server (Windows)
# # Run this from c:\CareTaker (project root)
# $ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
# $activate = Join-Path $ROOT "backend\venv\Scripts\Activate.ps1"
# if (-not (Test-Path $activate)) {
#     Write-Host "[ERROR] venv not found. Run setup.ps1 first." -ForegroundColor Red; exit 1
# }
# Write-Host "[CareTaker] Activating virtual environment..." -ForegroundColor Cyan
# & $activate
# Write-Host "[CareTaker] Starting FastAPI backend..." -ForegroundColor Green
# Set-Location (Join-Path $ROOT "backend")
# python main.py
# CareTaker.ai - Start Backend Server (Windows)
# Run this from c:\CareTaker (project root)
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
$activate = Join-Path $ROOT "backend\venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "[ERROR] venv not found. Run setup.ps1 first." -ForegroundColor Red; exit 1
}
Write-Host "[CareTaker] Activating virtual environment..." -ForegroundColor Cyan
& $activate

Write-Host "[CareTaker] Starting FastAPI backend..." -ForegroundColor Green
Set-Location (Join-Path $ROOT "backend")

# Demo Links - Print before server starts
Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "         CareTaker Backend Live & Ready!          " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Swagger Docs : http://localhost:8000/docs       " -ForegroundColor Yellow
Write-Host "  ReDoc Docs   : http://localhost:8000/redoc      " -ForegroundColor Yellow
Write-Host "  API Health   : http://localhost:8000/           " -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""

python main.py