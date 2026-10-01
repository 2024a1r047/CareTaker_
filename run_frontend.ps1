# CareTaker.ai - Start Frontend Dev Server (Windows)
# Refresh PATH so Node.js/npm is always found
$env:PATH = [System.Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('PATH','User')
# Run this from c:\CareTaker (project root)
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
$frontend = Join-Path $ROOT "frontend"
if (-not (Test-Path $frontend)) {
    Write-Host "[ERROR] ./frontend not found." -ForegroundColor Red; exit 1
}
Write-Host "[CareTaker] Starting frontend dev server..." -ForegroundColor Green
Set-Location $frontend

# Dashboard Links - Print before server starts
Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "         CareTaker Frontend Live & Ready!          " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Home Page       : http://localhost:3000           " -ForegroundColor Yellow
Write-Host "  Caretaker Login : http://localhost:3000/login.html?role=caretaker" -ForegroundColor Yellow
Write-Host "  Doctor Login    : http://localhost:3000/login.html?role=doctor" -ForegroundColor Yellow
Write-Host "  Admin Panel     : http://localhost:3000/admin_dashboard.html" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

npm run dev
