# =============================================================================
#  CareTaker.ai - Automated Setup Script (Windows / PowerShell)
#  Senior DevOps & Full-Stack Engineer Edition
# =============================================================================

$ErrorActionPreference = "Stop"

# ── Colour helpers ────────────────────────────────────────────────────────────
function Log-Info    { param($msg) Write-Host "  [INFO]  $msg" -ForegroundColor Cyan   }
function Log-Success { param($msg) Write-Host "  [OK]    $msg" -ForegroundColor Green  }
function Log-Warn    { param($msg) Write-Host "  [WARN]  $msg" -ForegroundColor Yellow }
function Log-Error   { param($msg) Write-Host "  [ERROR] $msg" -ForegroundColor Red    }
function Log-Step    { param($msg) Write-Host "`n== $msg" -ForegroundColor Magenta      }

Write-Host ""
Write-Host "  +------------------------------------------------------+" -ForegroundColor Blue
Write-Host "  |         CareTaker.ai -- Project Setup                |" -ForegroundColor Blue
Write-Host "  |     Windows PowerShell Automated Configurator        |" -ForegroundColor Blue
Write-Host "  +------------------------------------------------------+" -ForegroundColor Blue
Write-Host ""

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition

# =============================================================================
# STEP 1 - Directory Check
# =============================================================================
Log-Step "STEP 1 - Directory Check"

foreach ($dir in @("backend", "frontend")) {
    $path = Join-Path $ROOT $dir
    if (Test-Path $path) {
        Log-Success "Found ./$dir"
    } else {
        Log-Error   "./$dir folder not found. Run this script from the project root."
        exit 1
    }
}

# =============================================================================
# STEP 2 - Python Virtual Environment Setup
# =============================================================================
Log-Step "STEP 2 - Python Virtual Environment Setup"

Log-Info "Checking Python installation..."
try {
    $pyVer = & python --version 2>&1
} catch {
    Log-Error "Python is not installed or not on PATH. Please install Python 3.10+."
    exit 1
}

$verMatch = [regex]::Match($pyVer, '(\d+)\.(\d+)')
if ($verMatch.Success) {
    $major = [int]$verMatch.Groups[1].Value
    $minor = [int]$verMatch.Groups[2].Value
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Log-Error "Python 3.10+ required. Detected: $pyVer"
        exit 1
    }
    Log-Success "Python OK - $pyVer"
} else {
    Log-Error "Could not parse Python version: $pyVer"
    exit 1
}

# -- Virtual environment -------------------------------------------------------
$venvPath = Join-Path $ROOT "backend\venv"

if (Test-Path $venvPath) {
    Log-Warn "Virtual environment already exists at backend/venv - skipping creation."
} else {
    Log-Info "Creating virtual environment at backend/venv ..."
    Push-Location (Join-Path $ROOT "backend")
    & python -m venv venv
    Pop-Location
    Log-Success "Virtual environment created."
}

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Log-Error "Could not find venv activation script: $activateScript"
    exit 1
}
Log-Info "Activating virtual environment..."
& $activateScript
Log-Success "Virtual environment activated."

# -- Upgrade pip / setuptools / wheel ------------------------------------------
Log-Info "Upgrading pip, setuptools, and wheel..."
& python -m pip install --upgrade pip setuptools wheel --quiet
Log-Success "pip, setuptools, and wheel are up-to-date."

# -- Install backend requirements ----------------------------------------------
$reqFile = Join-Path $ROOT "backend\requirements.txt"
if (-not (Test-Path $reqFile)) {
    Log-Warn "backend/requirements.txt not found - skipping package installation."
} else {
    Log-Info "Installing Python packages from requirements.txt ..."
    & pip install -r $reqFile
    Log-Success "All Python packages installed."
}

# =============================================================================
# STEP 3 - Frontend (Node.js / npm) Setup
# =============================================================================
Log-Step "STEP 3 - Frontend Environment Setup"

try {
    $nodeVer = & node --version 2>&1
    Log-Success "Node.js found - $nodeVer"
} catch {
    Log-Error "Node.js is not installed or not on PATH. Please install Node.js 16+."
    exit 1
}

try {
    $npmVer = & npm --version 2>&1
    Log-Success "npm found - v$npmVer"
} catch {
    Log-Error "npm is not installed or not on PATH."
    exit 1
}

$frontendPath = Join-Path $ROOT "frontend"
Log-Info "Running npm install in ./frontend ..."
Push-Location $frontendPath
& npm install
Pop-Location
Log-Success "Frontend npm packages installed."

# =============================================================================
# STEP 4 - .env File Configuration
# =============================================================================
Log-Step "STEP 4 - Environment File Configuration (.env)"

$envFile = Join-Path $ROOT "backend\.env"

if (Test-Path $envFile) {
    Log-Warn "backend/.env already exists - skipping generation to preserve your existing config."
} else {
    Log-Info "Generating default backend/.env ..."

    $lines = @(
        "# ---------------------------------------------------------------",
        "#  CareTaker.ai - Backend Environment Configuration",
        "#  Generated automatically by setup.ps1",
        "#  !! IMPORTANT: Replace all placeholder values before running !!",
        "# ---------------------------------------------------------------",
        "",
        "# PostgreSQL - replace YOUR_POSTGRES_PASSWORD with your real password",
        "DATABASE_URL=postgresql://postgres:YOUR_POSTGRES_PASSWORD@localhost:5432/caretaker",
        "",
        "# JWT - change to a long, random string in production",
        "SECRET_KEY=caretaker_super_secret_jwt_key_2026",
        "ALGORITHM=HS256",
        "ACCESS_TOKEN_EXPIRE_MINUTES=60",
        "",
        "# Google Gemini API - obtain from https://aistudio.google.com/",
        "GEMINI_API_KEY=your_gemini_api_key_here",
        "",
        "# Mail (optional SMTP settings)",
        "MAIL_USERNAME=user@example.com",
        "MAIL_PASSWORD=password",
        "MAIL_FROM=noreply@example.com",
        "MAIL_SERVER=smtp.example.com",
        "MAIL_PORT=587"
    )

    $lines | Set-Content -Path $envFile -Encoding UTF8
    Log-Success "backend/.env created with default placeholders."
    Log-Warn "ACTION REQUIRED: Open backend/.env and fill in your real credentials."
}

# =============================================================================
# STEP 5 - Helper Run Scripts
# =============================================================================
Log-Step "STEP 5 - Generating Helper Run Scripts"

# ── run_backend.ps1 ───────────────────────────────────────────────────────────
$runBackendPs1Lines = @(
    '# CareTaker.ai - Start Backend Server (Windows)',
    '$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition',
    '$activate = Join-Path $ROOT "backend\venv\Scripts\Activate.ps1"',
    'if (-not (Test-Path $activate)) {',
    '    Write-Host "[ERROR] venv not found. Run setup.ps1 first." -ForegroundColor Red; exit 1',
    '}',
    'Write-Host "[CareTaker] Activating virtual environment..." -ForegroundColor Cyan',
    '& $activate',
    'Write-Host "[CareTaker] Starting FastAPI backend (main.py)..." -ForegroundColor Green',
    'Set-Location (Join-Path $ROOT "backend")',
    'python main.py'
)
$runBackendPs1Lines | Set-Content -Path (Join-Path $ROOT "run_backend.ps1") -Encoding UTF8
Log-Success "Created run_backend.ps1"

# ── run_frontend.ps1 ──────────────────────────────────────────────────────────
$runFrontendPs1Lines = @(
    '# CareTaker.ai - Start Frontend Dev Server (Windows)',
    '$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition',
    '$frontend = Join-Path $ROOT "frontend"',
    'if (-not (Test-Path $frontend)) {',
    '    Write-Host "[ERROR] ./frontend not found." -ForegroundColor Red; exit 1',
    '}',
    'Write-Host "[CareTaker] Starting frontend dev server..." -ForegroundColor Green',
    'Set-Location $frontend',
    'npm run dev'
)
$runFrontendPs1Lines | Set-Content -Path (Join-Path $ROOT "run_frontend.ps1") -Encoding UTF8
Log-Success "Created run_frontend.ps1"

# ── run_backend.sh ────────────────────────────────────────────────────────────
$runBackendShLines = @(
    '#!/usr/bin/env bash',
    '# CareTaker.ai - Start Backend Server (macOS/Linux)',
    'set -e',
    'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
    'ACTIVATE="$ROOT/backend/venv/bin/activate"',
    'if [ ! -f "$ACTIVATE" ]; then',
    '  echo "[ERROR] venv not found. Run setup.sh first." && exit 1',
    'fi',
    'echo "[CareTaker] Activating virtual environment..."',
    'source "$ACTIVATE"',
    'echo "[CareTaker] Starting FastAPI backend (main.py)..."',
    'cd "$ROOT/backend"',
    'python main.py'
)
$runBackendShLines | Set-Content -Path (Join-Path $ROOT "run_backend.sh") -Encoding UTF8
Log-Success "Created run_backend.sh"

# ── run_frontend.sh ───────────────────────────────────────────────────────────
$runFrontendShLines = @(
    '#!/usr/bin/env bash',
    '# CareTaker.ai - Start Frontend Dev Server (macOS/Linux)',
    'set -e',
    'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
    'if [ ! -d "$ROOT/frontend" ]; then',
    '  echo "[ERROR] ./frontend not found." && exit 1',
    'fi',
    'echo "[CareTaker] Starting frontend dev server..."',
    'cd "$ROOT/frontend"',
    'npm run dev'
)
$runFrontendShLines | Set-Content -Path (Join-Path $ROOT "run_frontend.sh") -Encoding UTF8
Log-Success "Created run_frontend.sh"

# =============================================================================
# DONE
# =============================================================================
Write-Host ""
Write-Host "  +------------------------------------------------------+" -ForegroundColor Green
Write-Host "  |       Setup Complete! CareTaker.ai is ready.         |" -ForegroundColor Green
Write-Host "  +------------------------------------------------------+" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "    1. Edit backend/.env  -- add your DB password + Gemini API key" -ForegroundColor Yellow
Write-Host "    2. Run backend  -->  .\run_backend.ps1"  -ForegroundColor Cyan
Write-Host "    3. Run frontend -->  .\run_frontend.ps1" -ForegroundColor Cyan
Write-Host ""
