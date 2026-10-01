#!/usr/bin/env bash
# =============================================================================
#  CareTaker.ai - Automated Setup Script (macOS / Linux / Bash)
#  Senior DevOps & Full-Stack Engineer Edition
# =============================================================================

set -e

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Colour

log_info()    { echo -e "${CYAN}  [INFO]  $1${NC}";    }
log_success() { echo -e "${GREEN}  [OK]    $1${NC}";   }
log_warn()    { echo -e "${YELLOW}  [WARN]  $1${NC}";  }
log_error()   { echo -e "${RED}  [ERROR] $1${NC}" >&2; }
log_step()    { echo -e "\n${MAGENTA}== $1${NC}";       }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo -e "${BLUE}  +------------------------------------------------------+${NC}"
echo -e "${BLUE}  |         CareTaker.ai -- Project Setup                |${NC}"
echo -e "${BLUE}  |        macOS/Linux Bash Automated Configurator       |${NC}"
echo -e "${BLUE}  +------------------------------------------------------+${NC}"
echo ""

# =============================================================================
# STEP 1 - Directory Check
# =============================================================================
log_step "STEP 1 - Directory Check"

for dir in backend frontend; do
    if [ -d "$ROOT/$dir" ]; then
        log_success "Found ./$dir"
    else
        log_error "./$dir folder not found. Run this script from the project root."
        exit 1
    fi
done

# =============================================================================
# STEP 2 - Python Virtual Environment Setup
# =============================================================================
log_step "STEP 2 - Python Virtual Environment Setup"

log_info "Checking Python installation..."

# Prefer python3, fallback to python
PYTHON_BIN=""
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    log_error "Python is not installed or not on PATH. Please install Python 3.10+."
    exit 1
fi

PY_VER=$("$PYTHON_BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    log_error "Python 3.10+ required. Detected: $("$PYTHON_BIN" --version 2>&1)"
    exit 1
fi
log_success "Python OK - $("$PYTHON_BIN" --version 2>&1)"

# -- Virtual environment -------------------------------------------------------
VENV_PATH="$ROOT/backend/venv"

if [ -d "$VENV_PATH" ]; then
    log_warn "Virtual environment already exists at backend/venv - skipping creation."
else
    log_info "Creating virtual environment at backend/venv ..."
    "$PYTHON_BIN" -m venv "$VENV_PATH"
    log_success "Virtual environment created."
fi

ACTIVATE="$VENV_PATH/bin/activate"
if [ ! -f "$ACTIVATE" ]; then
    log_error "Could not find venv activation script: $ACTIVATE"
    exit 1
fi

log_info "Activating virtual environment..."
source "$ACTIVATE"
log_success "Virtual environment activated."

# -- Upgrade pip / setuptools / wheel ------------------------------------------
log_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --quiet
log_success "pip, setuptools, and wheel are up-to-date."

# -- Install backend requirements ----------------------------------------------
REQ_FILE="$ROOT/backend/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    log_warn "backend/requirements.txt not found - skipping package installation."
else
    log_info "Installing Python packages from requirements.txt ..."
    pip install -r "$REQ_FILE"
    log_success "All Python packages installed."
fi

# =============================================================================
# STEP 3 - Frontend (Node.js / npm) Setup
# =============================================================================
log_step "STEP 3 - Frontend Environment Setup"

if ! command -v node &>/dev/null; then
    log_error "Node.js is not installed or not on PATH. Please install Node.js 16+."
    exit 1
fi
log_success "Node.js found - $(node --version)"

if ! command -v npm &>/dev/null; then
    log_error "npm is not installed or not on PATH."
    exit 1
fi
log_success "npm found - v$(npm --version)"

log_info "Running npm install in ./frontend ..."
cd "$ROOT/frontend"
npm install
cd "$ROOT"
log_success "Frontend npm packages installed."

# =============================================================================
# STEP 4 - .env File Configuration
# =============================================================================
log_step "STEP 4 - Environment File Configuration (.env)"

ENV_FILE="$ROOT/backend/.env"

if [ -f "$ENV_FILE" ]; then
    log_warn "backend/.env already exists - skipping generation to preserve your existing config."
else
    log_info "Generating default backend/.env ..."
    cat > "$ENV_FILE" <<'EOF'
# ---------------------------------------------------------------
#  CareTaker.ai - Backend Environment Configuration
#  Generated automatically by setup.sh
#  !! IMPORTANT: Replace all placeholder values before running !!
# ---------------------------------------------------------------

# PostgreSQL - replace YOUR_POSTGRES_PASSWORD with your real password
DATABASE_URL=postgresql://postgres:YOUR_POSTGRES_PASSWORD@localhost:5432/caretaker

# JWT - change to a long, random string in production
SECRET_KEY=caretaker_super_secret_jwt_key_2026
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Google Gemini API - obtain from https://aistudio.google.com/
GEMINI_API_KEY=your_gemini_api_key_here

# Mail (optional SMTP settings)
MAIL_USERNAME=user@example.com
MAIL_PASSWORD=password
MAIL_FROM=noreply@example.com
MAIL_SERVER=smtp.example.com
MAIL_PORT=587
EOF
    log_success "backend/.env created with default placeholders."
    log_warn "ACTION REQUIRED: Open backend/.env and fill in your real credentials."
fi

# =============================================================================
# STEP 5 - Helper Run Scripts
# =============================================================================
log_step "STEP 5 - Generating Helper Run Scripts"

# ── run_backend.sh ────────────────────────────────────────────────────────────
cat > "$ROOT/run_backend.sh" <<'EOF'
#!/usr/bin/env bash
# CareTaker.ai - Start Backend Server (macOS/Linux)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE="$ROOT/backend/venv/bin/activate"
if [ ! -f "$ACTIVATE" ]; then
  echo "[ERROR] venv not found. Run setup.sh first." && exit 1
fi
echo "[CareTaker] Activating virtual environment..."
source "$ACTIVATE"
echo "[CareTaker] Starting FastAPI backend (main.py)..."
cd "$ROOT/backend"
python main.py
EOF
chmod +x "$ROOT/run_backend.sh"
log_success "Created run_backend.sh"

# ── run_frontend.sh ───────────────────────────────────────────────────────────
cat > "$ROOT/run_frontend.sh" <<'EOF'
#!/usr/bin/env bash
# CareTaker.ai - Start Frontend Dev Server (macOS/Linux)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -d "$ROOT/frontend" ]; then
  echo "[ERROR] ./frontend not found." && exit 1
fi
echo "[CareTaker] Starting frontend dev server..."
cd "$ROOT/frontend"
npm run dev
EOF
chmod +x "$ROOT/run_frontend.sh"
log_success "Created run_frontend.sh"

# ── run_backend.ps1 ───────────────────────────────────────────────────────────
cat > "$ROOT/run_backend.ps1" <<'EOF'
# CareTaker.ai - Start Backend Server (Windows)
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
$activate = Join-Path $ROOT "backend\venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "[ERROR] venv not found. Run setup.ps1 first." -ForegroundColor Red; exit 1
}
Write-Host "[CareTaker] Activating virtual environment..." -ForegroundColor Cyan
& $activate
Write-Host "[CareTaker] Starting FastAPI backend (main.py)..." -ForegroundColor Green
Set-Location (Join-Path $ROOT "backend")
python main.py
EOF
log_success "Created run_backend.ps1"

# ── run_frontend.ps1 ──────────────────────────────────────────────────────────
cat > "$ROOT/run_frontend.ps1" <<'EOF'
# CareTaker.ai - Start Frontend Dev Server (Windows)
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
$frontend = Join-Path $ROOT "frontend"
if (-not (Test-Path $frontend)) {
    Write-Host "[ERROR] ./frontend not found." -ForegroundColor Red; exit 1
}
Write-Host "[CareTaker] Starting frontend dev server..." -ForegroundColor Green
Set-Location $frontend
npm run dev
EOF
log_success "Created run_frontend.ps1"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${GREEN}  +------------------------------------------------------+${NC}"
echo -e "${GREEN}  |       Setup Complete! CareTaker.ai is ready.         |${NC}"
echo -e "${GREEN}  +------------------------------------------------------+${NC}"
echo ""
echo "  Next steps:"
echo -e "${YELLOW}    1. Edit backend/.env  -- add your DB password + Gemini API key${NC}"
echo -e "${CYAN}    2. Run backend  -->  bash run_backend.sh${NC}"
echo -e "${CYAN}    3. Run frontend -->  bash run_frontend.sh${NC}"
echo ""
