# JARVIS Environment Setup Script for Windows

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  JARVIS Environment Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Create virtual environment if it doesn't exist
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment using Python 3.12..." -ForegroundColor Yellow
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.12 (py -3.12) command not found. Falling back to default python..." -ForegroundColor Yellow
        python -m venv .venv
    }
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 2. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Install core dependencies
Write-Host "Installing core dependencies (requirements.txt)..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Install voice dependencies
Write-Host "Installing voice dependencies (requirements-voice.txt)..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -m pip install -r requirements-voice.txt

# 5. Run environment verification
Write-Host "Verifying environment..." -ForegroundColor Yellow
& .venv\Scripts\python.exe backend/scripts/check_environment.py

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Environment Ready" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
