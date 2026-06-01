@echo off
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   Speaker Diarization -- Setup
echo ==================================================
echo.

:: ── 1. Check prerequisites ──────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Download from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause & exit /b 1
)

node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Download from https://nodejs.org
    pause & exit /b 1
)

npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Reinstall Node.js from https://nodejs.org
    pause & exit /b 1
)

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ffmpeg not found.
    echo         Download from https://ffmpeg.org/download.html
    echo         Extract and add the bin\ folder to your System PATH.
    pause & exit /b 1
)

echo [OK] Prerequisites found: Python, Node.js, npm, ffmpeg
echo.

:: ── 2. Python virtual environment ───────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo [->] Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
)

echo [->] Installing Python dependencies (~5-10 min first time)...
venv\Scripts\python.exe -m pip install --upgrade pip -q
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
echo [OK] Python dependencies installed
echo.

:: ── 3. Export ONNX model ─────────────────────────────────────────────────────
if not exist "speaker_diarization\models\ecapa_tdnn_int8.onnx" (
    echo [->] Exporting ECAPA-TDNN ONNX model (~3 min)...
    set PYTHONPATH=speaker_diarization
    venv\Scripts\python.exe speaker_diarization\models\export_onnx.py --fp32
    if errorlevel 1 ( echo [ERROR] ONNX export failed. & pause & exit /b 1 )
    echo [OK] ONNX model exported
) else (
    echo [OK] ONNX model already exists
)
echo.

:: ── 4. Environment file ───────────────────────────────────────────────────────
if not exist ".env" (
    copy .env.example .env >nul
    echo [!!] Created .env from template.
    echo      You MUST add your HuggingFace token before running:
    echo      1. Go to https://huggingface.co/settings/tokens
    echo      2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
    echo      3. Open .env and set: HF_TOKEN=hf_your_token_here
    echo.
    notepad .env
) else (
    echo [OK] .env already exists
)
echo.

:: ── 5. React frontend ─────────────────────────────────────────────────────────
echo [->] Installing frontend dependencies...
cd ui
npm install
if errorlevel 1 ( echo [ERROR] npm install failed. & cd .. & pause & exit /b 1 )
npm run build
if errorlevel 1 ( echo [ERROR] npm build failed. & cd .. & pause & exit /b 1 )
cd ..
echo [OK] Frontend built
echo.

echo ==================================================
echo   Setup complete!
echo.
echo   To start the app, run:
echo     run.bat
echo.
echo   Then open: http://localhost:5001
echo ==================================================
echo.
pause
