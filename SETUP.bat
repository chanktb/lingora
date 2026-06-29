@echo off
REM Lingora SETUP (Windows). Creates .venv, installs deps, checks Node + ffmpeg.
REM ASCII-only (cmd.exe parser is unreliable with non-ASCII).

setlocal

echo === Lingora SETUP ===
echo.

REM 1. Check Python 3.12+
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: python not found on PATH. Install Python 3.12+ from https://www.python.org/
  pause
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.12 or newer required.
  python --version
  pause
  exit /b 1
)

REM 2. Check Node 22+ (renderer shells out to npx hyperframes)
where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: node not found on PATH. Install Node.js 22+ from https://nodejs.org/
  pause
  exit /b 1
)
where npx >nul 2>nul
if errorlevel 1 (
  echo ERROR: npx not found. Reinstall Node.js with npm.
  pause
  exit /b 1
)

REM 3. Check ffmpeg
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ERROR: ffmpeg not found on PATH.
  echo   Install: https://www.gyan.dev/ffmpeg/builds/  ^(Windows^)
  echo   Or:      winget install ffmpeg
  pause
  exit /b 1
)

REM 4. Create .venv if missing
if not exist .venv (
  echo Creating .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo ERROR: failed to create .venv
    pause
    exit /b 1
  )
)

REM 5. Activate + install Python deps
echo Installing Python dependencies ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: pip install failed.
  pause
  exit /b 1
)

REM 6. Hint about first .env
if not exist channels\myvideo\.env (
  echo.
  echo NOTE: No channel config found yet.
  echo   Run RUN.bat once — the wizard will create channels\myvideo\.env for you.
  echo   Then open it and paste your GEMINI_API_KEYS ^(free: aistudio.google.com/apikey^).
)

echo.
echo === SETUP DONE ===
echo Run with: RUN.bat
pause
endlocal
