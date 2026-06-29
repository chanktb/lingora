@echo off
REM Lingora run wrapper (Windows). Activates .venv then runs run.py.
REM No args = interactive wizard. Any args = pass straight through.
REM Keeps window open at end so a double-click user can read the result.

setlocal

if not exist .venv (
  echo .venv not found. Run SETUP.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat

REM Force UTF-8 stdout for non-ASCII output (Vietnamese, German umlauts, ...)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if "%~1"=="" (
  python run.py
) else (
  python run.py %*
)

set EXITCODE=%ERRORLEVEL%

echo.
if "%EXITCODE%"=="0" (
  echo === Done. Your video is in channels\^<channel^>\jobs\^<auto-...^>\output.mp4 ===
) else (
  echo === Lingora exited with an error ^(code %EXITCODE%^). Read the messages above. ===
)
echo.
pause

endlocal
exit /b %EXITCODE%
