@echo off
rem Starts the Voice Clone Kit web app at http://127.0.0.1:7860 (only reachable from this PC).
if "%VOICECLONE_PYTHON%"=="" set "VOICECLONE_PYTHON=%USERPROFILE%\voice-clone-env\Scripts\python.exe"
if not exist "%VOICECLONE_PYTHON%" (
  echo The voice-clone Python environment was not found. Run setup.bat first.
  pause
  exit /b 1
)
cd /d "%~dp0"
echo Starting Voice Clone Kit... your browser will open at http://127.0.0.1:7860
echo Close this window to stop it.
"%VOICECLONE_PYTHON%" -X faulthandler app.py
pause
