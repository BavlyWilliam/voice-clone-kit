@echo off
setlocal
rem One-time setup for Voice Clone Kit: a Python environment with PyTorch and the open-source Chatterbox models.
rem The environment lives at %USERPROFILE%\voice-clone-env (a short path, which Windows needs for PyTorch).
rem Needs Python 3.10-3.13 and about 8 GB of disk (libraries ~4 GB, model downloads ~3-7 GB on first use).

set "VENV=%USERPROFILE%\voice-clone-env"
where ffmpeg >nul 2>nul || echo WARNING: ffmpeg is not on the PATH. Install it with:  winget install Gyan.FFmpeg

if exist "%VENV%\Scripts\python.exe" goto install
echo Creating the Python environment at %VENV% ...
py -3.13 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
if not exist "%VENV%\Scripts\python.exe" (
  echo Could not create a Python environment. Install Python 3.13 from python.org and run this again.
  exit /b 1
)

:install
set "PY=%VENV%\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
echo Installing PyTorch from %TORCH_INDEX% ...
"%PY%" -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url %TORCH_INDEX% || exit /b 1
echo Installing Chatterbox, faster-whisper and the app ...
"%PY%" -m pip install -r "%~dp0requirements.txt" || exit /b 1
echo.
echo Setup complete. Double-click start-app.bat, or use voiceclone.bat in a terminal.
