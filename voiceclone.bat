@echo off
setlocal EnableDelayedExpansion
rem Voice Clone Kit command line. Run from any folder, for example:
rem   voiceclone enroll bavly "my recording.m4a" --i-have-consent
rem   voiceclone speak bavly "Hello there, nice to meet you." --tone energetic
rem   voiceclone speak bavly --file script.txt --split --out C:\Videos\narration.wav
rem   voiceclone --help

if "%VOICECLONE_PYTHON%"=="" set "VOICECLONE_PYTHON=%USERPROFILE%\voice-clone-env\Scripts\python.exe"
if not exist "%VOICECLONE_PYTHON%" (
  echo The voice-clone Python environment was not found at %VOICECLONE_PYTHON%.
  echo Run setup.bat first.
  exit /b 1
)
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%VOICECLONE_PYTHON%" -X faulthandler -m voiceclone %*
set "CODE=!ERRORLEVEL!"
rem The model very occasionally crashes natively on Windows right after loading; one retry gets past it.
if "!CODE!"=="-1073741819" (
  echo Retrying once after a native crash...
  "%VOICECLONE_PYTHON%" -X faulthandler -m voiceclone %*
  set "CODE=!ERRORLEVEL!"
)
exit /b !CODE!
