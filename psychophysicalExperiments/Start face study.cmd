@echo off
cd /d "%~dp0"
python run_face_reconstruction_study.py %*
if errorlevel 1 pause
