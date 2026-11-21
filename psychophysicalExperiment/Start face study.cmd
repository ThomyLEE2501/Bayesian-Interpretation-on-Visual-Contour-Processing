@echo off
cd /d "%~dp0"
python face_slideshow.py %*
if errorlevel 1 pause
