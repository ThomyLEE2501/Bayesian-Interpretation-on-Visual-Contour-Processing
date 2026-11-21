@echo off
cd /d "%~dp0"
python choose_contour.py
if errorlevel 1 pause
