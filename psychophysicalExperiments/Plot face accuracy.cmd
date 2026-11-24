@echo off
cd /d "%~dp0"
python plot_accuracy_contours_gui.py
if errorlevel 1 pause
