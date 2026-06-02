@echo off

cd /d "%~dp0"

echo Starting Home Credit Pipeline...
echo.

call conda activate base

python pipeline.py

echo.
echo Pipeline Finished

pause