@echo off
REM Change directory to the location of the script
cd /d "%~dp0"

REM Install packages from requirements.txt
python -m pip install --upgrade pip  REM Upgrade pip to the latest version
pip install -r requirements.txt

REM Pause to see the results
pause
