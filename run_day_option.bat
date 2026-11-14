@echo off
chcp 65001 > nul
title Day Option Bot (EMA 200 + Fibo 0.500/0.618 + Real-Time Black-Scholes)
echo ======================================================================
echo   Launching DAY OPTION BOT: EMA 200 (2m) + Fibo 0.500/0.618 Scanner
echo ======================================================================
cd /d "%~dp0\.."
python -m day_option_bot.src.main --once
echo.
echo Scan complete! Opening Dashboard...
start "" "%~dp0docs\index.html"
pause
