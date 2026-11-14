@echo off
chcp 65001 > nul
title Day Option Bot - LIVE 30-Second Monitor
echo ======================================================================
echo   DAY OPTION BOT: Continuous 30-Second Real-Time Monitor + Telegram
echo ======================================================================
cd /d "%~dp0\.."
python -m day_option_bot.src.main --loop --interval 30
pause
