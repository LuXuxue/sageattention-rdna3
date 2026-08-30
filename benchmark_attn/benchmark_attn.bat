@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (cd /d %~dp0python_embeded)
choice /C yn /M "Native"
if !errorlevel! equ 1 (set mode=native) else (set mode=triton)
set SAGEATTN_BACKEND=%mode%

rd /s /q %UserProfile%\.triton\cache >nul 2>&1
echo %mode% mode
python.exe %~dp0benchmark_attn.py
python.exe %~dp0benchmark_attn.py>%~dp0benchmark_attn-result-%mode%.txt
