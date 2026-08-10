@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (set python_embeded_path=%~dp0python_embeded) else (set python_embeded_path=E:\python_embeded)
choice /C yn /M "Native"
if !errorlevel! equ 1 (set mode=native) else (set mode=triton)
set SAGEATTN_BACKEND=%mode%

rd /s /q %UserProfile%\.triton\cache >nul 2>&1
echo %mode% mode
%python_embeded_path%\python.exe benchmark_attn.py
%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-result-%mode%.txt 2>&1
