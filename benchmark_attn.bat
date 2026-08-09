@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (set python_embeded_path=%~dp0python_embeded) else (set python_embeded_path=E:\python_embeded)

rd /s /q %UserProfile%\.triton\cache >nul 2>&1
set SAGEATTN_BACKEND=triton
%python_embeded_path%\python.exe benchmark_attn.py
%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-result-triton.txt 2>&1

rd /s /q %UserProfile%\.triton\cache >nul 2>&1
set SAGEATTN_BACKEND=native
%python_embeded_path%\python.exe benchmark_attn.py
%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-result-native.txt 2>&1
