@echo off
setlocal enabledelayedexpansion
set SAGEATTN_BACKEND=native
rd /s /q %UserProfile%\.triton\cache >nul 2>&1
python.exe benchmark_attn.py
