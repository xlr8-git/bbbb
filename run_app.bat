@echo off
title Soil Particle Size Distribution Analyzer
echo ========================================================
echo  Launching Soil PSD Analyzer (Conda env: dl_clean)
echo ========================================================
echo.

call conda activate dl_clean
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'dl_clean'.
    echo Trying direct python path...
    "%USERPROFILE%\.conda\envs\dl_clean\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)

pause
