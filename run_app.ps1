# PowerShell launcher script for Soil PSD Analyzer
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Launching Soil PSD Analyzer (Conda env: dl_clean)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

conda activate dl_clean
if ($LASTEXITCODE -ne 0) {
    & "$env:USERPROFILE\.conda\envs\dl_clean\Scripts\streamlit.exe" run app.py
} else {
    streamlit run app.py
}
