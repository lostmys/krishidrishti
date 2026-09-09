Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "   KRISHIDRISHTI AI - SIH 2026 DEMO STACK LAUNCHER (PowerShell)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Green

Write-Host "[1/3] Starting Core AI & Farmer WhatsApp Portal (Port 8001)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONPATH='src'; & '.\.venv\Scripts\python.exe' -m uvicorn krishidrishti_ai.api:app --host 127.0.0.1 --port 8001"

Start-Sleep -Seconds 2

Write-Host "[2/3] Starting Cases API Service (Port 8002)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:CASES_API_PORT='8002'; `$env:IMAGE_AI_URL='http://127.0.0.1:8001'; `$env:GEE_API_URL='http://127.0.0.1:8001'; & '.\.venv\Scripts\python.exe' -m uvicorn cases_api:app --host 127.0.0.1 --port 8002"

Start-Sleep -Seconds 2

Write-Host "[3/3] Starting Officer Dashboard (Port 8501)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:KRISHI_CASES_API='http://127.0.0.1:8002'; python -m streamlit run dashboard.py --server.port 8501 --server.headless true"

Write-Host "`nAll 3 services are launching in separate windows:" -ForegroundColor Green
Write-Host "  1. Farmer WhatsApp Interface : http://127.0.0.1:8001/whatsapp" -ForegroundColor White
Write-Host "  2. Core AI Health            : http://127.0.0.1:8001/health" -ForegroundColor White
Write-Host "  3. Cases API Docs            : http://127.0.0.1:8002/docs" -ForegroundColor White
Write-Host "  4. Officer Dashboard         : http://127.0.0.1:8501" -ForegroundColor White
