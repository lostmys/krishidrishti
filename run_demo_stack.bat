@echo off
echo =====================================================================
echo    KRISHIDRISHTI AI - SIH 2026 DEMO STACK LAUNCHER
echo =====================================================================
echo Starting all 3 services...
echo.

echo [1/3] Starting Core AI + WhatsApp Farmer Portal on http://127.0.0.1:8001...
start "KrishiDrishti - Service 1: Core AI & WhatsApp (Port 8001)" cmd /k "set PYTHONPATH=src && .venv\Scripts\python.exe -m uvicorn krishidrishti_ai.api:app --host 127.0.0.1 --port 8001"

timeout /t 3 /nobreak >nul

echo [2/3] Starting Cases API on http://127.0.0.1:8002...
start "KrishiDrishti - Service 2: Cases API (Port 8002)" cmd /k "set CASES_API_PORT=8002 && set IMAGE_AI_URL=http://127.0.0.1:8001 && set GEE_API_URL=http://127.0.0.1:8001 && .venv\Scripts\python.exe -m uvicorn cases_api:app --host 127.0.0.1 --port 8002"

timeout /t 2 /nobreak >nul

echo [3/3] Starting Officer Dashboard on http://127.0.0.1:8501...
start "KrishiDrishti - Service 3: Officer Dashboard (Port 8501)" cmd /k "set KRISHI_CASES_API=http://127.0.0.1:8002 && python -m streamlit run dashboard.py --server.port 8501 --server.headless true"

echo.
echo =====================================================================
echo    ALL SERVICES LAUNCHED SUCCESSFULLY!
echo =====================================================================
echo  - Farmer WhatsApp:   http://127.0.0.1:8001/whatsapp
echo  - Core AI Health:    http://127.0.0.1:8001/health
echo  - Cases API Docs:    http://127.0.0.1:8002/docs
echo  - Officer Dashboard: http://127.0.0.1:8501
echo =====================================================================
