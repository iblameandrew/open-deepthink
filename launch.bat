@echo off
echo ===================================================
echo   open-deepthink Launcher
echo ===================================================
echo.

echo [1/2] Installing/Updating dependencies (library + web UI)...
pip install -e ".[web]"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install package. Falling back to requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency install failed.
        pause
        exit /b %errorlevel%
    )
)

echo.
echo [2/2] Launching open-deepthink...
echo Access the app at: http://localhost:8000
echo Tip: copy .env.example to .env and set OPENROUTER_API_KEY
echo.
python -m deepthink
if %errorlevel% neq 0 (
    echo Module entry failed; trying app.py ...
    python app.py
)
pause
