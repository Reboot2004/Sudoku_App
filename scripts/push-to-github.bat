@echo off
setlocal
cd /d "%~dp0.."

if not exist .git (
    git init -b main
)

git remote get-url origin >nul 2>&1
if errorlevel 1 (
    git remote add origin https://github.com/Reboot2004/Sudoku_App.git
)

git add .
git commit -m "feat: initialize Sudoku mobile app and DC ingestion pipeline"
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo Push failed. Make sure GitHub authentication is configured.
    echo You can also run: git push -u origin main
    exit /b 1
)

echo.
echo Repository pushed successfully.
endlocal
