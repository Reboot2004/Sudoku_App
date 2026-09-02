@echo off
setlocal
cd /d "%~dp0..\frontend"
call npm install
if errorlevel 1 exit /b 1
call npm run build
if errorlevel 1 exit /b 1
if not exist android call npx cap add android
if errorlevel 1 exit /b 1
call npx cap sync android
if errorlevel 1 exit /b 1
cd android
call gradlew.bat assembleDebug
if errorlevel 1 (
  echo Wrapper failed. If Gradle is installed globally, run: gradle assembleDebug
  exit /b 1
)
echo APK: %CD%\app\build\outputs\apk\debug\app-debug.apk
endlocal
