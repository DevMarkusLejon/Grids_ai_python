@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "BUNDLED_NODE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"

if not exist "%BUNDLED_NODE%" (
  echo Bundled Node.js was not found at "%BUNDLED_NODE%"
  exit /b 1
)

"%BUNDLED_NODE%" --version
"%BUNDLED_NODE%" --check "%REPO_ROOT%\web\app.js"
