@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title Directeur de memoire

if not exist ".venv\Scripts\python.exe" (
    echo L'agent n'est pas encore installe : double-cliquez d'abord sur installer.bat
    pause
    exit /b 1
)

rem Ollama installe pour l'utilisateur courant mais absent du PATH
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set NO_PROXY=127.0.0.1,localhost

echo ==========================================================
echo    Directeur de memoire
echo    La page s'ouvrira toute seule dans votre navigateur.
echo    Pour arreter : fermez cette fenetre.
echo ==========================================================
echo.

".venv\Scripts\python.exe" -m agent

echo.
echo L'agent s'est arrete. Lisez les messages ci-dessus en cas de probleme.
pause
