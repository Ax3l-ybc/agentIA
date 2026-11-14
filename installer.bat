@echo off
setlocal
cd /d "%~dp0"
title Installation - Directeur de memoire

echo ==========================================================
echo    Installation du Directeur de memoire
echo    (a faire une seule fois, comptez 10 a 30 minutes)
echo ==========================================================
echo.

rem ---- 1. Outil "uv" (installe Python automatiquement) ----
where uv >nul 2>nul
if not errorlevel 1 goto uv_ok
echo [1/5] Installation de l'outil uv...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERREUR : l'outil uv n'a pas pu etre installe.
    echo Verifiez votre connexion internet puis relancez installer.bat.
    pause
    exit /b 1
)
:uv_ok
echo [1/5] Outil uv : OK

rem ---- 2. Python 3.12 dans un dossier .venv ----
echo [2/5] Preparation de Python...
if not exist ".venv\Scripts\python.exe" (
    uv venv .venv --python 3.12
    if errorlevel 1 goto erreur
)

rem ---- 3. Bibliotheques (voix Phonon, detecteur de parole, page web) ----
echo [3/5] Telechargement des bibliotheques...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 goto erreur

rem ---- 4. llama.cpp (le programme qui fait tourner Gemma 4) ----
echo [4/5] Verification de llama.cpp...
where llama-server >nul 2>nul
if not errorlevel 1 goto llama_ok
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\llama-server.exe" goto llama_ok
echo        llama.cpp n'est pas installe : installation automatique...
winget install llama.cpp --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
    echo.
    echo ATTENTION : llama.cpp n'a pas pu etre installe automatiquement.
    echo Voir le README, partie "Si llama.cpp ne s'installe pas".
    pause
    exit /b 1
)
:llama_ok
echo        llama.cpp : OK

rem ---- 5. Le cerveau : Gemma 4 E2B (environ 3,8 Go, une seule fois) ----
echo [5/5] Telechargement de Gemma 4 E2B (environ 3,8 Go, long la premiere fois)...
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
".venv\Scripts\python.exe" -m agent.telecharger
if errorlevel 1 goto erreur

echo.
echo ==========================================================
echo    Installation terminee !
echo    1. Verifiez config.yaml (vos dossiers et votre jeton Hugging Face).
echo    2. Double-cliquez sur lancer.bat
echo    Ollama n'est plus utilise : vous pouvez le quitter (icone pres de
echo    l'horloge) pour liberer de la memoire.
echo ==========================================================
pause
exit /b 0

:erreur
echo.
echo ERREUR pendant l'installation. Verifiez votre connexion internet
echo puis relancez installer.bat (il reprendra la ou il s'est arrete).
pause
exit /b 1
