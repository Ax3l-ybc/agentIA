@echo off
setlocal
cd /d "%~dp0"
title Installation - Mon Assistant

echo ==========================================================
echo    Installation de Mon Assistant
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
echo [5/5] Telechargement de Gemma 4 E2B (environ 3,8 Go, long la premiere fois ;
echo        deja present si vous avez installe le directeur de memoire)...
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
".venv\Scripts\python.exe" -m assistant.telecharger
if errorlevel 1 goto erreur

rem ---- Carte graphique : llama.cpp adapte, si le PC en a une ----
".venv\Scripts\python.exe" -m assistant.materiel --installer

rem ---- Raccourci "Meilleur collegue" sur le Bureau ----
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0assistant\raccourci.ps1"

echo.
echo ==========================================================
echo    Installation terminee !
echo    1. Pour la voix : indiquez votre jeton Hugging Face dans config.yaml
echo       (voir le README). Sans jeton, l'assistant marche a l'ecrit.
echo    2. Double-cliquez sur "Meilleur collegue" sur votre Bureau
echo       (ou sur lancer.bat dans ce dossier).
echo ==========================================================
pause
exit /b 0

:erreur
echo.
echo ERREUR pendant l'installation. Verifiez votre connexion internet
echo puis relancez installer.bat (il reprendra la ou il s'est arrete).
pause
exit /b 1
