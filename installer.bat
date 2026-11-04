@echo off
setlocal
cd /d "%~dp0"
title Installation - Directeur de memoire

echo ==========================================================
echo    Installation du Directeur de memoire
echo    (a faire une seule fois, comptez 10 a 20 minutes)
echo ==========================================================
echo.

rem ---- 1. Outil "uv" (installe Python automatiquement) ----
where uv >nul 2>nul
if not errorlevel 1 goto uv_ok
echo [1/4] Installation de l'outil uv...
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
echo [1/4] Outil uv : OK

rem ---- 2. Python 3.12 dans un dossier .venv ----
echo [2/4] Preparation de Python...
if not exist ".venv\Scripts\python.exe" (
    uv venv .venv --python 3.12
    if errorlevel 1 goto erreur
)

rem ---- 3. Bibliotheques (avec carte graphique NVIDIA si presente) ----
echo [3/4] Telechargement des bibliotheques (long la premiere fois)...
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo        Pas de carte graphique NVIDIA detectee : version processeur.
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
) else (
    echo        Carte graphique NVIDIA detectee : version acceleree.
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
)
uv pip install --python .venv\Scripts\python.exe "torch>=2.7,<2.10" --index-url %TORCH_INDEX%
if errorlevel 1 goto erreur
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 goto erreur

rem ---- 4. Ollama ----
echo [4/4] Verification d'Ollama...
where ollama >nul 2>nul
if not errorlevel 1 goto ollama_ok
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" goto ollama_ok
echo        Ollama n'est pas installe : tentative d'installation automatique...
winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
    echo.
    echo ATTENTION : installez Ollama vous-meme depuis https://ollama.com/download
    echo puis relancez ce fichier.
    pause
    exit /b 1
)
:ollama_ok
echo        Ollama : OK

echo.
echo ==========================================================
echo    Installation terminee !
echo    1. Ouvrez config.yaml avec le Bloc-notes et indiquez vos dossiers
echo       et votre jeton Hugging Face (voir le README).
echo    2. Double-cliquez sur lancer.bat
echo ==========================================================
pause
exit /b 0

:erreur
echo.
echo ERREUR pendant l'installation. Verifiez votre connexion internet
echo puis relancez installer.bat (il reprendra la ou il s'est arrete).
pause
exit /b 1
