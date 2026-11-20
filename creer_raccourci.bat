@echo off
rem Cree (ou recree) sur le Bureau le raccourci "Meilleur collegue", qui lance lancer.bat.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0assistant\raccourci.ps1"
if errorlevel 1 (
    echo Le raccourci n'a pas pu etre cree.
) else (
    echo Double-cliquez sur "Meilleur collegue" sur votre Bureau pour lancer l'assistant.
)
pause
