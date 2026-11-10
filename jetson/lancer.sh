#!/usr/bin/env bash
# Lance AlloIA :                ./jetson/lancer.sh
# Liste les cartes son :        ./jetson/lancer.sh --lister-audio
# Teste micro et haut-parleur : ./jetson/lancer.sh --test-audio
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
    echo "AlloIA n'est pas installé : lancez d'abord ./jetson/installer.sh"
    exit 1
fi
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export NO_PROXY=127.0.0.1,localhost
export OMP_NUM_THREADS=4   # Whisper sur 4 des 6 cœurs, le reste pour le système et la voix
exec .venv/bin/python -m alloia "$@"
