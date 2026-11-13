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
if [ $# -eq 0 ]; then
    # Libère la mémoire avant de charger le modèle sur la puce graphique.
    if systemctl is-active --quiet ollama 2>/dev/null; then
        echo "Attention : Ollama tourne encore et occupe de la mémoire (sudo systemctl stop ollama)."
    fi
    sudo -n /usr/local/sbin/alloia-vider-cache 2>/dev/null \
        || echo "(Cache mémoire non vidé : relancez ./jetson/installer.sh pour l'autoriser.)"
fi
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export NO_PROXY=127.0.0.1,localhost
export OMP_NUM_THREADS=4   # voix Phonon sur 4 des 6 cœurs, le reste pour le système
exec .venv/bin/python -m alloia "$@"
