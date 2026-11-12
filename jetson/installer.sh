#!/usr/bin/env bash
# Installation d'AlloIA (version « omni ») sur une Jetson Orin Nano (JetPack 6, Ubuntu 22.04).
# À lancer une fois, depuis le dossier du projet :   ./jetson/installer.sh
# Il peut être relancé sans risque : il reprend là où il s'est arrêté.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

# Ollama minimal : 0.40.13 (Gemma 4 demande 0.30.9, lfm3-omni 0.40.13).
OLLAMA_MINI="0.40.13"

etape() { echo; echo "=== [$1/4] $2 ==="; }

etape 1 "Paquets du système (mot de passe demandé)"
# Termine une éventuelle installation interrompue auparavant (sinon apt refuse de continuer).
sudo dpkg --configure -a || sudo apt-get install -f -y
sudo apt-get update
sudo apt-get install -y curl git libportaudio2 alsa-utils zstd

etape 2 "uv et Python 3.12"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
.venv/bin/python --version

etape 3 "Modules Python (voix Phonon, détecteur de parole, audio)"
uv pip install --python .venv/bin/python -r jetson/requirements.txt

etape 4 "Ollama (le cerveau qui écoute)"
version_ollama() { ollama -v 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1; }
actuelle="$(command -v ollama >/dev/null && version_ollama || echo 0.0.0)"
if [ "$(printf '%s\n%s\n' "$OLLAMA_MINI" "$actuelle" | sort -V | head -1)" != "$OLLAMA_MINI" ]; then
    echo "Ollama $actuelle est absent ou trop ancien : installation de la dernière version…"
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "Ollama : $(version_ollama)"
# Garder le modèle en mémoire (réponse rapide au décroché) et n'en charger qu'un à la fois.
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_KEEP_ALIVE=-1"\nEnvironment="OLLAMA_NUM_PARALLEL=1"\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\n' \
    | sudo tee /etc/systemd/system/ollama.service.d/alloia.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart ollama || true

echo
echo "=== Installation terminée ! ==="
echo "1. Branchez la carte son USB, puis :  ./jetson/lancer.sh --lister-audio"
echo "2. Indiquez le nom de la carte et votre jeton Hugging Face dans alloia.perso.yaml"
echo "   (voir le README, partie « Vos réglages personnels »)"
echo "3. Testez le son :                    ./jetson/lancer.sh --test-audio"
echo "4. Lancez AlloIA :                    ./jetson/lancer.sh"
