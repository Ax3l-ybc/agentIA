#!/usr/bin/env bash
# Installation d'AlloIA sur une Jetson Orin Nano (JetPack 6, Ubuntu 22.04).
# À lancer une fois, depuis le dossier du projet :   ./jetson/installer.sh
# Il peut être relancé sans risque : il reprend là où il s'est arrêté.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

etape() { echo; echo "=== [$1/6] $2 ==="; }

etape 1 "Paquets du système (mot de passe demandé)"
# Termine une éventuelle installation interrompue auparavant (sinon apt refuse de continuer).
sudo dpkg --configure -a || sudo apt-get install -f -y
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config curl git \
    libopus-dev libportaudio2 alsa-utils

etape 2 "Rust (nécessaire pour compiler Gradbot)"
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
fi
cargo --version

etape 3 "uv et Python 3.12"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
.venv/bin/python --version

etape 4 "Gradbot (compilation : 10 à 20 minutes la première fois)"
if ! .venv/bin/python -c "import gradbot" 2>/dev/null; then
    uv pip install --python .venv/bin/python \
        "gradbot @ git+https://github.com/gradium-ai/gradbot@v0.2.0#subdirectory=gradbot_py"
fi
.venv/bin/python -c "import gradbot; print('Gradbot : OK')"

etape 5 "Modules Python (Whisper, Phonon, audio)"
uv pip install --python .venv/bin/python -r jetson/requirements.txt

etape 6 "Ollama (le cerveau)"
if ! command -v ollama >/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
# Garder le modèle en mémoire (réponse rapide au décroché) et n'en charger qu'un à la fois.
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_KEEP_ALIVE=-1"\nEnvironment="OLLAMA_NUM_PARALLEL=1"\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\n' \
    | sudo tee /etc/systemd/system/ollama.service.d/alloia.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart ollama || true

echo
echo "=== Installation terminée ! ==="
echo "1. Branchez la carte son USB, puis :  ./jetson/lancer.sh --lister-audio"
echo "2. Indiquez le nom de la carte et votre jeton Hugging Face dans alloia.yaml"
echo "3. Testez le son :                    ./jetson/lancer.sh --test-audio"
echo "4. Lancez AlloIA :                    ./jetson/lancer.sh"
