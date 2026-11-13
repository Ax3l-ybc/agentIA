#!/usr/bin/env bash
# Installation d'AlloIA (version « rapide ») sur une Jetson Orin Nano (JetPack 6, Ubuntu 22.04).
# Le cerveau (Gemma 4 E2B, qui écoute directement la voix) tourne dans llama.cpp, compilé ici
# pour la puce graphique de la Jetson. À lancer une fois, depuis le dossier du projet :
#     ./jetson/installer.sh
# Il peut être relancé sans risque : il reprend là où il s'est arrêté.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"

DOSSIER="$HOME/AlloIA"                        # gros fichiers, hors du dépôt git
LLAMA_VERSION="v0.9.0"                         # version de llama.cpp testée avec AlloIA
MODELES="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main"
FICHIERS_MODELE="gemma-4-E2B-it-Q3_K_M.gguf mmproj-F16.gguf"

etape() { echo; echo "=== [$1/6] $2 ==="; }

etape 1 "Paquets du système (mot de passe demandé)"
# Termine une éventuelle installation interrompue auparavant (sinon apt refuse de continuer).
sudo dpkg --configure -a || sudo apt-get install -f -y
sudo apt-get update
sudo apt-get install -y curl git build-essential cmake libportaudio2 alsa-utils
if ! command -v nvcc >/dev/null; then
    echo "Le compilateur CUDA (nvcc) est absent : installation des outils CUDA de JetPack…"
    sudo apt-get install -y cuda-toolkit-12-6 || sudo apt-get install -y nvidia-jetpack
fi
nvcc --version | tail -1

etape 2 "uv et Python 3.12"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r jetson/requirements.txt

etape 3 "Arrêt d'Ollama (il garderait un modèle en mémoire)"
if systemctl list-unit-files 2>/dev/null | grep -q '^ollama.service'; then
    sudo systemctl disable --now ollama || true
    echo "Ollama est arrêté et ne démarre plus tout seul (pour le réactiver : sudo systemctl enable --now ollama)."
else
    echo "Ollama n'est pas installé : rien à faire."
fi

etape 4 "llama.cpp pour la puce graphique (compilation : 30 à 60 minutes la première fois)"
mkdir -p "$DOSSIER"
if [ ! -x "$DOSSIER/llama.cpp/build/bin/llama-server" ]; then
    [ -d "$DOSSIER/llama.cpp" ] || git clone --depth 1 --branch "$LLAMA_VERSION" \
        https://github.com/ggml-org/llama.cpp "$DOSSIER/llama.cpp"
    cd "$DOSSIER/llama.cpp"
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release
    echo "Compilation en cours. Si elle s'arrête faute de mémoire, elle reprend avec moins de cœurs."
    cmake --build build --target llama-server -j4 || cmake --build build --target llama-server -j2
    cd - >/dev/null
fi
"$DOSSIER/llama.cpp/build/bin/llama-server" --version 2>&1 | tail -2

etape 5 "Gemma 4 E2B pour llama.cpp (environ 3,3 Go à télécharger)"
mkdir -p "$DOSSIER/modeles"
for f in $FICHIERS_MODELE; do
    if [ ! -s "$DOSSIER/modeles/$f" ] || [ -f "$DOSSIER/modeles/$f.partiel" ]; then
        echo "Téléchargement de $f…"
        touch "$DOSSIER/modeles/$f.partiel"
        curl -L --fail -C - -o "$DOSSIER/modeles/$f" "$MODELES/$f"
        rm -f "$DOSSIER/modeles/$f.partiel"
    fi
done
ls -lh "$DOSSIER/modeles"

etape 6 "Autoriser AlloIA à vider le cache mémoire au lancement"
# La puce graphique de la Jetson ne récupère pas la mémoire occupée par le cache des fichiers :
# on le vide juste avant de charger le modèle. Seule cette petite commande est autorisée sans
# mot de passe.
printf '#!/bin/sh\nsync\necho 3 > /proc/sys/vm/drop_caches\n' | sudo tee /usr/local/sbin/alloia-vider-cache >/dev/null
sudo chmod 755 /usr/local/sbin/alloia-vider-cache
echo "$USER ALL=(root) NOPASSWD: /usr/local/sbin/alloia-vider-cache" | sudo tee /etc/sudoers.d/alloia >/dev/null
sudo chmod 440 /etc/sudoers.d/alloia
sudo visudo -cf /etc/sudoers.d/alloia

echo
echo "=== Installation terminée ! ==="
echo "1. Branchez la carte son USB, puis :  ./jetson/lancer.sh --lister-audio"
echo "2. Indiquez le nom de la carte et votre jeton Hugging Face dans alloia.perso.yaml"
echo "3. Testez le son :                    ./jetson/lancer.sh --test-audio"
echo "4. Lancez AlloIA :                    ./jetson/lancer.sh"
