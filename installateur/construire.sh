#!/usr/bin/env bash
# Fabrique l'installateur Windows (Meilleur-collegue-installation-<version>.exe), sous Linux ou macOS.
# Il faut NSIS : sudo apt install nsis   (ou : brew install makensis)
#     installateur/construire.sh            -> version 1.0.0
#     installateur/construire.sh 1.1.0      -> une autre version
set -euo pipefail
cd "$(dirname "$0")"
makensis -V2 -DVERSION="${1:-1.0.0}" meilleur-collegue.nsi
ls -la Meilleur-collegue-installation-*.exe
