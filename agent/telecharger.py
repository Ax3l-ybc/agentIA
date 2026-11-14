"""Télécharge Gemma 4 E2B pour llama.cpp (lancé par installer.bat).

    .venv\\Scripts\\python.exe -m agent.telecharger

Les fichiers vont dans %USERPROFILE%\\DirecteurMemoire\\modeles (environ 3,8 Go).
Un téléchargement interrompu reprend où il s'était arrêté.
"""

from __future__ import annotations

import sys

from .reglages import DOSSIER_DONNEES, MMPROJ_PAR_DEFAUT, MODELE_PAR_DEFAUT

DEPOT = "unsloth/gemma-4-E2B-it-GGUF"


def main() -> int:
    from huggingface_hub import hf_hub_download

    dossier = DOSSIER_DONNEES / "modeles"
    dossier.mkdir(parents=True, exist_ok=True)
    fichiers = [MODELE_PAR_DEFAUT, MMPROJ_PAR_DEFAUT] + sys.argv[1:]
    for nom in fichiers:
        cible = dossier / nom
        if cible.is_file() and cible.stat().st_size > 10_000_000:
            print(f"Déjà présent : {cible}")
            continue
        print(f"Téléchargement de {nom} (cela peut prendre plusieurs minutes)…")
        hf_hub_download(repo_id=DEPOT, filename=nom, local_dir=dossier)
        print(f"  -> {cible}")
    print("Modèle prêt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
