"""Télécharge Gemma 4 E2B pour llama.cpp (lancé par installer.bat).

    .venv\\Scripts\\python.exe -m assistant.telecharger

Les fichiers vont dans %USERPROFILE%\\MonAssistant\\modeles (environ 3,8 Go). S'ils ont déjà
été téléchargés pour le directeur de mémoire (DirecteurMemoire\\modeles), ils sont réutilisés.
Un téléchargement interrompu reprend où il s'était arrêté.
"""

from __future__ import annotations

import sys

from .reglages import DOSSIERS_MODELES, MMPROJ_PAR_DEFAUT, MODELE_PAR_DEFAUT

DEPOT = "unsloth/gemma-4-E2B-it-GGUF"


def main() -> int:
    from huggingface_hub import hf_hub_download

    dossier = DOSSIERS_MODELES[0]
    dossier.mkdir(parents=True, exist_ok=True)
    fichiers = [MODELE_PAR_DEFAUT, MMPROJ_PAR_DEFAUT] + sys.argv[1:]
    for nom in fichiers:
        present = next((d / nom for d in DOSSIERS_MODELES
                        if (d / nom).is_file() and (d / nom).stat().st_size > 10_000_000), None)
        if present:
            print(f"Déjà présent : {present}")
            continue
        cible = dossier / nom
        print(f"Téléchargement de {nom} (cela peut prendre plusieurs minutes)…")
        hf_hub_download(repo_id=DEPOT, filename=nom, local_dir=dossier)
        print(f"  -> {cible}")
    print("Modèle prêt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
