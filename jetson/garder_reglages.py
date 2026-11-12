"""Recopie vos réglages personnels d'un ancien alloia.yaml vers alloia.perso.yaml.

Utilisation (depuis le dossier du projet) :
    .venv/bin/python jetson/garder_reglages.py ~/ancien-alloia.yaml

Seuls le jeton Hugging Face, la carte son et la voix choisie sont repris : les autres
réglages viennent de la nouvelle version d'alloia.yaml.
"""

from __future__ import annotations

import pathlib
import sys

import yaml

RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from alloia.reglages import FICHIER_PERSO, lire_yaml  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    ancien = lire_yaml(pathlib.Path(sys.argv[1]).expanduser())
    perso = lire_yaml(FICHIER_PERSO) if FICHIER_PERSO.exists() else {}
    if ancien.get("jeton_huggingface"):
        perso["jeton_huggingface"] = ancien["jeton_huggingface"]
    audio = {k: v for k, v in (ancien.get("audio") or {}).items() if k in ("entree", "sortie", "volume") and v}
    if audio:
        perso.setdefault("audio", {}).update(audio)
    nom_voix = (ancien.get("voix") or {}).get("nom")
    if nom_voix:
        perso.setdefault("voix", {})["nom"] = nom_voix
    FICHIER_PERSO.write_text(
        "# Vos réglages personnels (prioritaires sur alloia.yaml, jamais écrasés par git pull).\n"
        + yaml.safe_dump(perso, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Réglages repris dans {FICHIER_PERSO.name} :")
    for cle, valeur in perso.items():
        if cle == "jeton_huggingface":
            valeur = str(valeur)[:6] + "…"
        print(f"  {cle} : {valeur}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
