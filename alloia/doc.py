"""Rédaction de la documentation du projet à partir de la conversation."""

from __future__ import annotations

import datetime
import logging
import pathlib
import re

from agent import ollama
from agent.consignes import slug

from .appel import Conversation

log = logging.getLogger("alloia.doc")

MOTS_MINIMUM = 15  # en dessous, le maker n'a pas raconté de projet

CONSIGNE = """
Tu rédiges la documentation d'un projet réalisé dans un FabLab, à partir de la transcription
d'un appel téléphonique où le maker a raconté son projet à AlloIA.
- Suis exactement le modèle ci-dessous (en-tête YAML compris), en français, en Markdown.
- Date du jour : {date}.
- N'invente rien : uniquement ce que le maker a dit. Si une information manque, ne la devine
  pas : indique-la dans la section « À compléter ».
- La transcription vient d'une reconnaissance vocale : corrige les mots manifestement mal
  reconnus (noms de machines, de matériaux, de logiciels).
- Écris pour quelqu'un qui voudrait refaire le projet : précis et concret.
- Réponds uniquement avec la documentation, sans commentaire avant ou après.

MODÈLE :
{modele}
"""


def nettoyer(reponse: str) -> str:
    texte = reponse.strip()
    blocs = re.findall(r"```(?:markdown|md)?\s*\n(.*?)\n```", texte, re.S)
    if blocs:
        texte = max(blocs, key=len).strip()
    return texte + "\n"


def nom_fichier(contenu: str, quand: datetime.datetime) -> str:
    titre = re.search(r"^#\s+(.+)$", contenu, re.M)
    projet = re.search(r"^projet:\s*\[?([^\]\n]+)", contenu, re.M)
    sujet = (projet.group(1) if projet else titre.group(1) if titre else "projet").strip(" \"'")
    if sujet.startswith("[") or sujet.lower().startswith("nom du projet"):
        sujet = "projet"
    return f"{quand:%Y-%m-%d_%Hh%M}_{slug(sujet)}.md"


def ecrire(dossier: pathlib.Path, nom: str, contenu: str) -> pathlib.Path:
    """Crée un NOUVEAU fichier (n'écrase jamais une doc existante)."""
    dossier.mkdir(parents=True, exist_ok=True)
    base = nom.removesuffix(".md")
    for i in range(1, 1000):
        cible = dossier / (f"{base}.md" if i == 1 else f"{base} ({i}).md")
        try:
            with open(cible, "x", encoding="utf-8") as f:
                f.write(contenu)
            return cible
        except FileExistsError:
            continue
    raise RuntimeError("Impossible de trouver un nom de fichier libre.")


def rediger(
    conv: Conversation,
    adresse: str,
    modele_llm: str,
    contexte: int,
    modele_doc: str,
    dossier: pathlib.Path,
    reflexion: bool | None = None,
) -> pathlib.Path | None:
    mots_maker = sum(len(e.texte.split()) for e in conv.echanges if e.role == "maker")
    if mots_maker < MOTS_MINIMUM:
        log.info("Appel trop court (%d mots du maker) : pas de documentation.", mots_maker)
        return None
    quand = conv.debut
    systeme = CONSIGNE.format(date=f"{quand:%Y-%m-%d}", modele=modele_doc)
    reponse = ollama.discuter(
        adresse,
        modele_llm,
        [
            {"role": "system", "content": systeme},
            {"role": "user", "content": "TRANSCRIPTION DE L'APPEL :\n" + conv.transcription()},
        ],
        contexte,
        reflexion,
    )
    contenu = nettoyer(reponse)
    contenu += (
        "\n---\n\n<details><summary>Transcription de l'appel</summary>\n\n"
        + conv.transcription()
        + "\n\n</details>\n"
    )
    chemin = ecrire(dossier, nom_fichier(contenu, quand), contenu)
    log.info("Documentation enregistrée : %s", chemin)
    return chemin
