"""Consignes données au modèle.

Le comportement de l'agent vient du fichier « mode de conduite » (mode_de_conduite.md),
écrit par l'étudiant et relu au début de chaque séance. Ce module y ajoute seulement :
- une courte adaptation à l'oral (l'agent parle, il n'écrit pas) ;
- les fiches des séances précédentes et les notes, qui forment les « connaissances du projet ».
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import unicodedata

MODE_PAR_DEFAUT = """
Tu joues le rôle de mon directeur de mémoire. Tu es bienveillant mais exigeant : tu poses
plus de questions que tu ne donnes de réponses, une seule à la fois, et tu me dis franchement
quand une idée ne tient pas. Tu t'appuies sur mes notes, en citant leur nom, et tu relèves
les zones floues. N'invente jamais de référence. Tutoiement, réponses courtes.
"""

ADAPTATION_ORALE = """
# Adaptation à la conversation orale (à appliquer en plus du mode de conduite)
- La séance se fait à voix haute : tes réponses sont lues par une voix de synthèse.
  Phrases courtes, pas de listes, pas de titres, pas de symboles, pas de Markdown.
- Les « connaissances du projet » sont fournies plus bas : les fiches des séances
  précédentes et mes notes. Pour une note qui n'est pas reproduite en entier, utilise
  l'outil lire_note ; pour retrouver un passage, chercher_dans_notes ; pour une fiche plus
  ancienne, lire_fiche.
- Clôture : quand je dis « fiche », « slash fiche », qu'on termine, ou quand j'accepte ta
  proposition de fiche, appelle l'outil rediger_fiche_recap. Ne dicte surtout pas la fiche
  à voix haute : elle est rédigée selon le modèle du mode de conduite et enregistrée
  automatiquement dans le dossier des fiches. Dis simplement, en une phrase, qu'elle est
  en cours de rédaction.
- Tu ne modifies jamais mes notes et tu ne le proposes pas : tu travailles en lecture seule.
- Au tout début (message « [start] »), reprends le fil comme le demande le mode de conduite,
  en deux ou trois phrases, puis pose une seule question.
"""


def lire_mode(fichier: pathlib.Path | None) -> str:
    """Relit le mode de conduite (à chaque séance : une modification s'applique tout de suite)."""
    if fichier and fichier.is_file():
        texte = fichier.read_text(encoding="utf-8-sig").strip()
        if texte:
            return texte
    return MODE_PAR_DEFAUT.strip()


def construire(
    mode: str,
    sujet: str,
    index: list[dict],
    texte_complet: str | None,
    fiches: list[dict],
    noms_fiches: list[str],
    aujourd_hui: datetime.date | None = None,
) -> str:
    aujourd_hui = aujourd_hui or datetime.date.today()
    morceaux = [
        "# MODE DE CONDUITE (à suivre pendant toute la séance)\n" + mode,
        ADAPTATION_ORALE.strip(),
        f"Nous sommes le {aujourd_hui:%Y-%m-%d}.",
    ]
    if sujet:
        morceaux.append(f"SUJET DU MÉMOIRE : {sujet}")
    if fiches:
        bloc = "\n\n".join(f"=== FICHE : {f['nom']} ===\n{f['contenu']}" for f in fiches)
        morceaux.append("FICHES DES SÉANCES LES PLUS RÉCENTES (de la plus récente à la plus ancienne) :\n" + bloc)
        anciennes = [n for n in noms_fiches if n not in {f["nom"] for f in fiches}]
        if anciennes:
            morceaux.append("Fiches plus anciennes (outil lire_fiche) : " + ", ".join(anciennes[:30]))
    else:
        morceaux.append("Aucune fiche de séance précédente : c'est la première séance.")
    morceaux.append(
        "LISTE DE MES NOTES (avec leurs intertitres) :\n" + json.dumps(index, ensure_ascii=False, indent=0)
    )
    if texte_complet:
        morceaux.append("CONTENU COMPLET DE MES NOTES :\n" + texte_complet)
    else:
        morceaux.append(
            "Mes notes sont trop longues pour être fournies en entier : "
            "lis celles dont tu as besoin avec l'outil lire_note."
        )
    return "\n\n".join(morceaux)


# ---------------------------------------------------------------------------
# Fiche de clôture
# ---------------------------------------------------------------------------

CONSIGNE_FICHE = """
La séance est terminée : produis maintenant la fiche récapitulative de clôture, en suivant
exactement la section « Clôture de séance » et le modèle du mode de conduite ci-dessus.
- Date du jour : {date}.
- Première ligne : le nom de fichier, au format {date}_fiche_[sujet-court].md
- Ensuite, le contenu de la fiche (en-tête YAML compris), dans un seul bloc de code Markdown.
- Appuie-toi uniquement sur la transcription, mes notes et les fiches précédentes fournies.
  N'invente rien. La transcription vient d'une reconnaissance vocale : corrige les mots
  manifestement mal reconnus.
"""

_NOM_FICHIER = re.compile(r"(\d{4}-\d{2}-\d{2}_fiche_[^\s`*/\\:\"<>|?]+?\.md)")
_BLOC_CODE = re.compile(r"```(?:markdown|md)?\s*\n(.*?)\n```", re.S)


def slug(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c)).lower()
    texte = re.sub(r"[^a-z0-9]+", "-", texte).strip("-")
    return texte[:50] or "seance"


def extraire_fiche(reponse: str, date: datetime.date) -> tuple[str, str]:
    """Sépare le nom de fichier et le contenu de la fiche produite par le modèle."""
    contenu = reponse.strip()
    blocs = _BLOC_CODE.findall(contenu)
    if blocs:
        contenu = max(blocs, key=len).strip()
    else:  # pas de bloc de code : on retire une éventuelle ligne de nom de fichier
        contenu = re.sub(r"^\W*" + _NOM_FICHIER.pattern + r"\W*\n", "", contenu).strip()
        contenu = contenu.removeprefix("```markdown").removeprefix("```").removesuffix("```").strip()
    nom = None
    m = _NOM_FICHIER.search(reponse)
    if m:
        nom = m.group(1)
    else:
        titre = re.search(r"^#\s+Séance du \S+\s+[—–-]\s*(.+)$", contenu, re.M)
        nom = f"{date:%Y-%m-%d}_fiche_{slug(titre.group(1) if titre else 'seance')}.md"
    nom = re.sub(r'[<>:"/\\|?*\n]', "-", nom)
    return nom, contenu + "\n"
