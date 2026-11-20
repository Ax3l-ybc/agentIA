"""Lecture du fichier config.yaml (réglages techniques : cerveau, voix, page web).

Ce que dit l'assistant (ses consignes, sa voix, ses documents) ne se règle pas ici : chaque
assistant a son fichier dans le dossier « assistants », modifiable depuis la page web ou avec
le Bloc-notes (voir profils.py).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re

import yaml

RACINE = pathlib.Path(__file__).resolve().parent.parent
FICHIER_CONFIG = RACINE / "config.yaml"
DOSSIER_ASSISTANTS = RACINE / "assistants"

# Dossier des données (modèle Gemma 4, conversations), hors du dossier du programme : il
# survit quand on remplace le programme par une nouvelle version.
DOSSIER_DONNEES = pathlib.Path(os.path.expanduser("~")) / "MonAssistant"
# Gemma 4 a peut-être déjà été téléchargé pour le directeur de mémoire : on le réutilise.
DOSSIERS_MODELES = [DOSSIER_DONNEES / "modeles",
                    pathlib.Path(os.path.expanduser("~")) / "DirecteurMemoire" / "modeles"]
MODELE_PAR_DEFAUT = "gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PAR_DEFAUT = "mmproj-F16.gguf"


class ErreurReglages(Exception):
    """Erreur de configuration, avec un message lisible par un non-codeur."""


def trouver_modele(nom: str) -> pathlib.Path:
    """Le fichier `nom` dans le premier dossier de modèles qui le contient (sinon le premier)."""
    for dossier in DOSSIERS_MODELES:
        if (dossier / nom).is_file():
            return dossier / nom
    return DOSSIERS_MODELES[0] / nom


@dataclasses.dataclass
class Reglages:
    # Le cerveau : Gemma 4 E2B dans llama.cpp (il lit le texte et écoute la voix)
    llamacpp_serveur: str = ""  # vide : trouvé automatiquement (installé par winget)
    llamacpp_modele: pathlib.Path = dataclasses.field(default_factory=lambda: trouver_modele(MODELE_PAR_DEFAUT))
    llamacpp_mmproj: pathlib.Path = dataclasses.field(default_factory=lambda: trouver_modele(MMPROJ_PAR_DEFAUT))
    llamacpp_contexte: int = 16384
    llamacpp_couches_gpu: int = 0  # 0 = tout sur le processeur
    llamacpp_port: int = 8090
    temperature: float = 0.6
    # La voix : « kokoro » (libre, y compris en entreprise ; par défaut) ou « phonon »
    # (15 voix françaises, usage non commercial, compte Hugging Face nécessaire)
    voix_moteur: str = "kokoro"
    voix_modele: str = "hf://gradium/phonon-40m-fr"
    voix_vitesse: float = 0.0
    jeton_hf: str = ""
    # La conversation à la voix
    pause_s: float = 1.2
    anticipation_s: float = 0.4
    # La page web
    port: int = 8000
    # Où sont rangés les assistants et les conversations
    dossier_assistants: pathlib.Path = DOSSIER_ASSISTANTS
    dossier_conversations: pathlib.Path = DOSSIER_DONNEES / "conversations"
    dossier_utilisateurs: pathlib.Path = DOSSIER_DONNEES / "utilisateurs"


_LIGNE_APOSTROPHES = re.compile(r"^(\s*[\w-]+:\s*)'(.*)'(\s*(?:#.*)?)$")


def reparer_apostrophes(texte: str) -> str:
    """Double les apostrophes à l'intérieur des valeurs écrites 'comme ceci'."""
    lignes = []
    for ligne in texte.splitlines():
        m = _LIGNE_APOSTROPHES.match(ligne)
        if m and "'" in m.group(2):
            interieur = m.group(2).replace("''", "'").replace("'", "''")
            ligne = f"{m.group(1)}'{interieur}'{m.group(3)}"
        lignes.append(ligne)
    return "\n".join(lignes)


def lire_yaml(texte: str, nom: str) -> dict:
    try:
        return yaml.safe_load(texte) or {}
    except yaml.YAMLError:
        # Cas fréquent : une apostrophe dans le texte (« d'appui ») ferme trop tôt la valeur
        # écrite entre apostrophes. On la répare et on réessaie.
        try:
            return yaml.safe_load(reparer_apostrophes(texte)) or {}
        except yaml.YAMLError as exc:
            raise ErreurReglages(
                f"{nom} est mal écrit. Vérifiez les apostrophes ' ' autour des textes et "
                f"l'alignement des lignes (gardez les espaces en début de ligne).\nDétail : {exc}"
            ) from exc


def _chemin(valeur, defaut: pathlib.Path, base: pathlib.Path = RACINE) -> pathlib.Path:
    if not isinstance(valeur, str) or not valeur.strip():
        return defaut
    p = pathlib.Path(os.path.expandvars(os.path.expanduser(valeur.strip())))
    return p if p.is_absolute() else base / p


def charger(fichier: pathlib.Path = FICHIER_CONFIG) -> Reglages:
    brut = lire_yaml(fichier.read_text(encoding="utf-8-sig"), fichier.name) if fichier.exists() else {}
    cerveau = brut.get("cerveau") or {}
    voix = brut.get("voix") or {}
    conversation = brut.get("conversation") or {}
    d = Reglages()

    def val(dico, cle, defaut, conv):
        x = dico.get(cle)
        return defaut if x is None or x == "" else conv(x)

    def modele(valeur, defaut: pathlib.Path) -> pathlib.Path:
        if isinstance(valeur, str) and valeur.strip() and not os.path.isabs(os.path.expanduser(valeur.strip())):
            return trouver_modele(valeur.strip())
        return _chemin(valeur, defaut)

    return Reglages(
        llamacpp_serveur=str(cerveau.get("serveur") or "").strip(),
        llamacpp_modele=modele(cerveau.get("modele"), d.llamacpp_modele),
        llamacpp_mmproj=modele(cerveau.get("module_audio"), d.llamacpp_mmproj),
        llamacpp_contexte=val(cerveau, "taille_contexte", d.llamacpp_contexte, int),
        llamacpp_couches_gpu=val(cerveau, "couches_gpu", d.llamacpp_couches_gpu, int),
        llamacpp_port=val(cerveau, "port", d.llamacpp_port, int),
        temperature=val(cerveau, "temperature", d.temperature, float),
        voix_moteur=str(voix.get("moteur") or d.voix_moteur).strip().lower(),
        voix_modele=val(voix, "modele", d.voix_modele, str),
        voix_vitesse=val(voix, "vitesse", 0.0, float),
        jeton_hf=str(brut.get("jeton_huggingface") or "").strip(),
        pause_s=val(conversation, "pause_fin_de_phrase_s", d.pause_s, float),
        anticipation_s=val(conversation, "anticipation_s", d.anticipation_s, float),
        port=int(brut.get("port") or 8000),
        dossier_assistants=_chemin(brut.get("dossier_assistants"), d.dossier_assistants),
        dossier_conversations=_chemin(brut.get("dossier_conversations"), d.dossier_conversations),
    )
