"""Lecture et vérification du fichier config.yaml."""

from __future__ import annotations

import dataclasses
import os
import re
import pathlib

import yaml

RACINE = pathlib.Path(__file__).resolve().parent.parent
FICHIER_CONFIG = RACINE / "config.yaml"


class ErreurReglages(Exception):
    """Erreur de configuration, avec un message lisible par un non-codeur."""


@dataclasses.dataclass
class Reglages:
    dossier_notes: pathlib.Path
    dossier_recaps: pathlib.Path
    sujet_memoire: str = ""
    ollama_adresse: str = "http://127.0.0.1:11434"
    ollama_modele: str = "qwen2.5:7b"
    ollama_contexte: int = 16384
    voix_modele: str = "hf://gradium/phonon-40m-fr"
    voix_nom: str = ""
    voix_vitesse: float = 0.0
    stt_modele: str = "kyutai/stt-1b-en_fr-candle"
    jeton_hf: str = ""
    materiel: str = "auto"
    silence_relance_s: float = 15.0
    seuil_notes_entieres: int = 20000
    port: int = 8000


def _chemin(valeur: object, nom: str) -> pathlib.Path:
    if not isinstance(valeur, str) or not valeur.strip():
        raise ErreurReglages(f"Le réglage « {nom} » est vide dans config.yaml.")
    return pathlib.Path(os.path.expandvars(os.path.expanduser(valeur.strip())))


def est_dans(enfant: pathlib.Path, parent: pathlib.Path) -> bool:
    """Vrai si `enfant` est `parent` lui-même ou se trouve à l'intérieur."""
    enfant, parent = enfant.resolve(), parent.resolve()
    return enfant == parent or parent in enfant.parents


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


def charger(fichier: pathlib.Path = FICHIER_CONFIG) -> Reglages:
    if not fichier.exists():
        raise ErreurReglages(f"Fichier de réglages introuvable : {fichier}")
    texte = fichier.read_text(encoding="utf-8-sig")
    try:
        brut = yaml.safe_load(texte) or {}
    except yaml.YAMLError:
        # Cas fréquent : une apostrophe dans le texte (« d'appui ») ferme trop tôt
        # la valeur écrite entre apostrophes. On la répare et on réessaie.
        try:
            brut = yaml.safe_load(reparer_apostrophes(texte)) or {}
        except yaml.YAMLError as exc:
            raise ErreurReglages(
                "config.yaml est mal écrit. Vérifiez les apostrophes ' ' autour des "
                "chemins et l'alignement des lignes (gardez les espaces en début de ligne).\n"
                f"Détail : {exc}"
            ) from exc

    ollama = brut.get("ollama") or {}
    voix = brut.get("voix") or {}
    transcription = brut.get("transcription") or {}
    conversation = brut.get("conversation") or {}

    r = Reglages(
        dossier_notes=_chemin(brut.get("dossier_notes"), "dossier_notes"),
        dossier_recaps=_chemin(brut.get("dossier_recaps"), "dossier_recaps"),
        sujet_memoire=str(brut.get("sujet_memoire") or "").strip(),
        ollama_adresse=str(ollama.get("adresse") or Reglages.ollama_adresse).rstrip("/"),
        ollama_modele=str(ollama.get("modele") or Reglages.ollama_modele),
        ollama_contexte=int(ollama.get("taille_contexte") or Reglages.ollama_contexte),
        voix_modele=str(voix.get("modele") or Reglages.voix_modele),
        voix_nom=str(voix.get("nom") or "").strip(),
        voix_vitesse=float(voix.get("vitesse") or 0.0),
        stt_modele=str(transcription.get("modele") or Reglages.stt_modele),
        jeton_hf=str(brut.get("jeton_huggingface") or "").strip(),
        materiel=str(brut.get("materiel") or "auto").lower(),
        silence_relance_s=float(
            conversation.get("silence_avant_relance_s") or Reglages.silence_relance_s
        ),
        seuil_notes_entieres=int(
            conversation.get("notes_en_entier_si_moins_de")
            if conversation.get("notes_en_entier_si_moins_de") is not None
            else Reglages.seuil_notes_entieres
        ),
        port=int(brut.get("port") or 8000),
    )
    verifier_dossiers(r)
    return r


def verifier_dossiers(r: Reglages) -> None:
    if not r.dossier_notes.is_dir():
        raise ErreurReglages(
            f"Le dossier des notes n'existe pas : {r.dossier_notes}\n"
            "Corrigez « dossier_notes » dans config.yaml."
        )
    if est_dans(r.dossier_recaps, r.dossier_notes):
        raise ErreurReglages(
            "Le dossier des recaps ne doit pas être à l'intérieur du dossier des notes "
            "(l'agent ne doit jamais rien écrire dans vos notes).\n"
            "Choisissez un autre « dossier_recaps » dans config.yaml."
        )
    r.dossier_recaps.mkdir(parents=True, exist_ok=True)
