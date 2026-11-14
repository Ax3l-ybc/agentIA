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


# Dossier des gros fichiers (modèle Gemma 4), hors du dossier du projet : il survit
# quand on remplace le projet par une nouvelle version.
DOSSIER_DONNEES = pathlib.Path(os.path.expanduser("~")) / "DirecteurMemoire"
MODELE_PAR_DEFAUT = "gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PAR_DEFAUT = "mmproj-F16.gguf"


@dataclasses.dataclass
class Reglages:
    dossier_notes: pathlib.Path
    dossier_recaps: pathlib.Path
    sujet_memoire: str = ""
    # Le cerveau qui écoute : Gemma 4 E2B dans llama.cpp
    llamacpp_serveur: str = ""  # vide : trouvé automatiquement (installé par winget)
    llamacpp_modele: pathlib.Path = DOSSIER_DONNEES / "modeles" / MODELE_PAR_DEFAUT
    llamacpp_mmproj: pathlib.Path = DOSSIER_DONNEES / "modeles" / MMPROJ_PAR_DEFAUT
    llamacpp_contexte: int = 16384
    llamacpp_couches_gpu: int = 0  # 0 = tout sur le processeur
    llamacpp_port: int = 8090
    temperature: float = 0.6
    # La voix
    voix_modele: str = "hf://gradium/phonon-40m-fr"
    voix_nom: str = ""
    voix_vitesse: float = 0.0
    jeton_hf: str = ""
    # La conversation
    pause_s: float = 1.2
    silence_relance_s: float = 20.0
    seuil_notes_entieres: int = 20000
    port: int = 8000
    mode_de_conduite: pathlib.Path | None = None  # fichier du prompt de l'agent


def _chemin(valeur: object, nom: str) -> pathlib.Path:
    if not isinstance(valeur, str) or not valeur.strip():
        raise ErreurReglages(f"Le réglage « {nom} » est vide dans config.yaml.")
    return pathlib.Path(os.path.expandvars(os.path.expanduser(valeur.strip())))


def _chemin_mode(valeur: object) -> pathlib.Path:
    """Fichier du mode de conduite ; un chemin relatif part du dossier de l'agent."""
    if not isinstance(valeur, str) or not valeur.strip():
        valeur = "mode_de_conduite.md"
    chemin = pathlib.Path(os.path.expandvars(os.path.expanduser(valeur.strip())))
    return chemin if chemin.is_absolute() else RACINE / chemin


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

    cerveau = brut.get("cerveau") or {}
    voix = brut.get("voix") or {}
    conversation = brut.get("conversation") or {}
    d = Reglages(dossier_notes=pathlib.Path("."), dossier_recaps=pathlib.Path("."))

    def val(dico, cle, defaut, conv):
        x = dico.get(cle)
        return defaut if x is None or x == "" else conv(x)

    def fichier_modele(valeur, defaut: pathlib.Path) -> pathlib.Path:
        if not isinstance(valeur, str) or not valeur.strip():
            return defaut
        p = pathlib.Path(os.path.expandvars(os.path.expanduser(valeur.strip())))
        return p if p.is_absolute() else DOSSIER_DONNEES / "modeles" / p

    # Les anciens réglages (ollama, transcription, materiel…) sont simplement ignorés.
    r = Reglages(
        dossier_notes=_chemin(brut.get("dossier_notes"), "dossier_notes"),
        dossier_recaps=_chemin(brut.get("dossier_recaps"), "dossier_recaps"),
        sujet_memoire=str(brut.get("sujet_memoire") or "").strip(),
        llamacpp_serveur=str(cerveau.get("serveur") or "").strip(),
        llamacpp_modele=fichier_modele(cerveau.get("modele"), d.llamacpp_modele),
        llamacpp_mmproj=fichier_modele(cerveau.get("module_audio"), d.llamacpp_mmproj),
        llamacpp_contexte=val(cerveau, "taille_contexte", d.llamacpp_contexte, int),
        llamacpp_couches_gpu=val(cerveau, "couches_gpu", d.llamacpp_couches_gpu, int),
        llamacpp_port=val(cerveau, "port", d.llamacpp_port, int),
        temperature=val(cerveau, "temperature", d.temperature, float),
        voix_modele=val(voix, "modele", d.voix_modele, str),
        voix_nom=str(voix.get("nom") or "").strip(),
        voix_vitesse=val(voix, "vitesse", 0.0, float),
        jeton_hf=str(brut.get("jeton_huggingface") or "").strip(),
        pause_s=val(conversation, "pause_fin_de_phrase_s", d.pause_s, float),
        silence_relance_s=val(conversation, "silence_avant_relance_s", d.silence_relance_s, float),
        seuil_notes_entieres=val(conversation, "notes_en_entier_si_moins_de", d.seuil_notes_entieres, int),
        port=int(brut.get("port") or 8000),
        mode_de_conduite=_chemin_mode(brut.get("mode_de_conduite")),
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
