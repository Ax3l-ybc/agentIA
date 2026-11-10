"""Lecture du fichier alloia.yaml."""

from __future__ import annotations

import dataclasses
import os
import pathlib

import yaml

from agent.reglages import ErreurReglages, reparer_apostrophes

RACINE = pathlib.Path(__file__).resolve().parent.parent
FICHIER_CONFIG = RACINE / "alloia.yaml"


@dataclasses.dataclass
class Reglages:
    dossier_docs: pathlib.Path
    mode_de_conduite: pathlib.Path = RACINE / "alloia_conduite.md"
    modele_doc: pathlib.Path = RACINE / "alloia_modele_doc.md"
    ollama_adresse: str = "http://127.0.0.1:11434"
    ollama_modele: str = "ministral-3:3b"
    ollama_contexte: int = 8192
    ollama_temperature: float = 0.4
    whisper_modele: str = "small"
    pause_s: float = 0.9
    voix_modele: str = "hf://gradium/phonon-40m-fr"
    voix_nom: str = ""
    voix_vitesse: float = 0.0
    jeton_hf: str = ""
    audio_entree: str = ""
    audio_sortie: str = ""
    micro_coupe_quand_agent_parle: bool = True
    volume: float = 1.0
    parole_pour_decrocher_s: float = 0.6
    relance_apres_s: float = 12.0
    fin_apres_silence_s: float = 60.0
    duree_max_min: float = 25.0
    port: int = 8765


def _chemin(valeur, defaut: pathlib.Path) -> pathlib.Path:
    if not isinstance(valeur, str) or not valeur.strip():
        return defaut
    p = pathlib.Path(os.path.expandvars(os.path.expanduser(valeur.strip())))
    return p if p.is_absolute() else RACINE / p


def charger(fichier: pathlib.Path = FICHIER_CONFIG) -> Reglages:
    if not fichier.exists():
        raise ErreurReglages(f"Fichier de réglages introuvable : {fichier}")
    texte = fichier.read_text(encoding="utf-8-sig")
    try:
        brut = yaml.safe_load(texte) or {}
    except yaml.YAMLError:
        try:
            brut = yaml.safe_load(reparer_apostrophes(texte)) or {}
        except yaml.YAMLError as exc:
            raise ErreurReglages(f"alloia.yaml est mal écrit.\nDétail : {exc}") from exc

    o = brut.get("ollama") or {}
    e = brut.get("ecoute") or {}
    v = brut.get("voix") or {}
    a = brut.get("audio") or {}
    ap = brut.get("appel") or {}
    d = Reglages(dossier_docs=_chemin(brut.get("dossier_docs"), RACINE / "docs"))

    def val(dico, cle, defaut, conv):
        x = dico.get(cle)
        return defaut if x is None or x == "" else conv(x)

    r = Reglages(
        dossier_docs=d.dossier_docs,
        mode_de_conduite=_chemin(brut.get("mode_de_conduite"), d.mode_de_conduite),
        modele_doc=_chemin(brut.get("modele_doc"), d.modele_doc),
        ollama_adresse=str(val(o, "adresse", d.ollama_adresse, str)).rstrip("/"),
        ollama_modele=val(o, "modele", d.ollama_modele, str),
        ollama_contexte=val(o, "taille_contexte", d.ollama_contexte, int),
        ollama_temperature=val(o, "temperature", d.ollama_temperature, float),
        whisper_modele=val(e, "modele_whisper", d.whisper_modele, str),
        pause_s=val(e, "pause_fin_de_phrase_s", d.pause_s, float),
        voix_modele=val(v, "modele", d.voix_modele, str),
        voix_nom=str(v.get("nom") or "").strip(),
        voix_vitesse=val(v, "vitesse", 0.0, float),
        jeton_hf=str(brut.get("jeton_huggingface") or "").strip(),
        audio_entree=str(a.get("entree") or "").strip(),
        audio_sortie=str(a.get("sortie") or "").strip(),
        micro_coupe_quand_agent_parle=bool(a.get("couper_micro_quand_agent_parle", True)),
        volume=val(a, "volume", 1.0, float),
        parole_pour_decrocher_s=val(ap, "parole_pour_decrocher_s", d.parole_pour_decrocher_s, float),
        relance_apres_s=val(ap, "relance_apres_s", d.relance_apres_s, float),
        fin_apres_silence_s=val(ap, "fin_apres_silence_s", d.fin_apres_silence_s, float),
        duree_max_min=val(ap, "duree_max_min", d.duree_max_min, float),
        port=val(brut, "port", d.port, int),
    )
    r.dossier_docs.mkdir(parents=True, exist_ok=True)
    return r
