"""Petits utilitaires pour parler à Ollama (le « cerveau » local)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

log = logging.getLogger("agent.ollama")

NOM_MODELE_AGENT = "directeur-memoire"


class ErreurOllama(Exception):
    pass


def _requete(adresse: str, chemin: str, corps: dict | None = None, delai: float = 30.0):
    donnees = json.dumps(corps).encode("utf-8") if corps is not None else None
    req = urllib.request.Request(
        adresse + chemin,
        data=donnees,
        headers={"Content-Type": "application/json"},
        method="POST" if corps is not None else "GET",
    )
    return urllib.request.urlopen(req, timeout=delai)


def _repond(adresse: str) -> bool:
    try:
        with _requete(adresse, "/api/version", delai=5) as r:
            r.read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def verifier(adresse: str) -> None:
    """Vérifie qu'Ollama tourne ; essaie de le démarrer sinon."""
    if _repond(adresse):
        return
    executable = shutil.which("ollama")
    if executable:
        log.info("Démarrage d'Ollama…")
        options = {}
        if sys.platform == "win32":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **options,
        )
        for _ in range(30):
            time.sleep(1)
            if _repond(adresse):
                return
    raise ErreurOllama(
        "Ollama ne répond pas. Vérifiez qu'Ollama est installé (https://ollama.com/download) "
        "et lancé (icône de lama près de l'horloge), puis relancez."
    )


def modeles_installes(adresse: str) -> set[str]:
    with _requete(adresse, "/api/tags") as r:
        donnees = json.load(r)
    noms = set()
    for m in donnees.get("models", []):
        noms.add(m["name"])
        if m["name"].endswith(":latest"):
            noms.add(m["name"].removesuffix(":latest"))
    return noms


def preparer(adresse: str, modele: str, contexte: int) -> str:
    """Télécharge le modèle si besoin et crée une variante à grand contexte.

    Renvoie le nom du modèle à utiliser.
    """
    verifier(adresse)
    if modele not in modeles_installes(adresse):
        log.info("Téléchargement du modèle %s (une seule fois, cela peut être long)…", modele)
        with _requete(adresse, "/api/pull", {"model": modele, "stream": True}, delai=None) as r:
            dernier = -10
            for ligne in r:
                etat = json.loads(ligne or b"{}")
                if "error" in etat:
                    raise ErreurOllama(f"Téléchargement impossible : {etat['error']}")
                total, fait = etat.get("total"), etat.get("completed")
                if total and fait:
                    pourcent = int(100 * fait / total)
                    if pourcent >= dernier + 10:
                        log.info("  … %s %%", pourcent)
                        dernier = pourcent
    # Variante du modèle avec une mémoire de travail suffisante pour les notes.
    try:
        with _requete(
            adresse,
            "/api/create",
            {
                "model": NOM_MODELE_AGENT,
                "from": modele,
                "parameters": {"num_ctx": contexte, "temperature": 0.6},
                "stream": False,
            },
            delai=300,
        ) as r:
            r.read()
        return NOM_MODELE_AGENT
    except (urllib.error.URLError, OSError) as exc:
        log.warning(
            "Impossible de créer la variante « %s » (%s). Utilisation directe de %s.",
            NOM_MODELE_AGENT,
            exc,
            modele,
        )
        return modele


def discuter(adresse: str, modele: str, messages: list[dict], contexte: int) -> str:
    """Appel simple (non vocal) au modèle, utilisé pour rédiger les fiches."""
    corps = {
        "model": modele,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": contexte, "temperature": 0.3},
    }
    with _requete(adresse, "/api/chat", corps, delai=600) as r:
        reponse = json.load(r)
    return (reponse.get("message") or {}).get("content", "").strip()
