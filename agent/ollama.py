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


def telecharger(adresse: str, modele: str) -> None:
    """Télécharge le modèle dans Ollama s'il n'y est pas encore (avec la progression)."""
    if modele not in modeles_installes(adresse):
        log.info("Téléchargement du modèle %s (une seule fois, cela peut être long)…", modele)
        with _requete(adresse, "/api/pull", {"model": modele, "stream": True}, delai=None) as r:
            derniers: dict[str, int] = {}
            for ligne in r:
                etat = json.loads(ligne or b"{}")
                if "error" in etat:
                    raise ErreurOllama(f"Téléchargement impossible : {etat['error']}")
                total, fait = etat.get("total"), etat.get("completed")
                # On n'affiche que les gros fichiers, chacun de 5 % en 5 %.
                if total and fait and total > 50_000_000:
                    pourcent = int(100 * fait / total)
                    cle = etat.get("digest", "")
                    if pourcent >= derniers.get(cle, -5) + 5:
                        log.info(
                            "  … cerveau : %s %% (%.1f / %.1f Go)", pourcent, fait / 1e9, total / 1e9
                        )
                        derniers[cle] = pourcent


def preparer(
    adresse: str,
    modele: str,
    contexte: int,
    nom: str = NOM_MODELE_AGENT,
    temperature: float = 0.6,
) -> str:
    """Télécharge le modèle si besoin et crée une variante à grand contexte.

    Renvoie le nom du modèle à utiliser.
    """
    log.info("Vérification d'Ollama…")
    verifier(adresse)
    telecharger(adresse, modele)
    # Variante du modèle avec une mémoire de travail suffisante pour les notes.
    log.info("Préparation du cerveau « %s »…", nom)
    try:
        with _requete(
            adresse,
            "/api/create",
            {
                "model": nom,
                "from": modele,
                "parameters": {"num_ctx": contexte, "temperature": temperature},
                "stream": False,
            },
            delai=300,
        ) as r:
            r.read()
        return nom
    except (urllib.error.URLError, OSError) as exc:
        log.warning(
            "Impossible de créer la variante « %s » (%s). Utilisation directe de %s.",
            nom,
            exc,
            modele,
        )
        return modele


def precharger(adresse: str, modele: str, contexte: int | None = None) -> None:
    """Charge le modèle en mémoire tout de suite (le premier appel sera rapide)."""
    corps = {"model": modele, "keep_alive": -1}
    if contexte:
        corps["options"] = {"num_ctx": contexte}
    with _requete(adresse, "/api/generate", corps, delai=600) as r:
        r.read()


def discuter(
    adresse: str, modele: str, messages: list[dict], contexte: int, reflexion: bool | None = None
) -> str:
    """Appel simple (non vocal) au modèle, utilisé pour rédiger les fiches."""
    corps = {
        "model": modele,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": contexte, "temperature": 0.3},
    }
    if reflexion is not None:
        corps["think"] = reflexion
    with _requete(adresse, "/api/chat", corps, delai=600) as r:
        reponse = json.load(r)
    return (reponse.get("message") or {}).get("content", "").strip()
