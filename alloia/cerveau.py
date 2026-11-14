"""Le cerveau qui écoute : Gemma 4 dans llama.cpp (rapide) ou dans Ollama.

Les deux offrent la même interface :
- `message_vocal(wav, consigne)` : message « maker » contenant l'enregistrement ;
- `discuter_en_flux(messages, outils)` : réponse en flux ({"texte"} ou {"outil"}) ;
- `transcrire(wav)` : retranscription d'un message vocal ;
- `rediger(messages)` : réponse complète (rédaction de la documentation).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
from typing import Iterator

from . import ollama_audio

log = logging.getLogger("alloia.cerveau")

ErreurCerveau = ollama_audio.ErreurOllama


class CerveauOllama:
    nom = "Ollama"

    def __init__(self, adresse: str, modele: str, contexte: int, temperature: float, reflexion: bool):
        self.adresse, self.modele = adresse, modele
        self.contexte, self.temperature, self.reflexion = contexte, temperature, reflexion

    def message_vocal(self, wav: bytes, consigne: str) -> dict:
        return ollama_audio.message_vocal(wav, consigne)

    def discuter_en_flux(self, messages: list[dict], outils: list[dict] | None = None) -> Iterator[dict]:
        yield from ollama_audio.discuter_en_flux(
            self.adresse, self.modele, messages, outils, self.contexte, self.temperature, self.reflexion
        )

    def transcrire(self, wav: bytes) -> str:
        return ollama_audio.transcrire(self.adresse, self.modele, wav)

    def rediger(self, messages: list[dict]) -> str:
        from agent import ollama

        return ollama.discuter(self.adresse, self.modele, messages, self.contexte, self.reflexion)


class CerveauLlamaCpp:
    """Serveur llama.cpp (API compatible OpenAI), lancé par AlloIA sur la Jetson."""

    nom = "llama.cpp"

    def __init__(self, adresse: str, temperature: float = 0.4):
        self.adresse = adresse.rstrip("/")
        self.temperature = temperature

    # ------------------------------------------------------------ requêtes

    def _post(self, chemin: str, corps: dict, delai: float = 600):
        req = urllib.request.Request(
            self.adresse + chemin, json.dumps(corps).encode(), {"Content-Type": "application/json"}
        )
        try:
            return urllib.request.urlopen(req, timeout=delai)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ErreurCerveau(f"llama.cpp a refusé la requête ({exc.code}) : {detail}") from exc
        except OSError as exc:
            raise ErreurCerveau(f"llama.cpp ne répond pas : {exc}") from exc

    def message_vocal(self, wav: bytes, consigne: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": consigne},
                {"type": "input_audio",
                 "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"}},
            ],
        }

    def discuter_en_flux(self, messages: list[dict], outils: list[dict] | None = None) -> Iterator[dict]:
        corps = {
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "cache_prompt": True,  # réutilise les consignes et l'historique déjà calculés
        }
        if outils:
            corps["tools"] = outils
        appels: dict[int, dict] = {}
        with self._post("/v1/chat/completions", corps) as r:
            for ligne in r:
                ligne = ligne.decode("utf-8", "replace").strip()
                if not ligne.startswith("data:"):
                    continue
                donnees = ligne[5:].strip()
                if donnees == "[DONE]":
                    break
                morceau = json.loads(donnees)
                if "error" in morceau:
                    raise ErreurCerveau(str(morceau["error"]))
                for choix in morceau.get("choices") or []:
                    delta = choix.get("delta") or {}
                    if delta.get("content"):
                        yield {"texte": delta["content"]}
                    for tc in delta.get("tool_calls") or []:
                        a = appels.setdefault(tc.get("index", 0), {"nom": "", "args": ""})
                        fonction = tc.get("function") or {}
                        a["nom"] += fonction.get("name") or ""
                        a["args"] += fonction.get("arguments") or ""
        for a in appels.values():
            try:
                arguments = json.loads(a["args"]) if a["args"] else {}
            except json.JSONDecodeError:
                arguments = {}
            yield {"outil": a["nom"], "arguments": arguments}

    def transcrire(self, wav: bytes) -> str:
        # Même requête que pour Ollama (point d'accès /v1/audio/transcriptions).
        return ollama_audio.transcrire(self.adresse, "gemma", wav)

    def rediger(self, messages: list[dict]) -> str:
        corps = {"messages": messages, "stream": False, "temperature": 0.3, "cache_prompt": False}
        with self._post("/v1/chat/completions", corps, delai=900) as r:
            reponse = json.load(r)
        return (reponse["choices"][0]["message"].get("content") or "").strip()

    def pret(self) -> bool:
        try:
            with urllib.request.urlopen(self.adresse + "/health", timeout=2) as r:
                return r.status == 200
        except (OSError, urllib.error.URLError):
            return False


# Réglages essayés dans l'ordre si la puce graphique manque de mémoire au démarrage.
PROFILS_MEMOIRE = [
    {"nom": "normal", "contexte": None, "args": ["-b", "512", "-ub", "256"]},
    {"nom": "économe", "contexte": 2048, "args": ["-b", "256", "-ub", "128"]},
    {"nom": "très économe (module audio hors de la puce graphique)", "contexte": 2048,
     "args": ["-b", "256", "-ub", "128", "--no-mmproj-offload"]},
]
MANQUE_MEMOIRE = ("out of memory", "NvMapMem", "failed to allocate", "cudaMalloc failed")


def memoire_disponible_go() -> float | None:
    try:
        for ligne in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if ligne.startswith("MemAvailable:"):
                return int(ligne.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def gros_consommateurs(n: int = 6) -> str:
    """Les programmes qui occupent le plus de mémoire (pour aider au diagnostic)."""
    try:
        sortie = subprocess.run(
            ["ps", "-eo", "rss=,comm=", "--sort=-rss"], capture_output=True, text=True, timeout=5
        ).stdout.splitlines()[:n]
    except (OSError, subprocess.SubprocessError):
        return ""
    lignes = []
    for ligne in sortie:
        rss, _, nom = ligne.strip().partition(" ")
        if rss.isdigit():
            lignes.append(f"    {int(rss) / 1024 / 1024:5.2f} Go  {nom.strip()}")
    return "\n".join(lignes)


def vider_cache_memoire() -> None:
    """Vide le cache des fichiers (autorisé sans mot de passe par jetson/installer.sh)."""
    try:
        subprocess.run(["sudo", "-n", "/usr/local/sbin/alloia-vider-cache"],
                       capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


def lancer_serveur_llamacpp(r, journal: pathlib.Path) -> subprocess.Popen:
    """Démarre llama-server avec Gemma 4 E2B (puce graphique), et attend qu'il soit prêt.

    Si la puce graphique manque de mémoire, réessaie avec des réglages plus économes.
    """
    serveur, modele, mmproj = r.llamacpp_serveur, r.llamacpp_modele, r.llamacpp_mmproj
    for nom, chemin in (("le serveur llama.cpp", serveur), ("le modèle", modele), ("le module audio", mmproj)):
        if not chemin.exists():
            raise ErreurCerveau(
                f"Fichier introuvable pour {nom} : {chemin}\nRelancez ./jetson/installer.sh."
            )
    dispo = memoire_disponible_go()
    if dispo is not None:
        log.info("Mémoire disponible avant de charger le cerveau : %.1f Go.", dispo)
        if dispo < 4.5:
            log.warning(
                "C'est peu : le cerveau a besoin d'environ 4 Go. Programmes qui occupent le plus "
                "de mémoire :\n%s", gros_consommateurs()
            )
    journal.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{serveur.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    cerveau = CerveauLlamaCpp(f"http://127.0.0.1:{r.llamacpp_port}", r.ollama_temperature)
    derniere_fin = ""
    for i, profil in enumerate(PROFILS_MEMOIRE):
        if i:
            log.warning("La puce graphique manque de mémoire : nouvel essai en mode %s.", profil["nom"])
            vider_cache_memoire()
            time.sleep(2)
        contexte = min(r.llamacpp_contexte, profil["contexte"] or r.llamacpp_contexte)
        commande = [
            str(serveur),
            "-m", str(modele),
            "--mmproj", str(mmproj),
            "-c", str(contexte),
            "-ngl", "99",              # tout le modèle sur la puce graphique
            "--reasoning", "off",      # pas de « réflexion » silencieuse avant de répondre
            "--jinja",                 # modèle de conversation de Gemma 4 (outils)
            "-fa", "on",               # attention plus rapide et plus économe en mémoire
            "--cache-ram", "0",        # pas de cache de conversations en mémoire vive (8 Go sinon)
            "--host", "127.0.0.1",
            "--port", str(r.llamacpp_port),
            "-np", "1",                # une seule conversation à la fois
            *profil["args"],
        ]
        debut_journal = journal.stat().st_size if journal.exists() else 0
        log.info("Démarrage de llama.cpp (mode %s, contexte %d ; journal : %s)…",
                 profil["nom"], contexte, journal)
        processus = subprocess.Popen(
            commande, stdout=open(journal, "ab"), stderr=subprocess.STDOUT, env=env
        )
        debut = time.monotonic()
        while not cerveau.pret():
            if processus.poll() is not None:
                break
            if time.monotonic() - debut > 300:
                processus.terminate()
                raise ErreurCerveau("llama.cpp met plus de 5 minutes à démarrer.")
            time.sleep(1)
        else:
            log.info("Cerveau prêt en %.0f s.", time.monotonic() - debut)
            return processus
        with open(journal, "rb") as f:
            f.seek(debut_journal)
            texte = f.read().decode("utf-8", "replace")
        derniere_fin = texte[-1500:]
        if not any(m in texte for m in MANQUE_MEMOIRE):
            break  # autre erreur : inutile de réessayer
    message = f"llama.cpp s'est arrêté au démarrage. Fin du journal :\n{derniere_fin}"
    if any(m in derniere_fin for m in MANQUE_MEMOIRE):
        message += (
            "\n\nLa puce graphique manque de mémoire, même avec les réglages économes."
            f"\nProgrammes qui occupent le plus de mémoire :\n{gros_consommateurs()}"
            "\nFermez-les (navigateur, autre AlloIA, Ollama…) ou passez sans bureau graphique "
            "(voir le README, « Mémoire de la carte »)."
        )
    raise ErreurCerveau(message)
