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


def wav_de_test(secondes: float = 1.0) -> bytes:
    """Un court son (16 kHz) pour vérifier au démarrage que l'écoute fonctionne."""
    import numpy as np

    t = np.arange(int(secondes * 16000)) / 16000
    return ollama_audio.wav_16k((0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


class ServeurLlamaCpp:
    """Fait tourner llama-server avec Gemma 4 E2B, et le redémarre s'il s'arrête.

    Si la puce graphique manque de mémoire, il passe au profil suivant (plus économe).
    """

    def __init__(self, r, journal: pathlib.Path):
        self.r = r
        self.journal = journal
        self.profil = 0
        self.processus: subprocess.Popen | None = None
        self.cerveau = CerveauLlamaCpp(f"http://127.0.0.1:{r.llamacpp_port}", r.ollama_temperature)

    # ------------------------------------------------------------------

    def _commande(self, profil: dict) -> list[str]:
        r = self.r
        contexte = min(r.llamacpp_contexte, profil["contexte"] or r.llamacpp_contexte)
        return [
            str(r.llamacpp_serveur),
            "-m", str(r.llamacpp_modele),
            "--mmproj", str(r.llamacpp_mmproj),
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

    def _lire_journal(self, depuis: int) -> str:
        try:
            with open(self.journal, "rb") as f:
                f.seek(depuis)
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def _verifier(self) -> None:
        """Une vraie question avec un court son : l'écoute demande de la mémoire en plus."""
        debut = time.monotonic()
        message = self.cerveau.message_vocal(wav_de_test(), "Dis seulement : prêt.")
        self.cerveau.rediger([message])
        log.info("Écoute vérifiée : première réponse du cerveau en %.1f s.", time.monotonic() - debut)

    def demarrer(self) -> None:
        r = self.r
        for nom, chemin in (("le serveur llama.cpp", r.llamacpp_serveur), ("le modèle", r.llamacpp_modele),
                            ("le module audio", r.llamacpp_mmproj)):
            if not chemin.exists():
                raise ErreurCerveau(f"Fichier introuvable pour {nom} : {chemin}\nRelancez ./jetson/installer.sh.")
        dispo = memoire_disponible_go()
        if dispo is not None:
            log.info("Mémoire disponible avant de charger le cerveau : %.1f Go.", dispo)
            if dispo < 4.5:
                log.warning("C'est peu : le cerveau a besoin d'environ 4 Go. Programmes qui occupent "
                            "le plus de mémoire :\n%s", gros_consommateurs())
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{r.llamacpp_serveur.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        dernier = ""
        while self.profil < len(PROFILS_MEMOIRE):
            profil = PROFILS_MEMOIRE[self.profil]
            vider_cache_memoire()
            depuis = self.journal.stat().st_size if self.journal.exists() else 0
            commande = self._commande(profil)
            log.info("Démarrage de llama.cpp (mode %s, contexte %s ; journal : %s)…",
                     profil["nom"], commande[commande.index("-c") + 1], self.journal)
            self.processus = subprocess.Popen(
                commande, stdout=open(self.journal, "ab"), stderr=subprocess.STDOUT, env=env
            )
            debut = time.monotonic()
            pret = False
            while self.processus.poll() is None:
                if self.cerveau.pret():
                    pret = True
                    break
                if time.monotonic() - debut > 300:
                    self.arreter()
                    raise ErreurCerveau("llama.cpp met plus de 5 minutes à démarrer.")
                time.sleep(1)
            if pret:
                log.info("Cerveau prêt en %.0f s.", time.monotonic() - debut)
                try:
                    self._verifier()
                    return
                except ErreurCerveau as exc:
                    log.warning("Le cerveau n'a pas supporté un message vocal : %s", exc)
                    time.sleep(1)
            texte = self._lire_journal(depuis)
            dernier = texte[-1500:]
            self.arreter()
            manque = any(m in texte for m in MANQUE_MEMOIRE) or (pret and self.processus.poll() is not None)
            if not manque:
                break  # autre erreur : inutile de réessayer
            self.profil += 1
            if self.profil < len(PROFILS_MEMOIRE):
                log.warning("La puce graphique manque de mémoire : nouvel essai en mode %s.",
                            PROFILS_MEMOIRE[self.profil]["nom"])
                time.sleep(2)
        message = f"llama.cpp n'a pas pu démarrer. Fin du journal :\n{dernier}"
        if any(m in dernier for m in MANQUE_MEMOIRE) or self.profil >= len(PROFILS_MEMOIRE):
            message += (
                "\n\nLa puce graphique manque de mémoire, même avec les réglages économes."
                f"\nProgrammes qui occupent le plus de mémoire :\n{gros_consommateurs()}"
                "\nFermez-les (navigateur, autre AlloIA, Ollama…) ou passez sans bureau graphique "
                "(voir le README, « Mémoire de la carte »)."
            )
        raise ErreurCerveau(message)

    def vivant(self) -> bool:
        return self.processus is not None and self.processus.poll() is None and self.cerveau.pret()

    def reparer(self) -> None:
        """Redémarre llama.cpp s'il s'est arrêté (en mode plus économe si c'était la mémoire)."""
        if self.vivant():
            return
        if self.processus is not None and self.processus.poll() is not None:
            fin = self._lire_journal(max(0, self.journal.stat().st_size - 4000))
            log.warning("llama.cpp s'est arrêté. Fin du journal :\n%s", fin[-800:])
            if any(m in fin for m in MANQUE_MEMOIRE) and self.profil < len(PROFILS_MEMOIRE) - 1:
                self.profil += 1
        self.arreter()
        self.demarrer()

    def arreter(self) -> None:
        if self.processus is not None and self.processus.poll() is None:
            self.processus.terminate()
            try:
                self.processus.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.processus.kill()


def lancer_serveur_llamacpp(r, journal: pathlib.Path) -> ServeurLlamaCpp:
    serveur = ServeurLlamaCpp(r, journal)
    serveur.demarrer()
    return serveur
