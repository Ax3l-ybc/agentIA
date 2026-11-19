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
import socket
import logging
import os
import pathlib
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Iterator

from . import ollama_audio

log = logging.getLogger("alloia.cerveau")

ErreurCerveau = ollama_audio.ErreurOllama

# llama.cpp garde en mémoire ce qu'il a déjà lu (consignes, outils, historique de l'appel) :
# au tour suivant, il ne lit que la nouveauté. Mais une retranscription ou la rédaction de la
# documentation effacerait cette mémoire : on la sauvegarde juste avant (quelques centièmes de
# seconde, en mémoire vive) et on la restaure juste après.
# (Deux « espaces » séparés, -np 2, feraient la même chose, mais llama.cpp lit alors chaque
# jeton environ 3 fois plus lentement : mesuré avec Gemma 4 E2B.)
FICHIER_MEMOIRE = "conversation.bin"


def dossier_memoire() -> pathlib.Path:
    """Où llama.cpp sauvegarde sa mémoire : en mémoire vive (/dev/shm) si possible."""
    shm = pathlib.Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm / f"alloia-{os.getuid()}"
    return pathlib.Path(os.path.expanduser("~/AlloIA/memoire"))

CONSIGNE_TRANSCRIPTION = (
    "Transcris mot pour mot ce message vocal, en français. "
    "Réponds uniquement par la transcription, sans rien ajouter."
)


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
        # Une requête à la fois : une retranscription ne doit pas se glisser entre la
        # sauvegarde et la restauration de la mémoire de la conversation.
        self.verrou = threading.Lock()
        self._deja_signale = False

    def _memoire(self, action: str) -> bool:
        try:
            with self._post(f"/slots/0?action={action}", {"filename": FICHIER_MEMOIRE}, delai=60) as r:
                json.load(r)
            return True
        except (ErreurCerveau, ValueError) as exc:
            if not self._deja_signale:  # pas grave : le tour suivant relira tout l'historique
                log.warning("Mémoire de la conversation non %s : %s",
                            "sauvegardée" if action == "save" else "restaurée", exc)
                self._deja_signale = True
            return False

    def _a_part(self, corps: dict, delai: float) -> dict:
        """Requête hors conversation (retranscription, doc) qui n'efface pas la conversation."""
        with self.verrou:
            sauvee = self._memoire("save")
            try:
                with self._post("/v1/chat/completions", corps, delai=delai) as r:
                    return json.load(r)
            finally:
                if sauvee:
                    self._memoire("restore")

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

    def interrompre(self, arret: threading.Event) -> None:
        """Coupe net la réponse en cours si c'est celle marquée par `arret` (envoi anticipé
        abandonné) : llama.cpp est aussitôt libre pour la vraie phrase."""
        arret.set()
        en_cours = getattr(self, "_en_cours", None)
        if en_cours is not None and en_cours[0] is arret:
            try:
                en_cours[1].fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass

    def discuter_en_flux(self, messages: list[dict], outils: list[dict] | None = None,
                         max_mots: int | None = None, arret: threading.Event | None = None) -> Iterator[dict]:
        corps = {
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "cache_prompt": True,  # réutilise les consignes et l'historique déjà calculés
        }
        if outils:
            corps["tools"] = outils
        if max_mots:
            corps["max_tokens"] = max_mots
        appels: dict[int, dict] = {}
        with self.verrou:
            if arret is not None and arret.is_set():
                return  # abandonnée avant même d'avoir commencé
            try:
                yield from self._flux(corps, appels, max_mots, arret)
            except ErreurCerveau:
                if arret is not None and arret.is_set():
                    return  # coupée exprès
                raise
            except (OSError, ValueError):
                if arret is not None and arret.is_set():
                    return
                raise
        for a in appels.values():
            try:
                arguments = json.loads(a["args"]) if a["args"] else {}
            except json.JSONDecodeError:
                arguments = {}
            yield {"outil": a["nom"], "arguments": arguments}

    def _flux(self, corps: dict, appels: dict, max_mots, arret) -> Iterator[dict]:
        with self._post("/v1/chat/completions", corps) as r:
            self._en_cours = (arret, r)
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
                temps = morceau.get("timings")
                if temps and temps.get("prompt_ms") is not None and not max_mots:
                    # Ce que llama.cpp a dû (re)lire avant de répondre : consignes, historique,
                    # message vocal. Peu de jetons = la mémoire des tours précédents a servi.
                    log.info("  lecture : %d jetons lus en %.1f s", temps.get("prompt_n", 0),
                             temps["prompt_ms"] / 1000)
                for choix in morceau.get("choices") or []:
                    delta = choix.get("delta") or {}
                    if delta.get("content"):
                        yield {"texte": delta["content"]}
                    for tc in delta.get("tool_calls") or []:
                        a = appels.setdefault(tc.get("index", 0), {"nom": "", "args": ""})
                        fonction = tc.get("function") or {}
                        a["nom"] += fonction.get("name") or ""
                        a["args"] += fonction.get("arguments") or ""
                if arret is not None and arret.is_set():
                    return
        self._en_cours = None

    def transcrire(self, wav: bytes) -> str:
        """Retranscrit un message vocal dans l'espace annexe (l'appel en cours reste en mémoire)."""
        texte = self.rediger([self.message_vocal(wav, CONSIGNE_TRANSCRIPTION)], temperature=0.0)
        return texte.strip().strip('"«»').strip()

    def rediger(self, messages: list[dict], temperature: float = 0.3) -> str:
        corps = {"messages": messages, "stream": False, "temperature": temperature, "cache_prompt": False}
        reponse = self._a_part(corps, delai=900)
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
    # (La taille de la mémoire de travail coûte peu avec Gemma 4 : 6 Mo par tranche de 1 000
    # jetons. La limiter à 2 048 faisait échouer les appels après une dizaine d'échanges.)
    {"nom": "économe", "contexte": None, "args": ["-b", "256", "-ub", "128"]},
    {"nom": "très économe (module audio hors de la puce graphique)", "contexte": None,
     "args": ["-b", "256", "-ub", "128", "--no-mmproj-offload"]},
]
# Où commencer selon le réglage « mode » (alloia.yaml) ; « auto » essaie du plus rapide au
# plus économe. Sur la Jetson Orin Nano avec bureau graphique, seul le très économe tient.
PROFIL_DE_DEPART = {"auto": 0, "normal": 0, "econome": 1, "tres_econome": 2}
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
        self.profil = PROFIL_DE_DEPART.get(getattr(r, "llamacpp_mode", "auto"), len(PROFILS_MEMOIRE) - 1)
        self.processus: subprocess.Popen | None = None
        self.dossier_memoire = dossier_memoire()
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
            "--slot-save-path", str(self.dossier_memoire),  # sauvegarde de la mémoire (voir plus haut)
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
        self.dossier_memoire.mkdir(parents=True, exist_ok=True)
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
        (self.dossier_memoire / FICHIER_MEMOIRE).unlink(missing_ok=True)


def lancer_serveur_llamacpp(r, journal: pathlib.Path) -> ServeurLlamaCpp:
    serveur = ServeurLlamaCpp(r, journal)
    serveur.demarrer()
    return serveur
