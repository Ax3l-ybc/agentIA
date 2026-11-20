"""Le cerveau qui écoute : Gemma 4 E2B dans llama.cpp, sur l'ordinateur.

Gemma 4 lit le texte et comprend directement l'audio : chaque phrase dite à voix haute lui est envoyée telle
quelle, sans transcription préalable. llama.cpp garde en mémoire ce qu'il a déjà lu
(consignes, notes, fiches, historique) : à chaque phrase, il ne lit que la nouveauté.
Une retranscription ou la rédaction d'une fiche effacerait cette mémoire : on la sauvegarde
juste avant et on la restaure juste après (quelques centièmes de seconde).
(Deux « espaces » séparés, -np 2, feraient la même chose, mais llama.cpp lit alors chaque
jeton 2 à 3 fois plus lentement : mesuré avec Gemma 4 E2B.)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from typing import Iterator

import numpy as np

log = logging.getLogger("assistant.cerveau")

FICHIER_MEMOIRE = "conversation.bin"

CONSIGNE_TRANSCRIPTION = (
    "Transcris mot pour mot ce message vocal, en français. "
    "Réponds uniquement par la transcription, sans rien ajouter."
)


class ErreurCerveau(Exception):
    pass


def wav_16k(audio: np.ndarray, taux: int = 16000) -> bytes:
    """Encode un signal float32 mono en WAV 16 bits."""
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(taux)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return tampon.getvalue()


class CerveauLlamaCpp:
    """Client du serveur llama.cpp (API compatible OpenAI)."""

    def __init__(self, adresse: str, temperature: float = 0.6):
        self.adresse = adresse.rstrip("/")
        self.temperature = temperature
        # Une requête à la fois : une retranscription ne doit pas se glisser entre la
        # sauvegarde et la restauration de la mémoire de la conversation.
        self.verrou = threading.Lock()
        self._deja_signale = False
        self._en_cours = None

    def _memoire(self, action: str) -> bool:
        try:
            with self._post(f"/slots/0?action={action}", {"filename": FICHIER_MEMOIRE}, delai=60) as r:
                json.load(r)
            return True
        except (ErreurCerveau, ValueError) as exc:
            if not self._deja_signale:  # pas grave : le tour suivant relira tout
                log.warning("Mémoire de la conversation non %s : %s",
                            "sauvegardée" if action == "save" else "restaurée", exc)
                self._deja_signale = True
            return False

    def interrompre(self, arret: threading.Event) -> None:
        """Coupe net la réponse en cours si c'est celle marquée par `arret` (réponse préparée
        à l'avance puis abandonnée) : llama.cpp est aussitôt libre pour la vraie phrase."""
        arret.set()
        en_cours = self._en_cours
        if en_cours is not None and en_cours[0] is arret:
            try:
                en_cours[1].fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass

    def _post(self, chemin: str, corps: dict, delai: float = 900):
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
        """Message de l'utilisateur contenant l'enregistrement de sa phrase."""
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": consigne},
                {"type": "input_audio",
                 "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"}},
            ],
        }

    def discuter_en_flux(self, messages: list[dict], outils: list[dict] | None = None,
                         max_mots: int | None = None,
                         arret: threading.Event | None = None) -> Iterator[dict]:
        """Réponse en flux : {"texte": …} au fil de l'eau, puis {"outil", "arguments", "id"}."""
        corps = {
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "cache_prompt": True,  # réutilise les consignes et l'historique déjà lus
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
            except (ErreurCerveau, OSError, ValueError):
                if arret is not None and arret.is_set():
                    return  # coupée exprès
                raise
            finally:
                self._en_cours = None
        for i, a in sorted(appels.items()):
            try:
                arguments = json.loads(a["args"]) if a["args"] else {}
            except json.JSONDecodeError:
                arguments = {}
            yield {"outil": a["nom"], "arguments": arguments, "id": a["id"] or f"appel_{i}"}

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
                    # Ce que llama.cpp a dû lire avant de répondre (le message vocal et ce qui a
                    # changé depuis le tour précédent) : peu de jetons = sa mémoire a servi.
                    log.info("  lecture : %d jetons lus en %.1f s", temps.get("prompt_n", 0),
                             temps["prompt_ms"] / 1000)
                for choix in morceau.get("choices") or []:
                    delta = choix.get("delta") or {}
                    if delta.get("content"):
                        yield {"texte": delta["content"]}
                    for tc in delta.get("tool_calls") or []:
                        a = appels.setdefault(tc.get("index", 0), {"nom": "", "args": "", "id": ""})
                        fonction = tc.get("function") or {}
                        a["nom"] += fonction.get("name") or ""
                        a["args"] += fonction.get("arguments") or ""
                        a["id"] = tc.get("id") or a["id"]
                if arret is not None and arret.is_set():
                    return

    def rediger(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Réponse complète (fiches, retranscriptions), sans effacer la mémoire de la séance."""
        corps = {"messages": messages, "stream": False, "temperature": temperature, "cache_prompt": False}
        with self.verrou:
            sauvee = self._memoire("save")
            try:
                with self._post("/v1/chat/completions", corps) as r:
                    reponse = json.load(r)
            finally:
                if sauvee:
                    self._memoire("restore")
        return (reponse["choices"][0]["message"].get("content") or "").strip()

    def transcrire(self, wav: bytes) -> str:
        """Retranscrit un message vocal (pour l'historique et la fiche)."""
        texte = self.rediger([self.message_vocal(wav, CONSIGNE_TRANSCRIPTION)], temperature=0.0)
        return texte.strip().strip('"«»').strip()

    def pret(self) -> bool:
        try:
            with urllib.request.urlopen(self.adresse + "/health", timeout=2) as r:
                return r.status == 200
        except (OSError, urllib.error.URLError):
            return False


# ---------------------------------------------------------------------------
# Le serveur llama.cpp, lancé par l'application
# ---------------------------------------------------------------------------


def trouver_llama_server(chemin_reglage: str) -> pathlib.Path | None:
    """Cherche llama-server : réglage, PATH, puis dossier des liens de winget (Windows)."""
    candidats = []
    if chemin_reglage:
        candidats.append(pathlib.Path(os.path.expandvars(os.path.expanduser(chemin_reglage))))
    trouve = shutil.which("llama-server")
    if trouve:
        candidats.append(pathlib.Path(trouve))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidats.append(pathlib.Path(local) / "Microsoft" / "WinGet" / "Links" / "llama-server.exe")
        paquets = pathlib.Path(local) / "Microsoft" / "WinGet" / "Packages"
        if paquets.is_dir():
            candidats += sorted(paquets.glob("ggml.llamacpp*/**/llama-server.exe"))
    for c in candidats:
        if c.is_file():
            return c
    return None


def son_de_test(secondes: float = 1.0) -> bytes:
    t = np.arange(int(secondes * 16000)) / 16000
    return wav_16k((0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


class ServeurLlamaCpp:
    """Fait tourner llama-server avec Gemma 4 E2B, et le redémarre s'il s'arrête."""

    def __init__(self, r, journal: pathlib.Path):
        self.r = r
        self.journal = journal
        self.processus: subprocess.Popen | None = None
        from .reglages import DOSSIER_DONNEES

        self.dossier_memoire = DOSSIER_DONNEES / "memoire"
        self.cerveau = CerveauLlamaCpp(f"http://127.0.0.1:{r.llamacpp_port}", r.temperature)

    def commande(self, executable: pathlib.Path) -> list[str]:
        r = self.r
        return [
            str(executable),
            "-m", str(r.llamacpp_modele),
            "--mmproj", str(r.llamacpp_mmproj),
            "-c", str(r.llamacpp_contexte),
            "-np", "1",                         # une seule conversation à la fois
            "--slot-save-path", str(self.dossier_memoire),  # sauvegarde de la mémoire (voir plus haut)
            "-ngl", str(r.llamacpp_couches_gpu),
            "--reasoning", "off",               # pas de « réflexion » silencieuse
            "--jinja",                          # modèle de conversation de Gemma 4 (outils)
            "-fa", "auto",
            "--cache-ram", "0",
            "--host", "127.0.0.1",
            "--port", str(r.llamacpp_port),
        ]

    def demarrer(self) -> None:
        r = self.r
        executable = trouver_llama_server(r.llamacpp_serveur)
        if executable is None:
            raise ErreurCerveau(
                "llama.cpp est introuvable. Relancez installer.bat (il l'installe avec "
                "« winget install llama.cpp »), puis fermez et rouvrez lancer.bat."
            )
        for nom, chemin in (("le modèle", r.llamacpp_modele), ("le module audio", r.llamacpp_mmproj)):
            if not chemin.is_file():
                raise ErreurCerveau(f"Fichier introuvable pour {nom} : {chemin}\nRelancez installer.bat.")
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.dossier_memoire.mkdir(parents=True, exist_ok=True)
        options = {}
        if sys.platform == "win32":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        log.info("Démarrage de llama.cpp (%s)…", executable)
        depuis = self.journal.stat().st_size if self.journal.exists() else 0
        self.processus = subprocess.Popen(
            self.commande(executable), stdout=open(self.journal, "ab"), stderr=subprocess.STDOUT,
            **options,
        )
        debut = time.monotonic()
        while not self.cerveau.pret():
            if self.processus.poll() is not None:
                fin = self._lire_journal(depuis)[-1500:]
                raise ErreurCerveau(f"llama.cpp s'est arrêté au démarrage. Fin du journal :\n{fin}")
            if time.monotonic() - debut > 600:
                self.arreter()
                raise ErreurCerveau("llama.cpp met plus de 10 minutes à démarrer.")
            time.sleep(1)
        log.info("Cerveau prêt en %.0f s.", time.monotonic() - debut)
        debut = time.monotonic()
        self.cerveau.rediger([self.cerveau.message_vocal(son_de_test(), "Dis seulement : prêt.")])
        log.info("Écoute vérifiée (première réponse en %.1f s).", time.monotonic() - debut)

    def _lire_journal(self, depuis: int) -> str:
        try:
            with open(self.journal, "rb") as f:
                f.seek(depuis)
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def vivant(self) -> bool:
        return self.processus is not None and self.processus.poll() is None and self.cerveau.pret()

    def reparer(self) -> None:
        """Redémarre llama.cpp s'il s'est arrêté."""
        if self.vivant():
            return
        log.warning("llama.cpp s'est arrêté : redémarrage…")
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
