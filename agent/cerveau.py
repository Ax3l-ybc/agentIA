"""Le cerveau qui écoute : Gemma 4 E2B dans llama.cpp, sur l'ordinateur.

Gemma 4 comprend directement l'audio : chaque phrase de l'étudiant lui est envoyée telle
quelle, sans transcription préalable. llama.cpp est lancé par l'application avec deux
« espaces » (slots) : l'espace 0 garde la conversation (consignes, notes et fiches déjà
lues restent en mémoire d'un tour à l'autre), l'espace 1 sert aux retranscriptions et à
la rédaction des fiches, sans effacer l'espace 0.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from typing import Iterator

import numpy as np

log = logging.getLogger("agent.cerveau")

ESPACE_CONVERSATION = 0
ESPACE_ANNEXE = 1

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
        """Message « étudiant » contenant l'enregistrement de sa phrase."""
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": consigne},
                {"type": "input_audio",
                 "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"}},
            ],
        }

    def discuter_en_flux(self, messages: list[dict], outils: list[dict] | None = None,
                         espace: int = ESPACE_CONVERSATION, max_mots: int | None = None) -> Iterator[dict]:
        """Réponse en flux : {"texte": …} au fil de l'eau, puis {"outil", "arguments", "id"}."""
        corps = {
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "cache_prompt": True,  # réutilise les consignes et l'historique déjà lus
            "id_slot": espace,
        }
        if outils:
            corps["tools"] = outils
        if max_mots:
            corps["max_tokens"] = max_mots
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
                        a = appels.setdefault(tc.get("index", 0), {"nom": "", "args": "", "id": ""})
                        fonction = tc.get("function") or {}
                        a["nom"] += fonction.get("name") or ""
                        a["args"] += fonction.get("arguments") or ""
                        a["id"] = tc.get("id") or a["id"]
        for i, a in sorted(appels.items()):
            try:
                arguments = json.loads(a["args"]) if a["args"] else {}
            except json.JSONDecodeError:
                arguments = {}
            yield {"outil": a["nom"], "arguments": arguments, "id": a["id"] or f"appel_{i}"}

    def rediger(self, messages: list[dict], espace: int = ESPACE_ANNEXE,
                temperature: float = 0.3) -> str:
        """Réponse complète (fiches, retranscriptions), dans l'espace annexe."""
        corps = {"messages": messages, "stream": False, "temperature": temperature,
                 "cache_prompt": True, "id_slot": espace}
        with self._post("/v1/chat/completions", corps) as r:
            reponse = json.load(r)
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
        self.cerveau = CerveauLlamaCpp(f"http://127.0.0.1:{r.llamacpp_port}", r.temperature)

    def commande(self, executable: pathlib.Path) -> list[str]:
        r = self.r
        return [
            str(executable),
            "-m", str(r.llamacpp_modele),
            "--mmproj", str(r.llamacpp_mmproj),
            "-c", str(2 * r.llamacpp_contexte),  # partagé entre les deux espaces
            "-np", "2",                         # espace 0 : conversation ; 1 : annexe
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
