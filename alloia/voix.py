"""La voix d'AlloIA : Phonon (sur le processeur) ou un modèle « speech » d'Ollama."""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np

from agent.moteurs import Reechantillonneur

from . import ollama_audio

log = logging.getLogger("alloia.voix")


class Voix:
    taux = 24000

    def morceaux(self, texte: str, arret: threading.Event):
        """Produit l'audio float32 (à self.taux Hz) morceau par morceau."""
        raise NotImplementedError

    async def dire(self, texte: str, combine, arret: threading.Event | None = None) -> float:
        """Synthétise `texte` et l'envoie au haut-parleur. Renvoie le délai du premier son."""
        arret = arret or threading.Event()
        boucle = asyncio.get_running_loop()
        file: asyncio.Queue = asyncio.Queue()
        reech = Reechantillonneur(self.taux, 48000)
        debut = boucle.time()

        def produire():
            try:
                for m in self.morceaux(texte, arret):
                    boucle.call_soon_threadsafe(file.put_nowait, m)
            except Exception as exc:  # noqa: BLE001
                log.exception("Erreur de synthèse vocale")
                boucle.call_soon_threadsafe(file.put_nowait, exc)
            finally:
                boucle.call_soon_threadsafe(file.put_nowait, None)

        threading.Thread(target=produire, daemon=True).start()
        premier = None
        while True:
            m = await file.get()
            if m is None or isinstance(m, Exception):
                break
            if premier is None:
                premier = boucle.time() - debut
            pcm48 = reech(m)
            combine.jouer((np.clip(pcm48, -1, 1) * 32767).astype("<i2").tobytes())
        return premier if premier is not None else 0.0


class VoixPhonon(Voix):
    def __init__(self, modele: str, nom: str = "", vitesse: float = 0.0):
        from agent.moteurs import MoteurVoix

        self.moteur = MoteurVoix(modele, nom, vitesse)
        self.taux = self.moteur.taux

    def morceaux(self, texte, arret):
        yield from self.moteur.parler(texte, arret)


class VoixOllama(Voix):
    """Voix produite par Ollama (/v1/audio/speech), par exemple lfm3-omni."""

    taux = 24000

    def __init__(self, adresse: str, modele: str, nom: str = ""):
        self.adresse, self.modele, self.nom = adresse, modele, nom

    def morceaux(self, texte, arret):
        reste = b""
        for bloc in ollama_audio.parler_en_flux(self.adresse, self.modele, texte, self.nom):
            if arret.is_set():
                return
            bloc = reste + bloc
            n = len(bloc) // 2 * 2
            reste = bloc[n:]
            yield np.frombuffer(bloc[:n], dtype="<i2").astype(np.float32) / 32768.0


class VoixSimulee(Voix):
    """Pour les tests : un son de 50 ms par mot."""

    taux = 24000

    def morceaux(self, texte, arret):
        n = int(0.05 * max(1, len(texte.split())) * self.taux)
        t = np.arange(n) / self.taux
        yield (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
