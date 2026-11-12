"""Le combiné : micro et haut-parleur de la carte son USB branchée sur la Jetson.

Gradbot reçoit le micro en PCM 16 bits mono à 24 kHz et rend la voix à 48 kHz.
La carte son, elle, travaille souvent à 44,1 ou 48 kHz : on rééchantillonne
dans les deux sens.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np

from agent.moteurs import Reechantillonneur

log = logging.getLogger("alloia.audio")

TAUX_GRADBOT_ENTREE = 24000
TAUX_GRADBOT_SORTIE = 48000
TRAME_MICRO = 1920  # 80 ms à 24 kHz


class Haut_parleur_tampon:
    """Tampon de lecture partagé entre l'asyncio et le fil audio."""

    def __init__(self, volume: float = 1.0, queue_s: float = 0.4):
        self.verrou = threading.Lock()
        self.tampon = np.zeros(0, dtype=np.float32)
        self.volume = volume
        self.queue_s = queue_s  # le micro reste coupé un peu après la fin de la voix
        self.derniere_sortie = 0.0

    def ajouter(self, pcm: np.ndarray) -> None:
        with self.verrou:
            self.tampon = np.concatenate([self.tampon, pcm * self.volume])

    def vider(self) -> None:
        with self.verrou:
            self.tampon = np.zeros(0, dtype=np.float32)

    def prendre(self, n: int) -> np.ndarray:
        with self.verrou:
            morceau, self.tampon = self.tampon[:n], self.tampon[n:]
        if len(morceau):
            self.derniere_sortie = time.monotonic()
        if len(morceau) < n:
            morceau = np.concatenate([morceau, np.zeros(n - len(morceau), dtype=np.float32)])
        return morceau

    def parle(self) -> bool:
        with self.verrou:
            if len(self.tampon):
                return True
        return time.monotonic() - self.derniere_sortie < self.queue_s


class Combine:
    """Interface commune : la vraie carte son et le combiné simulé des tests."""

    def __init__(self):
        self.micro: asyncio.Queue = asyncio.Queue()  # trames float32 à 24 kHz
        self.hp = Haut_parleur_tampon()

    def jouer(self, pcm16_48k: bytes) -> None:
        pcm = np.frombuffer(pcm16_48k, dtype="<i2").astype(np.float32) / 32768.0
        self.hp.ajouter(self._vers_sortie(pcm))

    def _vers_sortie(self, pcm48k: np.ndarray) -> np.ndarray:
        return pcm48k

    def vider_haut_parleur(self) -> None:
        self.hp.vider()

    def agent_parle(self) -> bool:
        return self.hp.parle()

    def vider_micro(self) -> None:
        while not self.micro.empty():
            self.micro.get_nowait()

    async def demarrer(self) -> None: ...

    async def arreter(self) -> None: ...


def trouver_peripherique(sd, morceau_nom: str, entree: bool) -> int | None:
    """Numéro de la carte son dont le nom contient `morceau_nom` (None = par défaut)."""
    if not morceau_nom:
        return None
    cle = "max_input_channels" if entree else "max_output_channels"
    for i, p in enumerate(sd.query_devices()):
        if morceau_nom.lower() in p["name"].lower() and p[cle] > 0:
            return i
    raise RuntimeError(
        f"Carte son « {morceau_nom} » introuvable. Lancez « ./jetson/lancer.sh --lister-audio »."
    )


def lister_peripheriques() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


class CombineUSB(Combine):
    def __init__(self, entree: str = "", sortie: str = "", volume: float = 1.0):
        super().__init__()
        import sounddevice as sd

        self.sd = sd
        self.hp.volume = volume
        self.id_entree = trouver_peripherique(sd, entree, True)
        self.id_sortie = trouver_peripherique(sd, sortie, False)
        try:
            info_e = sd.query_devices(self.id_entree, "input")
            info_s = sd.query_devices(self.id_sortie, "output")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Aucun micro ou haut-parleur utilisable. Vérifiez que la carte son USB est "
                "branchée (commande « arecord -l » et « aplay -l »), puis indiquez son nom "
                "dans alloia.yaml (./jetson/lancer.sh --lister-audio)."
            ) from exc
        self.taux_entree = int(info_e["default_samplerate"])
        self.taux_sortie = int(info_s["default_samplerate"])
        self.canaux_sortie = 2 if info_s["max_output_channels"] >= 2 else 1
        log.info("Micro : %s (%s Hz)", info_e["name"], self.taux_entree)
        log.info("Haut-parleur : %s (%s Hz)", info_s["name"], self.taux_sortie)
        self.reech_entree = Reechantillonneur(self.taux_entree, TAUX_GRADBOT_ENTREE)
        self.reech_sortie = Reechantillonneur(TAUX_GRADBOT_SORTIE, self.taux_sortie)
        self.reste = np.zeros(0, dtype=np.float32)
        self.flux_e = self.flux_s = None

    def _vers_sortie(self, pcm48k: np.ndarray) -> np.ndarray:
        return self.reech_sortie(pcm48k)

    async def demarrer(self) -> None:
        boucle = asyncio.get_running_loop()

        def rappel_micro(donnees, n, t, statut):
            pcm = self.reech_entree(donnees[:, 0].astype(np.float32))
            self.reste = np.concatenate([self.reste, pcm])
            while len(self.reste) >= TRAME_MICRO:
                trame, self.reste = self.reste[:TRAME_MICRO], self.reste[TRAME_MICRO:]
                boucle.call_soon_threadsafe(self.micro.put_nowait, trame.copy())

        def rappel_hp(sortie, n, t, statut):
            morceau = self.hp.prendre(n)
            sortie[:] = np.repeat(morceau[:, None], self.canaux_sortie, axis=1)

        self.flux_e = self.sd.InputStream(
            device=self.id_entree, samplerate=self.taux_entree, channels=1,
            dtype="float32", callback=rappel_micro, blocksize=0,
        )
        self.flux_s = self.sd.OutputStream(
            device=self.id_sortie, samplerate=self.taux_sortie, channels=self.canaux_sortie,
            dtype="float32", callback=rappel_hp, blocksize=0,
        )
        self.flux_e.start()
        self.flux_s.start()

    async def arreter(self) -> None:
        for f in (self.flux_e, self.flux_s):
            if f is not None:
                f.stop()
                f.close()


class CombineSimule(Combine):
    """Pour les tests : joue un « appelant » enregistré et capture la voix de l'agent."""

    def __init__(self, scenario: list[tuple[float, np.ndarray | None]]):
        """`scenario` : liste de (attente, audio 24 kHz à « dire » ou None).

        L'attente est un nombre de secondes, ou ("apres_agent", s) pour attendre
        qu'AlloIA ait répondu puis s secondes.
        """
        super().__init__()
        self.scenario = scenario
        self.sortie: list[np.ndarray] = []
        self.tache = None

    async def demarrer(self) -> None:
        async def lecteur_hp():
            while True:  # le haut-parleur simulé consomme en temps réel
                self.sortie.append(self.hp.prendre(4800))
                await asyncio.sleep(0.1)

        async def micro():
            silence = np.zeros(TRAME_MICRO, dtype=np.float32)
            for attente, audio in self.scenario:
                if isinstance(attente, tuple):  # ("apres_agent", s) : comme une vraie personne,
                    a_parle, calme = False, 0.0  # on attend qu'AlloIA ait vraiment fini (3 s
                    while not (a_parle and calme >= 3.0):  # de silence), puis s secondes
                        parle = self.agent_parle()
                        a_parle = a_parle or parle
                        calme = 0.0 if parle else calme + 0.08
                        self.micro.put_nowait(silence)
                        await asyncio.sleep(0.08)
                    attente = attente[1]
                fin = time.monotonic() + attente
                while time.monotonic() < fin:
                    self.micro.put_nowait(silence)
                    await asyncio.sleep(0.08)
                if audio is not None:
                    for i in range(0, len(audio) - TRAME_MICRO + 1, TRAME_MICRO):
                        self.micro.put_nowait(audio[i : i + TRAME_MICRO])
                        await asyncio.sleep(0.08)
            while True:
                self.micro.put_nowait(silence)
                await asyncio.sleep(0.08)

        self.tache = [asyncio.create_task(lecteur_hp()), asyncio.create_task(micro())]

    async def arreter(self) -> None:
        for t in self.tache or []:
            t.cancel()

    def audio_joue(self) -> np.ndarray:
        return np.concatenate(self.sortie) if self.sortie else np.zeros(0, np.float32)
