"""Moteurs audio locaux : la voix (Phonon) et le détecteur de parole (Silero).

L'écoute et la réflexion sont faites par Gemma 4 dans llama.cpp (voir cerveau.py).
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Iterator

import numpy as np

log = logging.getLogger("agent.moteurs")


class DetecteurSilero:
    """Détecteur de parole Silero (fourni avec faster-whisper), utilisé en continu à 16 kHz."""

    def __init__(self):
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(pathlib.Path(get_assets_path()) / "silero_vad_v6.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self.reinitialiser()

    def reinitialiser(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.contexte = np.zeros(64, dtype=np.float32)

    def __call__(self, bloc: np.ndarray) -> float:
        """Probabilité de parole dans un bloc de 512 échantillons (32 ms à 16 kHz)."""
        entree = np.concatenate([self.contexte, bloc.astype(np.float32)])[None, :]
        sortie, self.h, self.c = self.session.run(None, {"input": entree, "h": self.h, "c": self.c})
        self.contexte = bloc[-64:].astype(np.float32)
        return float(np.asarray(sortie).reshape(-1)[0])


class MoteurVoix:
    """Phonon (Gradium) en local via le paquet `ptts`."""

    def __init__(self, modele: str, voix: str = "", vitesse: float = 0.0):
        import ptts

        log.info("Chargement de la voix Phonon (%s)…", modele)
        conditions = {"duration_delta": vitesse} if vitesse else None
        self.tts = ptts.TTS(modele, lang="fr", conditions=conditions)
        self.voix = voix or None
        self.taux = int(self.tts.sample_rate)
        self.verrou = threading.Lock()
        log.info("Voix prête (%s Hz). Voix disponibles : %s", self.taux, ", ".join(self.tts.voices))
        if self.voix:
            log.info("Voix choisie : %s", self.voix)

    def parler(self, texte: str, arret: threading.Event) -> Iterator[np.ndarray]:
        """Produit l'audio (float32, à self.taux Hz) morceau par morceau."""
        with self.verrou:
            if arret.is_set():
                return
            with self.tts.stream(texte, voice=self.voix) as flux:
                for morceau in flux:
                    if arret.is_set():
                        break
                    yield np.asarray(morceau, dtype=np.float32).reshape(-1)


class MoteurVoixSimulation:
    """Pour les tests : un son de 50 ms par mot."""

    taux = 24000

    def parler(self, texte: str, arret: threading.Event) -> Iterator[np.ndarray]:
        duree = 0.05 * max(1, len(texte.split()))
        t = np.arange(int(duree * self.taux)) / self.taux
        signal = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        for i in range(0, len(signal), 2400):
            if arret.is_set():
                return
            yield signal[i : i + 2400]


# ---------------------------------------------------------------------------
# Outils audio
# ---------------------------------------------------------------------------


class Reechantillonneur:
    """Rééchantillonnage linéaire continu (sans clic entre deux morceaux)."""

    def __init__(self, taux_entree: int, taux_sortie: int):
        self.pas_ = taux_entree / taux_sortie
        self.position = 0.0  # position du prochain échantillon de sortie
        self.precedent = np.zeros(0, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.pas_ == 1.0:
            return x.astype(np.float32)
        tampon = np.concatenate([self.precedent, x.astype(np.float32)])
        if len(tampon) < 2:
            self.precedent = tampon
            return np.zeros(0, dtype=np.float32)
        positions = np.arange(self.position, len(tampon) - 1, self.pas_)
        sortie = np.interp(positions, np.arange(len(tampon)), tampon).astype(np.float32)
        suivante = self.position + len(positions) * self.pas_
        garde = int(np.floor(suivante))
        self.precedent = tampon[garde:]
        self.position = suivante - garde
        return sortie


def float_vers_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_vers_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
