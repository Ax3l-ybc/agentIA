"""Moteurs audio locaux : la voix (Phonon) et le détecteur de parole (Silero).

L'écoute et la réflexion sont faites par Gemma 4 dans llama.cpp (voir cerveau.py).
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Iterator

import numpy as np

log = logging.getLogger("assistant.moteurs")


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

    @property
    def voices(self) -> list[str]:
        return list(self.tts.voices)

    def parler(self, texte: str, arret: threading.Event, voix: str | None = None) -> Iterator[np.ndarray]:
        """Produit l'audio (float32, à self.taux Hz) morceau par morceau.

        `voix` : nom d'une voix Phonon (celle de l'assistant), sinon la voix par défaut."""
        if voix and voix not in self.tts.voices:
            log.warning("Voix « %s » inconnue : voix par défaut. Voix disponibles : %s",
                        voix, ", ".join(self.tts.voices))
            voix = None
        with self.verrou:
            if arret.is_set():
                return
            with self.tts.stream(texte, voice=voix or self.voix) as flux:
                for morceau in flux:
                    if arret.is_set():
                        break
                    yield np.asarray(morceau, dtype=np.float32).reshape(-1)


# Débuts de réponse fréquents, préparés au démarrage : quand une réponse commence ainsi, sa
# voix sort aussitôt (sinon, il faut la fabriquer pendant que llama.cpp écrit la suite, et les
# deux se partagent le processeur).
DEBUTS_FREQUENTS = [
    "Bonjour !", "Bonjour,", "Salut !", "Salut,", "Bien sûr !", "Bien sûr,", "Oui,", "Oui !", "D'accord,",
    "D'accord !", "Très bien,", "Avec plaisir !", "Avec plaisir,", "Parfait,", "Super !", "Absolument !",
    "Je comprends,", "Excellente question !", "Bonne question !", "Alors,", "Voici", "En effet,",
    "Pas de souci !", "Volontiers !",
]


class MoteurKokoro:
    """Kokoro-82M en local (ONNX), voix française « Siwis » : libre, y compris en entreprise.

    Kokoro fabrique chaque morceau de phrase d'un coup (pas au fil de l'eau) : le temps avant
    le premier son dépend donc de la longueur du premier morceau (la page envoie un premier
    morceau court). Pour gagner encore du temps :
    - le modèle est préchauffé au chargement (la toute première synthèse est plus lente) ;
    - les débuts de réponse courts et fréquents (« Bonjour ! », « D'accord, »…) sont gardés en
      mémoire : ils sortent instantanément la fois suivante.
    """

    taux = 24000
    voices = ["Siwis"]
    FICHIERS = {"model_uint8.onnx": "onnx/model_uint8.onnx", "ff_siwis.bin": "voices/ff_siwis.bin",
                "tokenizer.json": "tokenizer.json"}
    DEPOT = "onnx-community/Kokoro-82M-v1.0-ONNX"

    def __init__(self, dossier: pathlib.Path, vitesse: float = 0.0, fils: int | None = None):
        import json
        import os

        import espeakng_loader
        import onnxruntime
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        log.info("Chargement de la voix Kokoro (Siwis)…")
        MoteurKokoro.telecharger(dossier)
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        self.phonemes = EspeakBackend("fr-fr", preserve_punctuation=True, with_stress=True,
                                      language_switch="remove-flags")
        self.vocab = json.loads((dossier / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
        self.style = np.fromfile(dossier / "ff_siwis.bin", dtype=np.float32).reshape(-1, 1, 256)
        options = onnxruntime.SessionOptions()
        # llama.cpp calcule la suite de la réponse en même temps : on lui laisse des cœurs.
        options.intra_op_num_threads = fils or max(2, (os.cpu_count() or 4) // 2)
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(str(dossier / "model_uint8.onnx"), options,
                                                    providers=["CPUExecutionProvider"])
        self.vitesse = max(0.5, min(2.0, 1.0 - vitesse))  # réglage commun : -0.2 = plus rapide
        self.verrou = threading.Lock()
        self.memoire: dict[str, np.ndarray] = {}
        self._synthese("Bonjour.")  # préchauffage
        log.info("Voix prête (Kokoro, Siwis).")
        self.preparer(DEBUTS_FREQUENTS)

    def preparer(self, textes: list[str]) -> None:
        """Fabrique à l'avance, en arrière-plan, la voix de débuts de réponse probables."""
        def travail():
            for texte in textes:
                if texte.strip() not in self.memoire:
                    with self.verrou:
                        self.memoire[texte.strip()] = self._synthese(texte.strip())

        threading.Thread(target=travail, daemon=True).start()

    @staticmethod
    def telecharger(dossier: pathlib.Path) -> None:
        """Télécharge les fichiers de Kokoro qui manquent (environ 180 Mo, une seule fois)."""
        from .modeles import telecharger_fichier, url

        for nom, chemin in MoteurKokoro.FICHIERS.items():
            if not (dossier / nom).is_file():
                log.info("Téléchargement de la voix : %s…", nom)
                telecharger_fichier(url(MoteurKokoro.DEPOT, chemin), dossier / nom)

    def _synthese(self, texte: str) -> np.ndarray:
        sons = self.phonemes.phonemize([texte])[0].replace("-", "")
        ids = [self.vocab[c] for c in sons if c in self.vocab]
        if not ids:
            return np.zeros(0, dtype=np.float32)
        morceaux = []
        for i in range(0, len(ids), 500):  # Kokoro lit au plus 510 sons d'un coup
            bout = ids[i:i + 500]
            sortie = self.session.run(None, {
                "input_ids": np.array([[0, *bout, 0]], dtype=np.int64),
                "style": self.style[min(len(bout), len(self.style) - 1)],
                "speed": np.array([self.vitesse], dtype=np.float32),
            })[0]
            morceaux.append(np.asarray(sortie, dtype=np.float32).reshape(-1))
        return np.concatenate(morceaux)

    def parler(self, texte: str, arret: threading.Event, voix: str | None = None) -> Iterator[np.ndarray]:
        """Produit l'audio (float32, 24 kHz) du texte. (Une seule voix française : `voix` est ignoré.)"""
        cle = texte.strip()
        if cle in self.memoire:
            yield self.memoire[cle]
            return
        with self.verrou:
            if arret.is_set():
                return
            audio = self._synthese(cle)
        if len(cle) <= 40:  # débuts de réponse fréquents : gardés pour la fois suivante
            if len(self.memoire) > 200:
                self.memoire.clear()
            self.memoire[cle] = audio
        if len(audio) and not arret.is_set():
            yield audio


class MoteurVoixSimulation:
    """Pour les tests : un son de 50 ms par mot."""

    taux = 24000

    voices = ["Hélène", "Jules"]

    def parler(self, texte: str, arret: threading.Event, voix: str | None = None) -> Iterator[np.ndarray]:
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
