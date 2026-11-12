"""Découpe le flux du micro en phrases, avec le détecteur de parole Silero.

Une phrase est terminée après une courte pause : son audio (16 kHz) est alors
envoyé tel quel au modèle, qui l'écoute directement.
"""

from __future__ import annotations

import numpy as np

from agent.moteurs import Reechantillonneur

TAUX = 16000
BLOC = 512  # 32 ms, taille attendue par Silero


def detecteur_silero():
    from agent.moteurs import _DetecteurSilero

    return _DetecteurSilero()


class DecoupeurPhrases:
    def __init__(self, prob_parole, pause_s: float = 0.9, max_phrase_s: float = 25.0,
                 min_parole_s: float = 0.3):
        self.prob_parole = prob_parole
        self.pause_s = pause_s
        self.max_phrase_s = max_phrase_s  # Gemma écoute au plus 30 s d'un coup
        self.min_parole_s = min_parole_s
        self.reinitialiser()

    def reinitialiser(self) -> None:
        self.reech = Reechantillonneur(24000, TAUX)
        self.reste = np.zeros(0, dtype=np.float32)
        self.avant = np.zeros(0, dtype=np.float32)  # 0,3 s gardées avant la parole
        self.blocs: list[np.ndarray] = []
        self.parle = False
        self.silence_s = 0.0
        self.parole_s = 0.0
        if hasattr(self.prob_parole, "reinitialiser"):
            self.prob_parole.reinitialiser()

    @property
    def en_cours(self) -> bool:
        """Vrai si la personne est en train de dire une phrase."""
        return self.parle

    def ajouter(self, trame_24k: np.ndarray) -> list[np.ndarray]:
        """Ajoute 80 ms de micro. Renvoie les phrases terminées (audio 16 kHz)."""
        self.reste = np.concatenate([self.reste, self.reech(trame_24k)])
        phrases = []
        while len(self.reste) >= BLOC:
            bloc, self.reste = self.reste[:BLOC], self.reste[BLOC:]
            voix = self.prob_parole(bloc) > 0.5
            if voix and not self.parle:
                self.parle = True
                self.blocs = [self.avant.copy()]
                self.parole_s = self.silence_s = 0.0
            if self.parle:
                self.blocs.append(bloc)
                if voix:
                    self.silence_s = 0.0
                    self.parole_s += BLOC / TAUX
                else:
                    self.silence_s += BLOC / TAUX
                duree = sum(len(b) for b in self.blocs) / TAUX
                if self.silence_s >= self.pause_s or duree >= self.max_phrase_s:
                    if self.parole_s >= self.min_parole_s:  # on ignore les bruits brefs
                        phrases.append(np.concatenate(self.blocs))
                    self.blocs, self.parle = [], False
                    self.avant = np.zeros(0, dtype=np.float32)
            else:
                self.avant = np.concatenate([self.avant, bloc])[-int(0.3 * TAUX):]
        return phrases
