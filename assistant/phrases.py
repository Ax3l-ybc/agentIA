"""Découpe le son du micro (16 kHz, envoyé par la page web) en phrases.

Le détecteur de parole Silero repère le début et la fin de la parole. Une phrase est
terminée après une pause : son audio est alors envoyé tel quel à Gemma 4, qui l'écoute.
"""

from __future__ import annotations

import numpy as np

TAUX = 16000
BLOC = 512  # 32 ms, taille attendue par Silero


class DecoupeurPhrases:
    def __init__(self, prob_parole, pause_s: float = 1.2, max_phrase_s: float = 25.0,
                 min_parole_s: float = 0.3, anticipation_s: float | None = None):
        self.prob_parole = prob_parole  # (bloc de 512 échantillons) -> probabilité de parole
        # Envoi anticipé : après `anticipation_s` de silence, la phrase « probablement finie »
        # est proposée (sur_provisoire) ; si la personne reprend la parole, sur_reprise.
        self.anticipation_s = anticipation_s
        self.sur_provisoire = None
        self.sur_reprise = None
        self.pause_s = pause_s
        self.max_phrase_s = max_phrase_s  # Gemma écoute au plus 30 s d'un coup
        self.min_parole_s = min_parole_s
        self.reinitialiser()

    def reinitialiser(self) -> None:
        self.reste = np.zeros(0, dtype=np.float32)
        self.avant = np.zeros(0, dtype=np.float32)  # 0,3 s gardées avant la parole
        self.blocs: list[np.ndarray] = []
        self.parle = False
        self.silence_s = 0.0
        self.parole_s = 0.0
        self.provisoire = False
        if hasattr(self.prob_parole, "reinitialiser"):
            self.prob_parole.reinitialiser()

    @property
    def en_cours(self) -> bool:
        """Vrai si la personne est en train de dire une phrase."""
        return self.parle

    def _sans_silence(self) -> np.ndarray:
        """La phrase sans son silence de fin (inutile au modèle, qui l'écouterait pour rien) :
        on n'en garde que 0,2 s. La phrase provisoire et la définitive sont donc identiques."""
        silencieux = int(round(self.silence_s * TAUX / BLOC)) - int(0.2 * TAUX / BLOC)
        garder = self.blocs[:len(self.blocs) - silencieux] if silencieux > 0 else self.blocs
        return np.concatenate(garder)

    def ajouter(self, audio_16k: np.ndarray) -> list[np.ndarray]:
        """Ajoute du son du micro. Renvoie les phrases terminées (audio 16 kHz)."""
        self.reste = np.concatenate([self.reste, audio_16k.astype(np.float32)])
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
                    if self.provisoire:  # finalement, la personne n'avait pas fini
                        self.provisoire = False
                        if self.sur_reprise:
                            self.sur_reprise()
                else:
                    self.silence_s += BLOC / TAUX
                duree = sum(len(b) for b in self.blocs) / TAUX
                if (self.anticipation_s and not self.provisoire and self.silence_s >= self.anticipation_s
                        and self.silence_s < self.pause_s and self.parole_s >= self.min_parole_s):
                    self.provisoire = True
                    if self.sur_provisoire:
                        self.sur_provisoire(self._sans_silence(), self.silence_s)
                if self.silence_s >= self.pause_s or duree >= self.max_phrase_s:
                    if self.parole_s >= self.min_parole_s:  # on ignore les bruits brefs
                        phrases.append(self._sans_silence())
                    self.blocs, self.parle, self.provisoire = [], False, False
                    self.avant = np.zeros(0, dtype=np.float32)
            else:
                self.avant = np.concatenate([self.avant, bloc])[-int(0.3 * TAUX):]
        return phrases
