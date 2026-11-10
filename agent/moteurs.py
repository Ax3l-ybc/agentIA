"""Moteurs locaux : Kyutai STT (écoute) et Phonon (voix).

Les deux moteurs sont chargés une seule fois au démarrage et tournent sur la
machine, sans aucun service payant. Des « moteurs de simulation » sans modèle
servent aux tests automatiques (variable d'environnement AGENT_SIMULATION=1).
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from typing import Iterator

import numpy as np

log = logging.getLogger("agent.moteurs")

TAUX_ENTREE = 24000  # Hz, ce que Gradbot envoie au STT (et ce qu'attend Kyutai/Mimi)
TRAME_ENTREE = 1920  # 80 ms
TAUX_SORTIE = 48000  # Hz, ce que Gradbot attend du TTS
TRAME_SORTIE = 3840  # 80 ms

# Jetons spéciaux du STT Kyutai.
JETON_DEBUT_MOT = 0  # « end of padding » : un nouveau mot commence
JETON_VIDE = 3  # « padding » : rien de nouveau


def choisir_materiel(souhait: str) -> str:
    import torch

    if souhait in ("cuda", "cpu"):
        return souhait
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Écoute : Kyutai STT
# ---------------------------------------------------------------------------


class MoteurEcoute:
    """Kyutai STT en streaming, avec ses têtes de détection de fin de tour."""

    taux = TAUX_ENTREE
    trame = TRAME_ENTREE

    def __init__(self, depot_hf: str, materiel: str = "auto"):
        import moshi.models
        import torch

        self.torch = torch
        self.materiel = choisir_materiel(materiel)
        log.info("Chargement de l'écoute Kyutai (%s) sur %s…", depot_hf, self.materiel)
        info = moshi.models.loaders.CheckpointInfo.from_hf_repo(depot_hf)
        self.mimi = info.get_mimi(device=self.materiel)
        self.tokenizer = info.get_text_tokenizer()
        dtype = torch.bfloat16 if self.materiel == "cuda" else torch.float32
        lm = info.get_moshi(device=self.materiel, dtype=dtype)
        self.lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
        self.delai_s = float(info.stt_config.get("audio_delay_seconds", 0.5))
        self.prefixe_s = float(info.stt_config.get("audio_silence_prefix_seconds", 0.0))
        self.verrou = threading.Lock()
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.a_des_tetes_vad = len(lm.extra_heads) > 0
        if not self.a_des_tetes_vad:
            log.warning("Ce modèle STT n'a pas de détection de fin de tour : mode dégradé.")
        if self.materiel == "cpu":
            log.warning(
                "ATTENTION : aucune carte graphique NVIDIA utilisable. L'écoute tourne sur le "
                "processeur et sera probablement trop lente pour une conversation fluide."
            )
        log.info("Écoute prête.")

    def reinitialiser(self) -> None:
        with self.verrou:
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            silence = np.zeros(TRAME_ENTREE, dtype=np.float32)
            for _ in range(int(round(self.prefixe_s / 0.08))):
                self._pas(silence)

    def pas(self, trame: np.ndarray) -> tuple[int | None, str | None, list[float] | None]:
        """Traite 80 ms d'audio. Renvoie (jeton, morceau de texte, probas d'inactivité)."""
        with self.verrou:
            return self._pas(trame)

    def _pas(self, trame: np.ndarray):
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(trame).to(self.materiel)[None, None, :]
            codes = self.mimi.encode(x)
            sortie = self.lm_gen.step_with_extra_heads(codes)
        if sortie is None:
            return None, None, None
        jetons, tetes = sortie
        jeton = int(jetons[0, 0, 0].item())
        probas = [float(t[0, 0, 0].item()) for t in tetes] if tetes else None
        morceau = None
        if jeton not in (JETON_DEBUT_MOT, JETON_VIDE):
            morceau = self.tokenizer.id_to_piece(jeton)
        return jeton, morceau, probas


class MoteurEcouteSimulation:
    """Faux STT pour les tests : « entend » une phrase dès qu'il y a du son."""

    taux = TAUX_ENTREE
    trame = TRAME_ENTREE
    delai_s = 0.0
    a_des_tetes_vad = True

    def __init__(self, phrase: str = "Bonjour, parlons de ma problématique."):
        self.pieces = []
        for mot in phrase.split():
            self.pieces.append(JETON_DEBUT_MOT)
            self.pieces.append("▁" + mot)
        self.reinitialiser()

    def reinitialiser(self) -> None:
        self.a_parle = False
        self.file: list = []
        self.silence = 0

    def pas(self, trame: np.ndarray):
        fort = float(np.abs(trame).mean()) > 0.01
        if fort and not self.a_parle:
            self.a_parle = True
            self.file = list(self.pieces)
        if self.file:
            p = self.file.pop(0)
            if p == JETON_DEBUT_MOT:
                return JETON_DEBUT_MOT, None, [0.05] * 4
            return 5, p, [0.05] * 4
        self.silence = 0 if fort else self.silence + 1
        p = 0.05 if fort else min(1.0, 0.2 + 0.1 * self.silence)
        return JETON_VIDE, None, [p] * 4


# ---------------------------------------------------------------------------
# Écoute de secours : Whisper (pour les PC sans carte NVIDIA)
# ---------------------------------------------------------------------------

# Phrases que Whisper « entend » parfois dans le silence ou le bruit.
HALLUCINATIONS = (
    "sous-titres réalisés par",
    "sous-titrage",
    "amara.org",
    "merci d'avoir regardé",
    "abonnez-vous",
    "sous-titres par",
)


class EcouteParPhrases:
    """Découpe le flux du micro en phrases et les fait transcrire une à une.

    Whisper ne transcrit pas en continu comme Kyutai : on détecte la parole
    avec un petit modèle (Silero), et quand vous marquez une pause, la phrase
    entière est transcrite d'un coup. Le résultat est présenté à Gradbot sous
    la même forme que le STT de Gradium (mots + probabilité d'inactivité).
    """

    taux = TAUX_ENTREE
    trame = TRAME_ENTREE
    delai_s = 0.0
    a_des_tetes_vad = True
    TAUX_WHISPER = 16000
    BLOC = 512  # 32 ms à 16 kHz, taille attendue par Silero

    def __init__(
        self,
        prob_parole,
        transcrire,
        pause_s: float = 0.8,
        max_phrase_s: float = 8.0,
    ):
        self.prob_parole = prob_parole  # (bloc 16 kHz) -> probabilité de parole
        self.transcrire = transcrire  # (audio 16 kHz) -> texte
        self.pause_s = pause_s
        # Au-delà, la phrase est transcrite par morceaux : Gradbot doit recevoir des
        # mots régulièrement, sinon il croit à un silence et redémarre l'écoute.
        self.max_phrase_s = max_phrase_s
        self.phrase: list[tuple[np.ndarray, float]] = []  # (bloc, probabilité de parole)
        self.parle = False
        self.silence_s = 0.0
        self.reinitialiser()

    def reinitialiser(self) -> None:
        """Nouvelle session d'écoute. Une phrase en cours n'est pas perdue."""
        self.reech = Reechantillonneur(TAUX_ENTREE, self.TAUX_WHISPER)
        self.reste = np.zeros(0, dtype=np.float32)
        self.avant = np.zeros(0, dtype=np.float32)  # 0,3 s gardées avant la parole
        self.temps_s = 0.0
        self.debut_s = 0.0  # la phrase en cours (s'il y en a une) repart de 0
        if hasattr(self.prob_parole, "reinitialiser"):
            self.prob_parole.reinitialiser()

    def _duree(self, blocs) -> float:
        return sum(len(b) for b, _ in blocs) / self.TAUX_WHISPER

    def _transcrire_blocs(self, blocs, debut_s: float) -> list[tuple[str, float]]:
        audio = np.concatenate([b for b, _ in blocs])
        texte = nettoyer_transcription(self.transcrire(audio))
        return [(m, debut_s) for m in texte.split()]

    def pas_mots(self, trame: np.ndarray) -> tuple[list[tuple[str, float]], list[float]]:
        """Traite 80 ms d'audio. Renvoie (mots transcrits avec leur début, probas d'inactivité)."""
        self.reste = np.concatenate([self.reste, self.reech(trame)])
        voix_dans_trame = False
        while len(self.reste) >= self.BLOC:
            bloc, self.reste = self.reste[: self.BLOC], self.reste[self.BLOC :]
            p = self.prob_parole(bloc)
            if p > 0.5:
                voix_dans_trame = True
                if not self.parle and not self.phrase:
                    self.debut_s = self.temps_s
                    self.phrase.append((self.avant.copy(), 0.0))
                self.parle, self.silence_s = True, 0.0
            if self.parle:
                self.phrase.append((bloc, p))
                if p <= 0.5:
                    self.silence_s += self.BLOC / self.TAUX_WHISPER
            else:
                self.avant = np.concatenate([self.avant, bloc])[-int(0.3 * self.TAUX_WHISPER) :]
            self.temps_s += self.BLOC / self.TAUX_WHISPER

        mots: list[tuple[str, float]] = []
        if self.phrase and self.silence_s >= self.pause_s:
            # Pause assez longue : fin de la phrase.
            blocs, self.phrase = self.phrase, []
            self.parle, self.silence_s = False, 0.0
            self.avant = np.zeros(0, dtype=np.float32)
            if self._duree(blocs) - self.pause_s >= 0.25:  # on ignore les bruits très courts
                mots = self._transcrire_blocs(blocs, self.debut_s)
        elif self.phrase and self._duree(self.phrase) >= self.max_phrase_s:
            # Vous parlez longtemps sans pause : on transcrit un morceau, coupé de
            # préférence sur la micro-pause la plus récente, et on continue d'écouter.
            coupe = len(self.phrase)
            for i in range(len(self.phrase) - 1, 0, -1):
                if self.phrase[i][1] <= 0.5 and self._duree(self.phrase[:i]) >= 3.0:
                    coupe = i + 1
                    break
            blocs, self.phrase = self.phrase[:coupe], self.phrase[coupe:]
            debut = self.debut_s
            self.debut_s = self.temps_s - self._duree(self.phrase)
            mots = self._transcrire_blocs(blocs, debut)

        # Probabilité d'inactivité présentée à Gradbot :
        # - 0,05 pendant que vous parlez (permet de couper la parole à l'agent) ;
        # - 0,6 tant que votre phrase n'est pas finie et transcrite
        #   (ni « vous parlez », ni « fin de tour ») ;
        # - 1,0 sinon : la phrase est livrée, c'est la fin de votre tour.
        if voix_dans_trame:
            p_inactif = 0.05
        elif self.parle:
            p_inactif = 0.6
        else:
            p_inactif = 1.0
        return mots, [p_inactif] * 4


def nettoyer_transcription(texte: str) -> str:
    texte = texte.strip()
    bas = texte.lower()
    if not texte or any(h in bas for h in HALLUCINATIONS):
        return ""
    return texte


class _DetecteurSilero:
    """Détecteur de parole Silero (fourni avec faster-whisper), utilisé en continu."""

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
        entree = np.concatenate([self.contexte, bloc.astype(np.float32)])[None, :]
        sortie, self.h, self.c = self.session.run(None, {"input": entree, "h": self.h, "c": self.c})
        self.contexte = bloc[-64:].astype(np.float32)
        return float(np.asarray(sortie).reshape(-1)[0])


class MoteurEcouteWhisper(EcouteParPhrases):
    def __init__(self, modele: str = "small", invite: str = "", pause_s: float = 0.8):
        from faster_whisper import WhisperModel

        log.info("Chargement de l'écoute Whisper « %s » (processeur)…", modele)
        self.whisper = WhisperModel(modele, device="cpu", compute_type="int8")
        self.invite = invite or None
        super().__init__(_DetecteurSilero(), self._transcrire, pause_s=pause_s)
        # Échauffement : la toute première transcription est plus lente.
        list(self.whisper.transcribe(np.zeros(16000, dtype=np.float32), language="fr")[0])
        log.info("Écoute Whisper prête. L'agent comprend vos phrases quand vous marquez une pause.")

    def _transcrire(self, audio: np.ndarray) -> str:
        debut = time.time()
        segments, _ = self.whisper.transcribe(
            audio,
            language="fr",
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=self.invite,
        )
        texte = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6)
        self.derniere_duree_s = time.time() - debut
        log.info(
            "  écoute : transcrit en %.1f s (%.1f s de parole) : %s",
            time.time() - debut,
            len(audio) / self.TAUX_WHISPER,
            texte,
        )
        return texte


# ---------------------------------------------------------------------------
# Voix : Phonon
# ---------------------------------------------------------------------------


class MoteurVoix:
    """Phonon (Gradium) en local via le paquet `ptts`."""

    def __init__(self, modele: str, voix: str = "", vitesse: float = 0.0, materiel: str = "auto"):
        import ptts

        log.info("Chargement de la voix Phonon (%s)…", modele)
        kwargs = {}
        # Phonon est minuscule : le processeur suffit. On utilise la carte
        # graphique seulement si ptts a été compilé pour elle.
        if materiel in ("cuda", "auto") and "cuda" in ptts.available_devices():
            kwargs["device"] = "cuda"
        conditions = {"duration_delta": vitesse} if vitesse else None
        self.tts = ptts.TTS(modele, lang="fr", conditions=conditions, **kwargs)
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
