"""Comptes rendus de réunion : enregistrement en direct ou fichier audio, tout sur le PC.

1. Le son est découpé en morceaux de 25 s au plus, coupés pendant les silences (Gemma 4
   écoute au plus 30 s d'un coup) ; les longs silences sont ignorés.
2. Chaque morceau est retranscrit par Gemma 4. En direct, la transcription avance pendant
   la réunion : à la fin, il ne reste que les derniers morceaux.
3. Gemma 4 rédige le compte rendu. Pour une longue réunion, il prend d'abord des notes
   partie par partie, puis rédige à partir de ces notes (sa mémoire de travail est limitée).
4. Le résultat devient une conversation : on peut poser des questions sur la réunion, et
   l'exporter en Word. La transcription complète est aussi gardée en texte.

L'enregistrement n'est pas gardé : seul le texte l'est, sur le PC.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import logging
import secrets
import time
from typing import TYPE_CHECKING

import numpy as np

from .cerveau import wav_16k
from .phrases import DecoupeurPhrases

if TYPE_CHECKING:
    from .application import Application

log = logging.getLogger("assistant.reunion")

TAUX = 16000
MORCEAU_MAX_S = 25.0
PARTIE_MAX_CARACTERES = 12000  # au-delà, le compte rendu se fait en plusieurs passes

CONSIGNE_COMPTE_RENDU = """Tu rédiges le compte rendu d'une réunion de travail, à partir de sa transcription
automatique (ci-dessous). La transcription peut contenir des erreurs de reconnaissance :
corrige les mots manifestement mal compris, mais n'invente rien. Les intervenants ne sont pas
identifiés : ne cite un nom que s'il est prononcé dans la réunion.

Rédige en français, en Markdown, exactement avec ces parties :

## Résumé
(3 à 5 phrases : l'objet de la réunion et ce qu'il faut en retenir)

## Points abordés
(une puce par sujet, avec l'essentiel de ce qui a été dit)

## Décisions
(une puce par décision prise ; « Aucune décision explicite. » s'il n'y en a pas)

## Actions à mener
(une puce par action : **qui** (si c'est dit) — quoi — échéance (si elle est dite) ;
« Aucune action explicite. » s'il n'y en a pas)

## Questions en suspens
(ce qui reste à trancher ou à vérifier ; « Aucune. » s'il n'y en a pas)

Réponds uniquement par le compte rendu, sans phrase d'introduction."""

CONSIGNE_NOTES = """Voici une partie de la transcription automatique d'une réunion de travail (la
transcription peut contenir des erreurs de reconnaissance). Prends des notes précises et
complètes sur cette partie : sujets abordés, informations importantes (chiffres, dates, noms
prononcés), décisions prises, actions à mener (qui, quoi, quand), questions restées ouvertes.
Réponds uniquement par les notes, en liste à puces, sans rien inventer."""


def decoder_audio(donnees: bytes) -> np.ndarray:
    """Un fichier audio (mp3, m4a, wav, ogg, webm…) en son mono 16 kHz (float32)."""
    import av

    morceaux = []
    with av.open(io.BytesIO(donnees)) as conteneur:
        if not conteneur.streams.audio:
            raise ValueError("ce fichier ne contient pas de son")
        conversion = av.AudioResampler(format="s16", layout="mono", rate=TAUX)
        for trame in conteneur.decode(audio=0):
            for sortie in conversion.resample(trame):
                morceaux.append(sortie.to_ndarray().reshape(-1))
        for sortie in conversion.resample(None):
            morceaux.append(sortie.to_ndarray().reshape(-1))
    if not morceaux:
        raise ValueError("ce fichier ne contient pas de son")
    return np.concatenate(morceaux).astype(np.float32) / 32768.0


def horodatage(secondes: float) -> str:
    secondes = int(secondes)
    h, reste = divmod(secondes, 3600)
    return f"{h}:{reste // 60:02d}:{reste % 60:02d}" if h else f"{reste // 60:02d}:{reste % 60:02d}"


def duree_lisible(secondes: float) -> str:
    minutes = round(secondes / 60)
    if minutes < 1:
        return f"{int(secondes)} s"
    h, m = divmod(minutes, 60)
    return f"{h} h {m:02d}" if h else f"{m} min"


def decouper_en_parties(lignes: list[str], maximum: int = PARTIE_MAX_CARACTERES) -> list[str]:
    parties, en_cours = [], ""
    for ligne in lignes:
        if en_cours and len(en_cours) + len(ligne) > maximum:
            parties.append(en_cours)
            en_cours = ""
        en_cours += ligne + "\n"
    if en_cours.strip():
        parties.append(en_cours)
    return parties


class Reunion:
    """Une réunion en cours de traitement (enregistrement, transcription, compte rendu)."""

    def __init__(self, app: "Application", assistant: str, source: str):
        self.app = app
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        self.assistant = assistant
        self.source = source  # "enregistrement" ou le nom du fichier audio
        self.debut = datetime.datetime.now()
        self.etape = "enregistrement" if source == "enregistrement" else "lecture"
        self.message = ""
        self.conversation: str | None = None
        self.echantillons = 0  # son reçu, en échantillons à 16 kHz
        self.decoupeur = DecoupeurPhrases(app.nouveau_detecteur(), pause_s=0.6, max_phrase_s=MORCEAU_MAX_S)
        self.attente: list[tuple[float, np.ndarray]] = []  # phrases pas encore groupées
        self.morceaux: asyncio.Queue = asyncio.Queue()
        self.total_morceaux = 0
        self.faits = 0
        self.transcription: list[tuple[float, str]] = []
        self.fin_audio = asyncio.Event()
        self.tache = asyncio.create_task(self._travailler())

    def etat(self) -> dict:
        return {"id": self.id, "etape": self.etape, "message": self.message,
                "duree": duree_lisible(self.echantillons / TAUX), "faits": self.faits,
                "total": self.total_morceaux, "conversation": self.conversation,
                "apercu": " ".join(t for _, t in self.transcription[-2:])[-300:]}

    # --------------------------------------------------------------- le son

    def _detecter(self, audio: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """Repère les phrases dans du son (16 kHz) : [(moment dans la réunion, phrase)]."""
        trouvees = []
        for i in range(0, len(audio), TAUX * 10):
            bloc = audio[i:i + TAUX * 10]
            self.echantillons += len(bloc)
            for phrase in self.decoupeur.ajouter(bloc):
                # Moment approximatif du début de la phrase (pour l'horodatage).
                fin_s = self.echantillons / TAUX - 0.6
                trouvees.append((max(0.0, fin_s - len(phrase) / TAUX), phrase))
        return trouvees

    def ajouter_audio(self, audio: np.ndarray) -> None:
        """Ajoute du son en direct (16 kHz). Les morceaux complets partent à la transcription."""
        for debut, phrase in self._detecter(audio):
            self._grouper(debut, phrase)

    def _grouper(self, debut: float, phrase: np.ndarray) -> None:
        """Regroupe les phrases en morceaux de 25 s au plus."""
        duree = sum(len(p) for _, p in self.attente) + len(phrase)
        if self.attente and duree > MORCEAU_MAX_S * TAUX:
            self._envoyer_attente()
        self.attente.append((debut, phrase))

    def _envoyer_attente(self) -> None:
        if not self.attente:
            return
        debut = self.attente[0][0]
        audio = np.concatenate([p for _, p in self.attente])
        self.attente = []
        self.total_morceaux += 1
        self.morceaux.put_nowait((debut, audio))

    def terminer(self) -> None:
        """Fin du son : les derniers morceaux partent à la transcription."""
        if self.fin_audio.is_set():
            return
        for phrase in self.decoupeur.vider():
            self._grouper(max(0.0, self.echantillons / TAUX - len(phrase) / TAUX), phrase)
        self._envoyer_attente()
        self.fin_audio.set()

    async def importer(self, donnees: bytes) -> None:
        """Fichier audio : on le décode, puis tout le son part à la transcription."""
        try:
            audio = await asyncio.to_thread(decoder_audio, donnees)
        except Exception as exc:  # noqa: BLE001
            self.etape, self.message = "erreur", f"Ce fichier audio n'a pas pu être lu ({exc})."
            self.fin_audio.set()
            return
        self.etape = "transcription"
        self.message = "Repérage des passages parlés…"
        # Le repérage tourne à part ; les morceaux sont envoyés depuis la boucle principale.
        for debut, phrase in await asyncio.to_thread(self._detecter, audio):
            self._grouper(debut, phrase)
        self.terminer()

    # ---------------------------------------------------------- le travail

    async def _travailler(self) -> None:
        try:
            await self._transcrire_tout()
            if self.etape == "erreur":
                return
            if not self.transcription:
                self.etape = "erreur"
                self.message = "Aucune parole n'a été entendue : vérifiez le micro ou le fichier audio."
                return
            self.etape = "redaction"
            self.message = "Rédaction du compte rendu…"
            compte_rendu = await asyncio.to_thread(self._rediger)
            self.conversation = self._garder(compte_rendu)
            self.etape, self.message = "termine", "Compte rendu prêt."
            log.info("Compte rendu de réunion prêt (%s).", self.conversation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Échec du compte rendu de réunion")
            self.etape, self.message = "erreur", f"Le compte rendu n'a pas pu être rédigé ({exc})."

    async def _transcrire_tout(self) -> None:
        while True:
            if self.fin_audio.is_set() and self.morceaux.empty():
                return
            try:
                debut, audio = await asyncio.wait_for(self.morceaux.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self.etape not in ("enregistrement", "erreur"):
                self.etape = "transcription"
                self.message = "Transcription de la réunion…"
            t0 = time.monotonic()
            try:
                texte = await asyncio.to_thread(self.app.cerveau.transcrire, wav_16k(audio))
            except Exception as exc:  # noqa: BLE001
                log.warning("Morceau de réunion non retranscrit : %s", exc)
                texte = ""
            self.faits += 1
            if texte.strip():
                self.transcription.append((debut, texte.strip()))
            log.info("  réunion : morceau %d/%d retranscrit en %.1f s", self.faits, self.total_morceaux,
                     time.monotonic() - t0)

    def lignes(self) -> list[str]:
        return [f"[{horodatage(t)}] {texte}" for t, texte in sorted(self.transcription)]

    def _rediger(self) -> str:
        cerveau = self.app.cerveau
        parties = decouper_en_parties(self.lignes())
        if len(parties) == 1:
            matiere = "TRANSCRIPTION DE LA RÉUNION :\n" + parties[0]
        else:
            notes = []
            for i, partie in enumerate(parties, 1):
                self.message = f"Prise de notes, partie {i} sur {len(parties)}…"
                notes.append(f"NOTES — PARTIE {i} :\n" + cerveau.rediger(
                    [{"role": "system", "content": CONSIGNE_NOTES}, {"role": "user", "content": partie}]))
            self.message = "Rédaction du compte rendu à partir des notes…"
            matiere = ("Voici les notes prises au fil de la réunion, partie par partie :\n\n"
                       + "\n\n".join(notes))
        return cerveau.rediger([{"role": "system", "content": CONSIGNE_COMPTE_RENDU},
                                {"role": "user", "content": matiere}]).strip()

    def _garder(self, compte_rendu: str) -> str:
        """Le compte rendu devient une conversation (la transcription y est jointe)."""
        from .fichiers import tronquer

        r = self.app.reglages
        date = self.debut.strftime("%d/%m/%Y à %Hh%M")
        duree = duree_lisible(self.echantillons / TAUX)
        transcription = "\n".join(self.lignes())
        # La transcription complète est aussi gardée en texte, à côté des conversations.
        dossier = r.dossier_conversations.parent / "reunions"
        dossier.mkdir(parents=True, exist_ok=True)
        fichier = dossier / f"{self.debut:%Y-%m-%d_%Hh%M}_transcription.txt"
        fichier.write_text(f"Réunion du {date} ({duree})\n\n{transcription}\n", encoding="utf-8")

        texte_joint, tronque = tronquer(transcription)
        conv = self.app.conversations.creer(self.assistant)
        conv.titre = f"Réunion du {self.debut:%d/%m/%Y %Hh%M}"
        message = conv.ajouter("utilisateur", f"Fais le compte rendu de la réunion du {date} (durée : {duree}).")
        message["fichiers"] = [{"nom": "Transcription de la réunion", "texte": texte_joint, "tronque": tronque}]
        conv.ajouter("assistant", f"# Compte rendu — réunion du {date}\n\n**Durée** : {duree}  \n"
                                  f"**Transcription complète** : `{fichier}`\n\n{compte_rendu}")
        conv.titre = f"Réunion du {self.debut:%d/%m/%Y %Hh%M}"
        self.app.conversations.enregistrer(conv)
        return conv.id
