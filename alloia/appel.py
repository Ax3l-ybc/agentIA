"""Un appel AlloIA (version « omni ») : le modèle écoute directement la voix du maker.

Déroulement d'un tour de parole :
1. le micro est découpé en phrases (détecteur Silero, pause de fin de phrase) ;
2. l'audio de la phrase est envoyé tel quel au modèle (Gemma 4), avec l'historique ;
3. la réponse arrive en flux et chaque phrase est dite par la voix dès qu'elle est complète ;
4. en arrière-plan, la phrase du maker est retranscrite (historique et documentation).
Le micro est coupé pendant que l'agent parle (anti-écho).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import re
import threading
import time

import numpy as np

from agent.serveur_voix import couper_phrases, nettoyer_pour_voix
from agent.moteurs import Reechantillonneur

from . import ollama_audio
from .audio import Combine
from .phrases import DecoupeurPhrases
from .reglages import Reglages

log = logging.getLogger("alloia.appel")

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "raccrocher",
            "description": "Termine l'appel. À utiliser après avoir remercié le maker, quand il a "
            "fini de raconter son projet ou qu'il dit au revoir.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

ADAPTATION_ORALE = """
# Conversation téléphonique (règles de forme, prioritaires)
- Tout ce que tu écris est lu à voix haute par une voix de synthèse : phrases courtes et
  naturelles, pas de listes, pas de titres, pas d'émojis, pas de symboles, pas de Markdown.
- Les messages du maker t'arrivent en audio. Réponds à ce qu'il dit : ne répète pas et ne
  transcris pas ses paroles.
- Les messages entre parenthèses ne viennent pas du maker : ce sont des indications sur l'appel.
"""

CONSIGNE_VOCALE = "(Message vocal du maker. Réponds-lui directement, sans répéter ses paroles.)"
DECROCHE = "(Le maker vient de décrocher le combiné. Commence l'appel.)"
RELANCE = "(Le maker ne dit plus rien depuis un moment. Relance-le gentiment, en une phrase.)"

_AU_REVOIR = re.compile(r"\b(au revoir|bonne journée|bonne soirée|à plus tard|à bientôt|ciao)\b", re.I)
_EMOJIS = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


@dataclasses.dataclass
class Echange:
    role: str  # "maker" ou "alloia"
    texte: str


@dataclasses.dataclass
class Conversation:
    debut: datetime.datetime
    echanges: list[Echange] = dataclasses.field(default_factory=list)
    raison_fin: str = ""

    def ajouter(self, role: str, texte: str) -> Echange:
        e = Echange(role, texte.strip())
        self.echanges.append(e)
        return e

    def maker_a_parle(self) -> bool:
        return any(e.role == "maker" and e.texte for e in self.echanges)

    def transcription(self) -> str:
        noms = {"maker": "Maker", "alloia": "AlloIA"}
        return "\n".join(f"{noms[e.role]} : {e.texte}" for e in self.echanges if e.texte)


# ---------------------------------------------------------------------------
# Veille : attendre que quelqu'un décroche et parle
# ---------------------------------------------------------------------------


async def attendre_decroche(
    combine: Combine, detecteur, duree_parole_s: float, evenement_clavier: asyncio.Event
) -> str:
    """Attend qu'on parle dans le combiné (ou qu'on appuie sur Entrée). Renvoie la cause."""
    reech = Reechantillonneur(24000, 16000)
    reste = np.zeros(0, dtype=np.float32)
    parole: list[float] = []
    if hasattr(detecteur, "reinitialiser"):
        detecteur.reinitialiser()
    combine.vider_micro()
    while True:
        if evenement_clavier.is_set():
            evenement_clavier.clear()
            return "clavier"
        try:
            trame = await asyncio.wait_for(combine.micro.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        reste = np.concatenate([reste, reech(trame)])
        maintenant = time.monotonic()
        while len(reste) >= 512:
            bloc, reste = reste[:512], reste[512:]
            if detecteur(bloc) > 0.5:
                parole.append(maintenant)
        parole = [t for t in parole if maintenant - t < 1.5]
        if len(parole) * 0.032 >= duree_parole_s:
            return "voix"


# ---------------------------------------------------------------------------
# L'appel
# ---------------------------------------------------------------------------


class Appel:
    def __init__(self, r: Reglages, consignes: str, voix, detecteur, cerveau):
        self.r = r
        self.systeme = consignes.strip() + "\n\n" + ADAPTATION_ORALE.strip()
        self.voix = voix
        self.cerveau = cerveau  # CerveauLlamaCpp ou CerveauOllama (alloia/cerveau.py)
        self.detecteur = detecteur
        self.historique: list[dict] = []  # messages texte (sans audio) des tours précédents
        self.transcriptions: set[asyncio.Task] = set()
        self.raccrocher = False

    # --------------------------------------------------------------- un tour

    async def tour(self, combine: Combine, conv: Conversation, audio: np.ndarray | None,
                   indication: str = "", fin_phrase: float | None = None) -> str:
        """Envoie une phrase du maker (ou une indication) et dit la réponse."""
        await self._attendre_transcriptions(10)
        if audio is not None:
            wav = ollama_audio.wav_16k(audio)
            actuel = self.cerveau.message_vocal(wav, CONSIGNE_VOCALE)
        else:
            actuel = {"role": "user", "content": indication}
        messages = [{"role": "system", "content": self.systeme}, *self.historique, actuel]

        boucle = asyncio.get_running_loop()
        file: asyncio.Queue = asyncio.Queue()
        debut = time.monotonic()

        def produire():
            try:
                for m in self.cerveau.discuter_en_flux(messages, OUTILS):
                    boucle.call_soon_threadsafe(file.put_nowait, m)
            except Exception as exc:  # noqa: BLE001
                boucle.call_soon_threadsafe(file.put_nowait, exc)
            finally:
                boucle.call_soon_threadsafe(file.put_nowait, None)

        threading.Thread(target=produire, daemon=True).start()

        reponse, tampon, premiere = "", "", True
        premier_mot = premier_son = None
        while True:
            m = await file.get()
            if isinstance(m, Exception):
                log.error("Le cerveau n'a pas répondu : %s", m)
                break
            phrases: list[str] = []
            if m is None:
                if tampon.strip():
                    phrases.append(tampon)
                tampon = ""
            elif "outil" in m:
                if m["outil"] == "raccrocher":
                    log.info("L'agent raccroche.")
                    self.raccrocher = True
            else:
                if premier_mot is None:
                    premier_mot = time.monotonic() - debut
                    log.info("  réflexion : premier mot après %.1f s", premier_mot)
                reponse += m["texte"]
                tampon += m["texte"]
                phrases, tampon = couper_phrases(tampon, premiere)
            for p in phrases:
                texte = _EMOJIS.sub("", nettoyer_pour_voix(p))
                if not re.search(r"\w", texte):
                    continue
                premiere = False
                delai = await self.voix.dire(texte, combine)
                if premier_son is None:
                    premier_son = time.monotonic()
                    total = premier_son - (fin_phrase if fin_phrase else debut)
                    log.info("  voix : premier son après %.1f s de synthèse (total : %.1f s)", delai, total)
            if m is None:
                break

        reponse = _EMOJIS.sub("", reponse).strip()
        # Historique en texte : la phrase du maker sera remplacée par sa transcription.
        message_maker = {"role": "user", "content": indication or "(message vocal)"}
        self.historique.append(message_maker)
        self.historique.append({"role": "assistant", "content": reponse or "…"})
        if audio is not None:
            echange = conv.ajouter("maker", "")
            self._lancer_transcription(wav, message_maker, echange)
        conv.ajouter("alloia", reponse)
        return reponse

    def _lancer_transcription(self, wav: bytes, message: dict, echange: Echange) -> None:
        async def tache():
            t0 = time.monotonic()
            try:
                texte = await asyncio.to_thread(self.cerveau.transcrire, wav)
            except Exception as exc:  # noqa: BLE001
                log.warning("Transcription impossible : %s", exc)
                texte = ""
            message["content"] = texte or "(message vocal inaudible)"
            echange.texte = texte
            log.info("  maker (transcrit en %.1f s) : %s", time.monotonic() - t0, texte)

        t = asyncio.create_task(tache())
        self.transcriptions.add(t)
        t.add_done_callback(self.transcriptions.discard)

    async def _attendre_transcriptions(self, delai: float) -> None:
        if self.transcriptions:
            await asyncio.wait(set(self.transcriptions), timeout=delai)

    # ------------------------------------------------------------ l'appel

    async def derouler(self, combine: Combine, evenement_clavier: asyncio.Event) -> Conversation:
        r = self.r
        conv = Conversation(debut=datetime.datetime.now())
        decoupeur = DecoupeurPhrases(self.detecteur, pause_s=r.pause_s)
        phrases: asyncio.Queue = asyncio.Queue()
        fin = asyncio.Event()
        debut = time.monotonic()
        derniere_parole = [time.monotonic()]
        combine.vider_micro()

        async def ecouter():
            while not fin.is_set():
                try:
                    trame = await asyncio.wait_for(combine.micro.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not accueil_fini.is_set():
                    decoupeur.reinitialiser()  # « Allô ? » du décroché : on attend la salutation
                    continue
                if r.micro_coupe_quand_agent_parle and combine.agent_parle():
                    continue  # l'agent ne doit pas s'entendre lui-même
                for p in decoupeur.ajouter(trame):
                    derniere_parole[0] = time.monotonic()
                    # La personne a fini de parler `pause_s` avant que la phrase soit livrée.
                    phrases.put_nowait((p, time.monotonic() - r.pause_s))

        accueil_fini = asyncio.Event()
        ecoute = asyncio.create_task(ecouter())
        try:
            await self.tour(combine, conv, None, DECROCHE)
            while combine.agent_parle():  # on écoute à nouveau une fois la salutation dite
                await asyncio.sleep(0.05)
            accueil_fini.set()
            relances = 0
            derniere_activite = time.monotonic()
            while True:
                try:
                    audio, fin_phrase = await asyncio.wait_for(phrases.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    audio = None
                maintenant = time.monotonic()
                if combine.agent_parle():
                    derniere_activite = maintenant
                if audio is not None:
                    # Plusieurs phrases arrivées pendant la réflexion : on les regroupe.
                    while not phrases.empty():
                        suite, fin_phrase = phrases.get_nowait()
                        audio = np.concatenate([audio, suite])
                    relances = 0
                    await self.tour(combine, conv, audio, fin_phrase=fin_phrase)
                    derniere_activite = time.monotonic()
                    continue
                if self.raccrocher and not combine.agent_parle():
                    conv.raison_fin = "l'agent a raccroché"
                    break
                if evenement_clavier.is_set():
                    evenement_clavier.clear()
                    conv.raison_fin = "touche Entrée"
                    break
                if maintenant - debut > r.duree_max_min * 60:
                    conv.raison_fin = "durée maximale atteinte"
                    break
                if decoupeur.en_cours or combine.agent_parle():
                    continue
                dernier_maker = next((e.texte for e in reversed(conv.echanges) if e.role == "maker"), "")
                # Filet de sécurité si le modèle n'appelle pas « raccrocher » : un « au revoir »
                # du maker termine l'appel quelques secondes après la dernière réponse.
                if _AU_REVOIR.search(dernier_maker[-60:]) and maintenant - derniere_activite > 4:
                    conv.raison_fin = "au revoir"
                    break
                if maintenant - derniere_parole[0] > r.fin_apres_silence_s:
                    conv.raison_fin = "silence prolongé"
                    break
                if maintenant - derniere_activite > r.relance_apres_s and relances < 2:
                    relances += 1
                    await self.tour(combine, conv, None, RELANCE)
                    derniere_activite = time.monotonic()
        finally:
            fin.set()
            ecoute.cancel()
            await self._attendre_transcriptions(30)
            combine.vider_haut_parleur()
        log.info("Fin de l'appel (%s), %d échanges.", conv.raison_fin, len(conv.echanges))
        return conv
