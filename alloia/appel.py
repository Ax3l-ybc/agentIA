"""Un appel : décrocher, converser avec la personne, raccrocher, puis rédiger la doc."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
import logging
import re
import time

import gradbot
import numpy as np

from agent.moteurs import Reechantillonneur
from agent.serveur_voix import MOTEURS

from .audio import Combine
from .reglages import Reglages

log = logging.getLogger("alloia.appel")

OUTILS = [
    gradbot.ToolDef(
        "raccrocher",
        "Termine l'appel. À utiliser après avoir remercié la personne, quand elle a fini de "
        "raconter son projet ou qu'elle dit au revoir.",
        json.dumps({"type": "object", "properties": {}, "required": []}),
    ),
]

_AU_REVOIR = re.compile(r"\b(au revoir|bonne journée|bonne soirée|à plus tard|à bientôt|ciao)\b", re.I)


@dataclasses.dataclass
class Echange:
    role: str  # "maker" ou "alloia"
    texte: str


@dataclasses.dataclass
class Conversation:
    debut: datetime.datetime
    echanges: list[Echange] = dataclasses.field(default_factory=list)
    raison_fin: str = ""

    def ajouter(self, role: str, texte: str) -> None:
        texte = texte.strip()
        if not texte:
            return
        if self.echanges and self.echanges[-1].role == role:
            joint = self.echanges[-1].texte + " " + texte
            self.echanges[-1].texte = re.sub(r"\s+([,.…])", r"\1", joint)
        else:
            self.echanges.append(Echange(role, texte))

    def maker_a_parle(self) -> bool:
        return any(e.role == "maker" for e in self.echanges)

    def transcription(self) -> str:
        noms = {"maker": "Maker", "alloia": "AlloIA"}
        return "\n".join(f"{noms[e.role]} : {e.texte}" for e in self.echanges)


# ---------------------------------------------------------------------------
# Veille : attendre que quelqu'un décroche et parle
# ---------------------------------------------------------------------------


async def attendre_decroche(combine: Combine, duree_parole_s: float, evenement_clavier: asyncio.Event) -> str:
    """Attend qu'on parle dans le combiné (ou qu'on appuie sur Entrée). Renvoie la cause."""
    ecoute = MOTEURS["ecoute"]
    detecteur = getattr(ecoute, "prob_parole", None)
    reech = Reechantillonneur(24000, 16000)
    reste = np.zeros(0, dtype=np.float32)
    parole: list[float] = []  # instants des blocs de parole récents
    if detecteur is not None and hasattr(detecteur, "reinitialiser"):
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
        if detecteur is None:
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
# L'appel lui-même
# ---------------------------------------------------------------------------


class Appel:
    def __init__(self, r: Reglages, modele_llm: str, consignes: str):
        self.r = r
        self.modele_llm = modele_llm
        self.consignes = consignes

    def config_session(self) -> gradbot.SessionConfig:
        return gradbot.SessionConfig(
            instructions=self.consignes,
            language=gradbot.Lang.Fr,
            tools=OUTILS,
            assistant_speaks_first=True,
            silence_timeout_s=self.r.relance_apres_s,
            flush_duration_s=0.5,
        )

    async def derouler(self, combine: Combine, evenement_clavier: asyncio.Event) -> Conversation:
        conv = Conversation(debut=datetime.datetime.now())
        entree, sortie = await gradbot.run(
            gradium_api_key="local",
            gradium_base_url=f"http://127.0.0.1:{self.r.port}/api",
            llm_base_url=f"{self.r.ollama_adresse}/v1",
            llm_model_name=self.modele_llm,
            llm_api_key="ollama",
            session_config=self.config_session(),
            input_format=gradbot.AudioFormat.Pcm,
            output_format=gradbot.AudioFormat.Pcm,
        )
        fin = asyncio.Event()
        raccrocher_demande = asyncio.Event()
        t_raccrocher = [0.0]
        derniere_parole = [time.monotonic()]
        au_revoir_entendu = [0.0]
        debut = time.monotonic()
        combine.vider_micro()
        chrono: dict[str, float] = {}

        def ecoute_duree() -> float:
            return getattr(MOTEURS["ecoute"], "derniere_duree_s", 0.0)

        async def envoyer_micro():
            silence = np.zeros(1920, dtype=np.float32)
            while not fin.is_set():
                try:
                    trame = await asyncio.wait_for(combine.micro.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if self.r.micro_coupe_quand_agent_parle and combine.agent_parle():
                    trame = silence  # l'agent ne doit pas s'entendre lui-même
                pcm16 = (np.clip(trame, -1, 1) * 32767).astype("<i2").tobytes()
                await entree.send_audio(pcm16)

        async def recevoir():
            while not fin.is_set():
                msg = await sortie.receive()
                if msg is None:
                    break
                t = msg.msg_type
                if t == "audio" and msg.data:
                    combine.jouer(bytes(msg.data))
                elif t == "stt_text" and msg.text:
                    if "fin_phrase" not in chrono:
                        # Whisper livre la phrase ~pause s après votre dernier mot, puis transcrit.
                        chrono["fin_phrase"] = time.monotonic() - self.r.pause_s - ecoute_duree()
                    conv.ajouter("maker", msg.text)
                    derniere_parole[0] = time.monotonic()
                    # « au revoir » dit après la première minute : l'appel se termine si
                    # l'agent ne raccroche pas lui-même et que la personne ne reprend pas.
                    # (les mots arrivent un par un : on regarde la fin de la phrase en cours)
                    fin_phrase = conv.echanges[-1].texte[-60:]
                    if _AU_REVOIR.search(fin_phrase) and time.monotonic() - debut > 60:
                        au_revoir_entendu[0] = time.monotonic()
                    else:
                        au_revoir_entendu[0] = 0.0
                elif t == "tts_text" and msg.text:
                    conv.ajouter("alloia", gradbot.schemas.sanitize(msg.text))
                elif t == "event" and msg.event is not None:
                    ev = msg.event.event_type
                    if ev == "interrupted":
                        combine.vider_haut_parleur()
                    # Mesure du temps de réponse, étape par étape (affichée dans le terminal).
                    elif ev == "push_to_llm":
                        chrono["llm"] = time.monotonic()
                    elif ev == "first_word" and chrono.get("llm"):
                        chrono["mot"] = time.monotonic()
                        log.info("  réflexion : premier mot après %.1f s", chrono["mot"] - chrono["llm"])
                    elif ev == "first_tts_audio" and chrono.get("llm"):
                        log.info(
                            "  voix : premier son après %.1f s (total depuis votre fin de phrase : %.1f s)",
                            time.monotonic() - chrono.get("mot", chrono["llm"]),
                            time.monotonic() - chrono.get("fin_phrase", chrono["llm"]),
                        )
                        chrono.clear()
                elif t == "tool_call":
                    outil = gradbot.ToolHandle(msg.tool_call_handle, msg.tool_call)
                    if outil.name == "raccrocher":
                        log.info("L'agent raccroche.")
                        t_raccrocher[0] = time.monotonic()
                        raccrocher_demande.set()
                        await outil.send_json({"statut": "L'appel va se terminer."})
                    else:
                        await outil.send_error(f"Outil inconnu : {outil.name}")
            fin.set()

        async def surveiller():
            while not fin.is_set():
                await asyncio.sleep(0.25)
                maintenant = time.monotonic()
                parle = combine.agent_parle()
                if raccrocher_demande.is_set():
                    # On laisse l'agent finir son « au revoir » (qui peut arriver juste après).
                    a_parle_depuis = combine.hp.derniere_sortie > t_raccrocher[0]
                    if not parle and (a_parle_depuis or maintenant - t_raccrocher[0] > 8):
                        conv.raison_fin = "l'agent a raccroché"
                        break
                elif au_revoir_entendu[0] and maintenant - au_revoir_entendu[0] > 12 and not parle:
                    conv.raison_fin = "au revoir"
                    break
                elif maintenant - derniere_parole[0] > self.r.fin_apres_silence_s and not parle:
                    conv.raison_fin = "silence prolongé"
                    break
                elif maintenant - debut > self.r.duree_max_min * 60:
                    conv.raison_fin = "durée maximale atteinte"
                    break
                elif evenement_clavier.is_set():
                    evenement_clavier.clear()
                    conv.raison_fin = "touche Entrée"
                    break
            fin.set()

        taches = [asyncio.create_task(c) for c in (envoyer_micro(), recevoir(), surveiller())]
        await fin.wait()
        with contextlib.suppress(Exception):
            await entree.close()
        for tache in taches:
            tache.cancel()
        await asyncio.gather(*taches, return_exceptions=True)
        combine.vider_haut_parleur()
        log.info("Fin de l'appel (%s), %d échanges.", conv.raison_fin or "fin du flux", len(conv.echanges))
        return conv
