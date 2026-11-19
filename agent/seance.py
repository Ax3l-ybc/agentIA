"""Une séance à la voix entre la page web et le directeur de mémoire.

Protocole WebSocket (/ws/seance) :
- page → serveur : son du micro en binaire (PCM 16 bits mono, 16 kHz) ;
  messages JSON {"type": "debut"}, {"type": "fiche"}, {"type": "stop"},
  {"type": "lecture_finie"} (la page a fini de jouer la voix).
- serveur → page : voix en binaire (PCM 16 bits mono, au taux annoncé) ;
  messages JSON {"type": "config"}, {"type": "etat"}, {"type": "directeur"},
  {"type": "etudiant"}, {"type": "fiche"}, {"type": "erreur"}.

Déroulement d'un tour : la phrase de l'étudiant est repérée (détecteur de parole), son
audio est envoyé tel quel à Gemma 4 avec l'historique ; la réponse arrive en flux, chaque
phrase est dite par Phonon dès qu'elle est complète ; en arrière-plan, la phrase de
l'étudiant est retranscrite pour l'historique et la fiche. Le micro est ignoré pendant que
le directeur parle (pas d'écho de sa propre voix).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time

import fastapi
import numpy as np

from . import consignes
from .application import OUTILS, Agent, Seance, envoyer
from .cerveau import ErreurCerveau, wav_16k
from .fil import Fil
from .moteurs import float_vers_pcm16, pcm16_vers_float
from .phrases import DecoupeurPhrases
from .texte_voix import couper_phrases, nettoyer_pour_voix

log = logging.getLogger("agent.seance")

EXCUSE = "Désolé, j'ai eu un petit souci. Peux-tu répéter ce que tu viens de dire ?"
MAX_TOURS_OUTILS = 3


class Conversation:
    """L'état d'une séance en cours."""

    def __init__(self, agent: Agent, ws: fastapi.WebSocket):
        self.agent = agent
        self.ws = ws
        self.seance = Seance()
        self.systeme = agent.consignes_seance()
        self.historique: list[dict] = []  # messages texte des tours précédents
        self.transcriptions: set[asyncio.Task] = set()
        self.fin_voix = 0.0  # moment où la voix déjà envoyée aura fini d'être jouée
        self.n_phrase = 0
        self.fil = Fil()  # questions déjà posées, réactions récentes (voir fil.py)

    def _faire_de_la_place(self, secondes_audio: float) -> None:
        """Si la mémoire de travail de llama.cpp va déborder (séance très longue), oublie les
        plus vieux échanges (en gardant le début). On en oublie beaucoup d'un coup : le cerveau
        doit alors relire l'historique, et c'est mieux que ça n'arrive qu'une fois."""
        def estimation() -> float:  # environ 3 caractères par jeton, 25 jetons/s d'audio
            texte = len(self.systeme) + sum(len(str(m.get("content"))) for m in self.historique)
            return texte / 3 + 600 + secondes_audio * 25 + 400  # + outils, réponse

        limite = self.agent.r.llamacpp_contexte
        if estimation() < 0.85 * limite:
            return
        oublies = 0
        while len(self.historique) > 4 and estimation() > 0.6 * limite:
            del self.historique[2:4]  # le plus vieil échange après le début de séance
            oublies += 1
        log.info("  mémoire de travail presque pleine : %d anciens échanges oubliés", oublies)

    # ------------------------------------------------------------- voix

    def directeur_parle(self) -> bool:
        # Petite marge : le son met un instant à sortir des haut-parleurs.
        return time.monotonic() < self.fin_voix + 0.4

    async def dire(self, texte: str) -> None:
        """Synthétise une phrase et l'envoie à la page, qui la joue."""
        await envoyer(self.ws, {"type": "directeur", "texte": texte})
        voix = self.agent.voix
        boucle = asyncio.get_running_loop()
        file: asyncio.Queue = asyncio.Queue()
        arret = threading.Event()

        def produire():
            try:
                for m in voix.parler(texte, arret):
                    boucle.call_soon_threadsafe(file.put_nowait, m)
            except Exception:  # noqa: BLE001
                log.exception("Erreur de synthèse vocale")
            finally:
                boucle.call_soon_threadsafe(file.put_nowait, None)

        threading.Thread(target=produire, daemon=True).start()
        while (m := await file.get()) is not None:
            try:
                await self.ws.send_bytes(float_vers_pcm16(m))
            except Exception:  # noqa: BLE001 (page fermée)
                arret.set()
                return
            debut = max(time.monotonic(), self.fin_voix)
            self.fin_voix = debut + len(m) / voix.taux

    # ------------------------------------------------------------- un tour

    async def _lancer_fiche_async(self, precisions: str = "") -> bool:
        return self.lancer_fiche(precisions)

    def lancer_fiche(self, precisions: str = "") -> bool:
        s = self.seance
        if s.fiche_en_cours and not s.fiche_en_cours.done():
            return False

        async def tache():
            await envoyer(self.ws, {"type": "fiche", "etat": "en_cours"})
            try:
                await self.attendre_transcriptions(60)
                chemin = await asyncio.to_thread(self.agent.rediger_fiche, s, precisions)
                await envoyer(self.ws, {"type": "fiche", "etat": "ok", "fichier": chemin.name})
            except Exception as exc:  # noqa: BLE001
                log.exception("Échec de la fiche")
                await envoyer(self.ws, {"type": "fiche", "etat": "erreur", "message": str(exc)})

        s.fiche_en_cours = asyncio.create_task(tache())
        return True

    def nouvel_id(self) -> int:
        self.n_phrase += 1
        return self.n_phrase

    async def tour(self, audio: np.ndarray | None, indication: str = "",
                   fin_phrase: float | None = None, ids: list[int] | None = None,
                   feu_vert: asyncio.Event | None = None, arret: threading.Event | None = None) -> str:
        """Envoie une phrase de l'étudiant (ou une indication) et dit la réponse.

        Envoi anticipé : avec `feu_vert`, la réponse est préparée mais rien n'est dit, exécuté
        ni gardé tant que le feu n'est pas vert (l'étudiant a vraiment fini sa phrase). S'il
        reprend la parole, la tâche est annulée et `arret` interrompt le cerveau.
        """
        await self.attendre_transcriptions(15)
        cerveau = self.agent.cerveau
        if audio is not None:
            wav = wav_16k(audio)
            actuel = cerveau.message_vocal(wav, self.fil.indication() + consignes.CONSIGNE_VOCALE)
        else:
            actuel = {"role": "user", "content": indication}
        self._faire_de_la_place(len(audio) / 16000 if audio is not None else 0.0)
        messages = [{"role": "system", "content": self.systeme}, *self.historique, actuel]
        debut = time.monotonic()
        if feu_vert is None:
            await envoyer(self.ws, {"type": "etat", "etat": "reflexion"})

        reponse_totale = ""
        premier_mot = None
        for _ in range(MAX_TOURS_OUTILS + 1):
            texte, appels, erreur = await self._repondre(messages, debut, fin_phrase, premier_mot is None,
                                                         feu_vert, arret)
            if texte and premier_mot is None:
                premier_mot = True
            reponse_totale += texte
            if feu_vert is not None:
                await feu_vert.wait()  # ni excuse ni outil avant que la phrase soit vraiment finie
            if erreur is not None:
                log.error("Le cerveau n'a pas répondu : %s", erreur)
                if self.agent.reparer is not None and not reponse_totale:
                    await self.dire("Désolé, une petite seconde.")
                    try:
                        await asyncio.to_thread(self.agent.reparer)
                        await self.dire(EXCUSE)
                        reponse_totale = EXCUSE
                    except Exception as exc:  # noqa: BLE001
                        log.error("Le cerveau n'a pas pu redémarrer : %s", exc)
                break
            if not appels:
                break
            # Le modèle a demandé des outils : on les exécute et il poursuit sa réponse.
            messages.append({
                "role": "assistant",
                "content": texte,
                "tool_calls": [
                    {"id": a["id"], "type": "function",
                     "function": {"name": a["outil"], "arguments": json.dumps(a["arguments"], ensure_ascii=False)}}
                    for a in appels
                ],
            })
            boucle = asyncio.get_running_loop()

            def lancer_fiche_depuis_outil(precisions: str = "") -> bool:
                # Les outils tournent à part : la rédaction est lancée dans la boucle principale.
                return asyncio.run_coroutine_threadsafe(
                    self._lancer_fiche_async(precisions), boucle
                ).result(timeout=10)

            for a in appels:
                resultat = await asyncio.to_thread(
                    self.agent.executer_outil, a["outil"], a["arguments"], self.seance,
                    lancer_fiche_depuis_outil,
                )
                messages.append({"role": "tool", "tool_call_id": a["id"],
                                 "content": json.dumps(resultat, ensure_ascii=False)[:12000]})

        reponse_totale = self.fil.nettoyer_debut(reponse_totale.strip())
        if self.fil.noter(reponse_totale, audio is not None):
            log.info("  question déjà posée : le directeur passera à autre chose au tour suivant.")
        message_etudiant = {"role": "user", "content": indication or "(message vocal)"}
        self.historique.append(message_etudiant)
        self.historique.append({"role": "assistant", "content": reponse_totale or "…"})
        if audio is not None:
            ids = ids or [self.nouvel_id()]
            for autre in ids[1:]:  # phrases regroupées : une seule bulle à l'écran
                await envoyer(self.ws, {"type": "etudiant", "id": autre, "texte": ""})
            echange = self.seance.ajouter("etudiant", "")
            self._lancer_transcription(wav, message_etudiant, echange, ids[0])
        self.seance.ajouter("directeur", reponse_totale)
        return reponse_totale

    async def _repondre(self, messages, debut, fin_phrase, premier, feu_vert=None, arret=None):
        """Une réponse en flux ; dit chaque phrase dès qu'elle est complète."""
        boucle = asyncio.get_running_loop()
        file: asyncio.Queue = asyncio.Queue()

        def produire():
            try:
                for m in self.agent.cerveau.discuter_en_flux(list(messages), OUTILS, arret=arret):
                    if arret is not None and arret.is_set():
                        break  # réponse préparée à l'avance puis abandonnée
                    boucle.call_soon_threadsafe(file.put_nowait, m)
            except Exception as exc:  # noqa: BLE001
                boucle.call_soon_threadsafe(file.put_nowait, exc)
            finally:
                boucle.call_soon_threadsafe(file.put_nowait, None)

        threading.Thread(target=produire, daemon=True).start()
        texte, tampon, appels, erreur = "", "", [], None
        premiere_phrase = premier
        while True:
            m = await file.get()
            if feu_vert is not None and not feu_vert.is_set():
                await feu_vert.wait()  # rien n'est dit avant que la phrase soit vraiment finie
            phrases: list[str] = []
            if isinstance(m, Exception):
                erreur = m
                continue
            if m is None:
                if tampon.strip():
                    phrases.append(tampon)
                tampon = ""
            elif "outil" in m:
                appels.append(m)
            else:
                if not texte:
                    log.info("  réflexion : premier mot après %.1f s", time.monotonic() - debut)
                texte += m["texte"]
                tampon += m["texte"]
                phrases, tampon = couper_phrases(tampon, premiere_phrase)
                if premiere_phrase and not phrases:
                    # Phonon parle dès le premier morceau : inutile d'attendre la fin de la
                    # première phrase, on commence à la première virgule.
                    i = tampon.find(",", 6)
                    if i > 0:
                        phrases, tampon = [tampon[: i + 1]], tampon[i + 1 :].lstrip()
            for p in phrases:
                if premiere_phrase:
                    p = self.fil.nettoyer_debut(p)
                propre = nettoyer_pour_voix(p)
                if not re.search(r"\w", propre):
                    continue
                if premiere_phrase:
                    depuis = time.monotonic() - (fin_phrase or debut)
                    log.info("  voix : premier son (total depuis la fin de votre phrase : %.1f s)", depuis)
                    await envoyer(self.ws, {"type": "etat", "etat": "parle"})
                premiere_phrase = False
                await self.dire(propre)
            if m is None:
                break
        return texte, appels, erreur

    # ------------------------------------------------------ transcriptions

    def _lancer_transcription(self, wav: bytes, message: dict, echange: list, numero: int) -> None:
        async def tache():
            t0 = time.monotonic()
            try:
                texte = await asyncio.to_thread(self.agent.cerveau.transcrire, wav)
            except ErreurCerveau as exc:
                log.warning("Retranscription impossible : %s", exc)
                texte = ""
            message["content"] = texte or "(message vocal inaudible)"
            echange[1] = texte
            log.info("  vous (retranscrit en %.1f s) : %s", time.monotonic() - t0, texte)
            await envoyer(self.ws, {"type": "etudiant", "id": numero, "texte": texte or "(inaudible)"})

        t = asyncio.create_task(tache())
        self.transcriptions.add(t)
        t.add_done_callback(self.transcriptions.discard)

    async def attendre_transcriptions(self, delai: float) -> None:
        if self.transcriptions:
            await asyncio.wait(set(self.transcriptions), timeout=delai)


async def fin_de_parole(conv: "Conversation", ws: fastapi.WebSocket, fin: asyncio.Event) -> None:
    """Attend que la voix du directeur soit jouée, puis annonce qu'on écoute."""
    while conv.directeur_parle() and not fin.is_set():
        await asyncio.sleep(0.05)
    await envoyer(ws, {"type": "etat", "etat": "ecoute"})


async def deroule_seance(agent: Agent, ws: fastapi.WebSocket) -> None:
    await ws.accept()
    conv = Conversation(agent, ws)
    r = agent.r
    anticipation = r.anticipation_s if 0 < r.anticipation_s < r.pause_s else None
    decoupeur = DecoupeurPhrases(agent.nouveau_detecteur(), pause_s=r.pause_s, anticipation_s=anticipation)
    phrases: asyncio.Queue = asyncio.Queue()
    # Envoi anticipé : après un court silence, la phrase « probablement finie » est proposée.
    decoupeur.sur_provisoire = lambda audio, silence: phrases.put_nowait(
        ("provisoire", audio, time.monotonic() - silence, None))
    decoupeur.sur_reprise = lambda: phrases.put_nowait(("reprise", None, 0.0, None))
    fin = asyncio.Event()
    accueil_fini = asyncio.Event()
    derniere_parole = [time.monotonic()]
    await envoyer(ws, {"type": "config", "taux": agent.voix.taux})

    async def recevoir():
        while not fin.is_set():
            try:
                message = await ws.receive()
            except (fastapi.WebSocketDisconnect, RuntimeError):
                break
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if not accueil_fini.is_set() or conv.directeur_parle():
                    decoupeur.reinitialiser()  # le directeur parle : on n'écoute pas
                    continue
                for p in decoupeur.ajouter(pcm16_vers_float(message["bytes"])):
                    derniere_parole[0] = time.monotonic()
                    numero = conv.nouvel_id()
                    await envoyer(ws, {"type": "etudiant", "id": numero, "texte": "…"})
                    await envoyer(ws, {"type": "etat", "etat": "reflexion"})
                    phrases.put_nowait(("phrase", p, time.monotonic() - r.pause_s, numero))
                continue
            try:
                donnees = json.loads(message.get("text") or "{}")
            except json.JSONDecodeError:
                continue
            genre = donnees.get("type")
            if genre == "stop":
                break
            if genre == "lecture_finie":  # la page a fini de jouer la voix
                conv.fin_voix = min(conv.fin_voix, time.monotonic())
            elif genre == "fiche":
                if not conv.seance.etudiant_a_parle():
                    await envoyer(ws, {"type": "fiche", "etat": "erreur",
                                       "message": "Rien à résumer pour l'instant."})
                elif not conv.lancer_fiche():
                    await envoyer(ws, {"type": "fiche", "etat": "en_cours"})
        fin.set()

    # Réponse préparée à l'avance : (tâche, feu vert, arrêt, audio, numéros des bulles)
    anticipe: list = [None]

    def abandonner_anticipe():
        if anticipe[0] is not None:
            tache, _, arret, _, _ = anticipe[0]
            agent.cerveau.interrompre(arret)  # coupe net la réponse préparée pour rien
            tache.cancel()
            anticipe[0] = None

    reception = asyncio.create_task(recevoir())
    try:
        await conv.tour(None, consignes.DEBUT_SEANCE)
        await fin_de_parole(conv, ws, fin)
        accueil_fini.set()
        relances = 0
        derniere_activite = time.monotonic()
        while not fin.is_set():
            try:
                genre, audio, fin_phrase, numero = await asyncio.wait_for(phrases.get(), timeout=0.25)
            except asyncio.TimeoutError:
                genre, audio, numero = None, None, None
            if genre == "provisoire":
                # L'étudiant s'est peut-être arrêté : le directeur prépare déjà sa réponse.
                abandonner_anticipe()
                feu, arret, ids_anticipes = asyncio.Event(), threading.Event(), []
                tache = asyncio.create_task(conv.tour(audio, fin_phrase=fin_phrase, ids=ids_anticipes,
                                                      feu_vert=feu, arret=arret))
                anticipe[0] = (tache, feu, arret, audio, ids_anticipes)
                continue
            if genre == "reprise":  # finalement non : il continue sa phrase
                if anticipe[0] is not None:
                    log.info("  (réponse préparée à l'avance abandonnée : vous continuez)")
                abandonner_anticipe()
                continue
            if genre == "phrase" and anticipe[0] is not None:
                tache, feu, arret, audio_anticipe, ids_anticipes = anticipe[0]
                if len(audio_anticipe) == len(audio) and phrases.empty():
                    anticipe[0] = None
                    ids_anticipes.append(numero)
                    feu.set()  # c'était bien la fin de sa phrase : la réponse est déjà prête
                    relances = 0
                    await tache
                    await fin_de_parole(conv, ws, fin)
                    derniere_activite = time.monotonic()
                    continue
                abandonner_anticipe()
            ids = [numero] if numero is not None else []
            maintenant = time.monotonic()
            if conv.directeur_parle():
                derniere_activite = maintenant
            if audio is not None:
                while not phrases.empty():  # phrases arrivées pendant la réflexion : regroupées
                    g, suite, f, n = phrases.get_nowait()
                    if g == "phrase":
                        audio, fin_phrase = np.concatenate([audio, suite]), f
                        ids.append(n)
                relances = 0
                await conv.tour(audio, fin_phrase=fin_phrase, ids=ids)
                await fin_de_parole(conv, ws, fin)
                derniere_activite = time.monotonic()
                continue
            if decoupeur.en_cours or conv.directeur_parle():
                continue
            if maintenant - derniere_activite > r.silence_relance_s and relances < 2:
                relances += 1
                await conv.tour(None, consignes.RELANCE)
                await fin_de_parole(conv, ws, fin)
                derniere_activite = time.monotonic()
    except Exception:  # noqa: BLE001
        log.exception("Erreur pendant la séance")
        await envoyer(ws, {"type": "erreur", "message": "Erreur pendant la séance (voir la fenêtre noire)."})
    finally:
        abandonner_anticipe()
        fin.set()
        reception.cancel()
        await conv.attendre_transcriptions(30)
        if conv.seance.fiche_en_cours and not conv.seance.fiche_en_cours.done():
            log.info("Fin de la rédaction de la fiche avant de fermer la séance…")
            try:
                await conv.seance.fiche_en_cours
            except Exception:  # noqa: BLE001
                pass
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("Séance terminée (%d échanges).", len(conv.seance.echanges))
        # Les consignes de la prochaine séance (avec la nouvelle fiche éventuelle) sont lues
        # à l'avance par le cerveau : le prochain « bonjour » sera rapide.
        asyncio.get_running_loop().run_in_executor(None, prechauffer_sans_erreur, agent)


def prechauffer_sans_erreur(agent: Agent) -> None:
    try:
        agent.prechauffer()
    except Exception as exc:  # noqa: BLE001
        log.warning("Préparation de la prochaine séance impossible : %s", exc)
