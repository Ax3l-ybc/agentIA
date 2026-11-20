"""Une conversation avec un assistant, à l'écrit ou à la voix (WebSocket /ws/{conversation}).

Protocole :
- page → serveur :
  {"type": "texte", "contenu": "…", "lire": true|false, "fichiers": [id…]}
                                                         message écrit (lire : réponse lue à voix haute ;
                                                         fichiers : pièces jointes envoyées avant par
                                                         POST /api/fichiers)
  {"type": "voix", "active": true|false}                 conversation à la voix (micro)
  binaire : le micro (PCM 16 bits mono, 16 kHz), quand la voix est active
  {"type": "stop"}                                       interrompre la réponse en cours
  {"type": "lecture_finie"}                              la page a fini de jouer la voix
- serveur → page :
  {"type": "config", "taux": 24000}
  {"type": "etat", "etat": "pret" | "reflexion" | "parle" | "ecoute"}
  {"type": "utilisateur", "id": n, "texte": "…"}         une phrase dite (puis retranscrite)
  {"type": "debut"}, {"type": "morceau", "texte": "…"}, {"type": "fin", "texte": "…"}
  {"type": "outil", "texte": "…"}                        l'assistant consulte un document
  {"type": "titre", "titre": "…"}
  {"type": "silence"}                                    couper la voix en cours
  binaire : la voix (PCM 16 bits mono, au taux annoncé)
  {"type": "erreur", "message": "…"}

La réponse arrive mot à mot (affichée au fur et à mesure) ; à la voix, chaque phrase est dite
par Phonon dès qu'elle est complète. Pour répondre vite, on reprend le moteur du directeur de
mémoire : llama.cpp garde la conversation en mémoire d'un tour à l'autre, et à la voix la
réponse est préparée dès 0,4 s de silence (envoi anticipé).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING

import fastapi
import numpy as np

from .cerveau import wav_16k
from .conversations import Conversation
from .documents import Documents
from .fichiers import bloc_pour_le_modele
from .moteurs import float_vers_pcm16, pcm16_vers_float
from .phrases import DecoupeurPhrases
from .profils import Profil
from .texte_voix import couper_phrases, nettoyer_pour_voix

if TYPE_CHECKING:
    from .application import Application

log = logging.getLogger("assistant.session")

MAX_TOURS_OUTILS = 3
DOCUMENTS_EN_ENTIER_SI_MOINS_DE = 20000  # caractères
EXCUSE = "Désolé, j'ai eu un petit souci. Peux-tu répéter ?"
CONSIGNE_VOCALE = ("(Message vocal de l'utilisateur. Réponds-lui à l'oral : phrases courtes et "
                   "naturelles, sans Markdown, sans liste ni symbole. Ne répète pas ses paroles.)")
DEBUT = "(Début de la conversation.)"

OUTILS_DOCUMENTS = [
    {"type": "function", "function": {
        "name": "lister_documents", "description": "Liste les documents disponibles.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "lire_document", "description": "Lit un document (par son nom).",
        "parameters": {"type": "object", "properties": {
            "nom": {"type": "string", "description": "Nom du document"}}, "required": ["nom"]}}},
    {"type": "function", "function": {
        "name": "chercher_dans_documents",
        "description": "Cherche des mots dans tous les documents et renvoie les passages.",
        "parameters": {"type": "object", "properties": {
            "requete": {"type": "string", "description": "Mots à chercher"}}, "required": ["requete"]}}},
]


def construire_systeme(profil: Profil, documents: Documents | None) -> str:
    morceaux = [
        profil.consignes.strip(),
        "# Cadre (en plus des consignes ci-dessus)\n"
        f"- Nous sommes le {datetime.date.today():%d/%m/%Y}.\n"
        "- Tu fonctionnes entièrement sur l'ordinateur de l'utilisateur, sans accès à internet : "
        "si on te demande une information récente, dis que tu ne peux pas la vérifier.\n"
        "- Dans une réponse écrite, tu peux utiliser du Markdown (titres, listes, gras, code).\n"
        "- Les messages entre parenthèses ne viennent pas de l'utilisateur : ce sont des "
        "indications sur la conversation.",
    ]
    if documents is not None:
        docs = documents.lister()
        if documents.taille_totale() <= DOCUMENTS_EN_ENTIER_SI_MOINS_DE:
            morceaux.append("# Documents de l'utilisateur (lecture seule)\n"
                            "Appuie-toi dessus quand c'est utile, en citant leur nom.\n\n"
                            + documents.tout_le_texte())
        else:
            noms = "\n".join(f"- {rel}" for rel, _ in docs[:200])
            morceaux.append(
                "# Documents de l'utilisateur (lecture seule)\n"
                "Pour répondre, tu peux consulter ses documents avec les outils lire_document et "
                "chercher_dans_documents, et citer leur nom. Tu ne peux pas les modifier.\n"
                f"Liste des documents :\n{noms}"
            )
    return "\n\n".join(morceaux)


def contenu_pour_le_modele(texte: str, fichiers: list[dict] | None) -> str:
    """Le message tel qu'il est donné au modèle : le texte, puis celui des fichiers joints."""
    if not fichiers:
        return texte
    return (texte or "Voici le document joint.") + "\n\n" + bloc_pour_le_modele(fichiers)


def historique_depuis(conv: Conversation) -> list[dict]:
    roles = {"utilisateur": "user", "assistant": "assistant"}
    return [{"role": roles[m["role"]],
             "content": contenu_pour_le_modele(m["texte"], m.get("fichiers")) or "(message vocal inaudible)"}
            for m in conv.messages if m.get("role") in roles]


class Session:
    def __init__(self, app: "Application", ws: fastapi.WebSocket, conv: Conversation, profil: Profil):
        self.app, self.ws, self.conv, self.profil = app, ws, conv, profil
        self.cerveau = app.cerveau
        dossier = profil.dossier()
        self.documents = Documents(dossier) if dossier is not None and dossier.is_dir() else None
        self.outils = OUTILS_DOCUMENTS if self.documents is not None else None
        self.systeme = construire_systeme(profil, self.documents)
        self.historique = historique_depuis(conv)
        self.transcriptions: set[asyncio.Task] = set()
        self.fin_voix = 0.0  # moment où la voix déjà envoyée aura fini d'être jouée
        self.n_phrase = 0
        self.file: asyncio.Queue = asyncio.Queue()
        self.voix_active = False
        self.en_cours: tuple[asyncio.Task, threading.Event] | None = None  # réponse en cours
        self.anticipe: tuple | None = None  # (tâche, feu vert, arrêt, audio, numéros)
        r = app.reglages
        anticipation = r.anticipation_s if 0 < r.anticipation_s < r.pause_s else None
        self.decoupeur = DecoupeurPhrases(app.nouveau_detecteur(), pause_s=r.pause_s,
                                          anticipation_s=anticipation)
        self.decoupeur.sur_provisoire = lambda audio, silence: self.file.put_nowait(
            ("provisoire", audio, time.monotonic() - silence, None))
        self.decoupeur.sur_reprise = lambda: self.file.put_nowait(("reprise", None, 0.0, None))

    # ----------------------------------------------------------- envoi à la page

    async def envoyer(self, donnees: dict) -> None:
        with contextlib.suppress(Exception):
            await self.ws.send_text(json.dumps(donnees, ensure_ascii=False))

    def parle(self) -> bool:
        return time.monotonic() < self.fin_voix + 0.4

    async def dire(self, texte: str, arret: threading.Event) -> None:
        """Synthétise une phrase avec la voix de l'assistant et l'envoie à la page."""
        voix = self.app.voix
        boucle = asyncio.get_running_loop()
        file: asyncio.Queue = asyncio.Queue()

        def produire():
            try:
                for m in voix.parler(texte, arret, self.profil.voix):
                    boucle.call_soon_threadsafe(file.put_nowait, m)
            except Exception:  # noqa: BLE001
                log.exception("Erreur de synthèse vocale")
            finally:
                boucle.call_soon_threadsafe(file.put_nowait, None)

        threading.Thread(target=produire, daemon=True).start()
        while (m := await file.get()) is not None:
            if arret.is_set():
                continue
            try:
                await self.ws.send_bytes(float_vers_pcm16(m))
            except Exception:  # noqa: BLE001 (page fermée)
                arret.set()
                return
            self.fin_voix = max(time.monotonic(), self.fin_voix) + len(m) / voix.taux

    # ------------------------------------------------------------ mémoire

    def _faire_de_la_place(self, secondes_audio: float) -> None:
        """Si la mémoire de travail de llama.cpp va déborder (très longue conversation), oublie
        les plus vieux échanges. On en oublie beaucoup d'un coup : le cerveau doit alors tout
        relire, et c'est mieux que ça n'arrive qu'une fois."""
        def estimation() -> float:  # environ 3 caractères par jeton, 25 jetons/s d'audio
            texte = len(self.systeme) + sum(len(str(m.get("content"))) for m in self.historique)
            return texte / 3 + 600 + secondes_audio * 25 + 800

        limite = self.app.reglages.llamacpp_contexte
        if estimation() < 0.85 * limite:
            return
        oublies = 0
        while len(self.historique) > 2 and estimation() > 0.6 * limite:
            del self.historique[0:2]
            oublies += 1
        log.info("  mémoire de travail presque pleine : %d anciens échanges oubliés", oublies)

    def prechauffer(self) -> None:
        """Fait lire à l'avance les consignes (et l'historique) : la première réponse est rapide."""
        messages = [{"role": "system", "content": self.systeme}, *self.historique,
                    {"role": "user", "content": DEBUT}]
        debut = time.monotonic()
        for _ in self.cerveau.discuter_en_flux(messages, self.outils, max_mots=1):
            pass
        log.info("Consignes de « %s » lues à l'avance (%.1f s).", self.profil.nom, time.monotonic() - debut)

    # ------------------------------------------------------------ une réponse

    async def repondre(self, contenu: str | None = None, audio: np.ndarray | None = None,
                       lire: bool = False, fichiers: list[dict] | None = None, fin_phrase: float | None = None,
                       ids: list[int] | None = None, feu_vert: asyncio.Event | None = None,
                       arret: threading.Event | None = None) -> str:
        """Répond à un message écrit (`contenu`) ou à une phrase dite (`audio`).

        Envoi anticipé : avec `feu_vert`, la réponse est préparée mais rien n'est montré, dit,
        exécuté ni gardé tant que le feu n'est pas vert (la phrase était vraiment finie).
        """
        arret = arret or threading.Event()
        await self.attendre_transcriptions(15)
        vocal = audio is not None
        if vocal:
            wav = wav_16k(audio)
            actuel = self.cerveau.message_vocal(wav, CONSIGNE_VOCALE)
        else:
            actuel = {"role": "user", "content": contenu_pour_le_modele(contenu, fichiers)}
        self._faire_de_la_place(len(audio) / 16000 if vocal else 0.0)
        messages = [{"role": "system", "content": self.systeme}, *self.historique, actuel]
        debut = time.monotonic()
        if feu_vert is None:
            await self.envoyer({"type": "etat", "etat": "reflexion"})
        if not vocal:  # le message écrit est gardé tout de suite (titre de la conversation)
            self._garder_utilisateur(contenu, "texte", fichiers)

        texte_total, interrompue = "", False
        etat = {"premier": True, "commence": False, "code": False}
        try:
            for _ in range(MAX_TOURS_OUTILS + 1):
                texte, appels, erreur = await self._flux(messages, debut, fin_phrase, etat,
                                                         vocal or lire, feu_vert, arret)
                texte_total += texte
                if feu_vert is not None:
                    await feu_vert.wait()  # ni excuse ni outil avant que la phrase soit finie
                if erreur is not None:
                    log.error("Le cerveau n'a pas répondu : %s", erreur)
                    if self.app.reparer is not None and not texte_total:
                        await asyncio.to_thread(self.app.reparer)
                        texte_total = EXCUSE
                        await self._montrer(EXCUSE, etat)
                        if vocal or lire:
                            await self.dire(EXCUSE, arret)
                    break
                if not appels:
                    break
                messages.append({
                    "role": "assistant", "content": texte,
                    "tool_calls": [{"id": a["id"], "type": "function", "function": {
                        "name": a["outil"], "arguments": json.dumps(a["arguments"], ensure_ascii=False)}}
                        for a in appels],
                })
                for a in appels:
                    resultat = await asyncio.to_thread(self._executer_outil, a["outil"], a["arguments"])
                    messages.append({"role": "tool", "tool_call_id": a["id"],
                                     "content": json.dumps(resultat, ensure_ascii=False)[:12000]})
        except asyncio.CancelledError:
            if feu_vert is not None and not feu_vert.is_set():
                raise  # réponse préparée à l'avance puis abandonnée : rien n'est gardé
            interrompue = True  # bouton « stop » : on garde ce qui a déjà été écrit

        texte_total = texte_total.strip()
        if interrompue:
            texte_total = (texte_total + " …").strip() if texte_total else "(réponse interrompue)"
        self.historique.append({"role": "user", "content": actuel["content"] if not vocal else "(message vocal)"})
        message_histo = self.historique[-1]
        self.historique.append({"role": "assistant", "content": texte_total or "…"})
        if vocal:
            numero = (ids or [self._nouvel_id()])[0]
            for autre in (ids or [])[1:]:  # phrases regroupées : une seule bulle
                await self.envoyer({"type": "utilisateur", "id": autre, "texte": ""})
            message = self._garder_utilisateur("", "voix")
            self._lancer_transcription(wav, message_histo, message, numero)
        self.conv.ajouter("assistant", texte_total, "voix" if vocal else "texte")
        self.app.conversations.enregistrer(self.conv)
        await self.envoyer({"type": "fin", "texte": texte_total, "interrompue": interrompue,
                            "index": len(self.conv.messages) - 1})
        return texte_total

    def _garder_utilisateur(self, texte: str, mode: str, fichiers: list[dict] | None = None) -> dict:
        titre = self.conv.titre
        # Un fichier envoyé sans texte donne son nom comme titre à la conversation.
        message = self.conv.ajouter("utilisateur", texte or (fichiers[0]["nom"] if fichiers else ""), mode)
        if fichiers:
            message["texte"] = texte
            message["fichiers"] = fichiers
        if self.conv.titre != titre:
            asyncio.get_running_loop().create_task(self.envoyer({"type": "titre", "titre": self.conv.titre}))
        self.app.conversations.enregistrer(self.conv)
        return message

    async def _montrer(self, texte: str, etat: dict) -> None:
        if not etat["commence"]:
            etat["commence"] = True
            await self.envoyer({"type": "debut"})
        await self.envoyer({"type": "morceau", "texte": texte})

    async def _flux(self, messages, debut, fin_phrase, etat, dire_a_voix_haute, feu_vert, arret):
        """Une réponse en flux : chaque morceau est montré, chaque phrase dite si besoin."""
        boucle = asyncio.get_running_loop()
        file: asyncio.Queue = asyncio.Queue()

        def produire():
            try:
                for m in self.cerveau.discuter_en_flux(list(messages), self.outils, arret=arret):
                    if arret.is_set():
                        break
                    boucle.call_soon_threadsafe(file.put_nowait, m)
            except Exception as exc:  # noqa: BLE001
                boucle.call_soon_threadsafe(file.put_nowait, exc)
            finally:
                boucle.call_soon_threadsafe(file.put_nowait, None)

        threading.Thread(target=produire, daemon=True).start()
        texte, tampon, appels, erreur = "", "", [], None
        while True:
            m = await file.get()
            if feu_vert is not None and not feu_vert.is_set():
                await feu_vert.wait()  # rien n'est montré ni dit avant que la phrase soit finie
            if isinstance(m, Exception):
                erreur = m
                continue
            phrases: list[str] = []
            if m is None:
                if tampon.strip():
                    phrases.append(tampon)
                tampon = ""
            elif "outil" in m:
                appels.append(m)
                await self.envoyer({"type": "outil", "texte": decrire_outil(m["outil"], m["arguments"])})
            else:
                if not texte and etat["premier"]:
                    log.info("  réflexion : premier mot après %.1f s", time.monotonic() - debut)
                texte += m["texte"]
                await self._montrer(m["texte"], etat)
                if dire_a_voix_haute:
                    tampon += m["texte"]
                    phrases, tampon = couper_phrases(tampon, etat["premier"])
                    if etat["premier"] and not phrases:
                        # Phonon parle dès le premier morceau : on commence à la première virgule.
                        i = tampon.find(",", 6)
                        if i > 0:
                            phrases, tampon = [tampon[: i + 1]], tampon[i + 1 :].lstrip()
            for p in phrases:
                if p.strip().startswith("```"):  # le code ne se lit pas à voix haute
                    etat["code"] = not etat["code"]
                    continue
                propre = nettoyer_pour_voix(p)
                if etat["code"] or not re.search(r"\w", propre) or arret.is_set():
                    continue
                if etat["premier"]:
                    depuis = time.monotonic() - (fin_phrase or debut)
                    log.info("  voix : premier son (total depuis la fin de la phrase : %.1f s)", depuis)
                    await self.envoyer({"type": "etat", "etat": "parle"})
                etat["premier"] = False
                await self.dire(propre, arret)
            if texte and etat["premier"] and not dire_a_voix_haute:
                etat["premier"] = False
            if m is None:
                break
        return texte, appels, erreur

    def _executer_outil(self, nom: str, args: dict) -> dict:
        log.info("Outil %s %s", nom, args)
        if self.documents is None:
            return {"erreur": "Aucun dossier de documents pour cet assistant."}
        if nom == "lister_documents":
            return {"documents": [rel for rel, _ in self.documents.lister()]}
        if nom == "lire_document":
            return self.documents.lire(str(args.get("nom", "")))
        if nom == "chercher_dans_documents":
            return self.documents.chercher(str(args.get("requete", "")))
        return {"erreur": f"Outil inconnu : {nom}"}

    # ------------------------------------------------------ retranscriptions

    def _nouvel_id(self) -> int:
        self.n_phrase += 1
        return self.n_phrase

    def _lancer_transcription(self, wav: bytes, message_histo: dict, message: dict, numero: int) -> None:
        async def tache():
            t0 = time.monotonic()
            try:
                texte = await asyncio.to_thread(self.cerveau.transcrire, wav)
            except Exception as exc:  # noqa: BLE001
                log.warning("Retranscription impossible : %s", exc)
                texte = ""
            message_histo["content"] = texte or "(message vocal inaudible)"
            message["texte"] = texte
            titre = self.conv.titre
            if self.conv.titre == "Nouvelle conversation" and texte:
                from .conversations import titre_depuis

                self.conv.titre = titre_depuis(texte)
            self.app.conversations.enregistrer(self.conv)
            if self.conv.titre != titre:
                await self.envoyer({"type": "titre", "titre": self.conv.titre})
            log.info("  vous (retranscrit en %.1f s) : %s", time.monotonic() - t0, texte)
            await self.envoyer({"type": "utilisateur", "id": numero, "texte": texte or "(inaudible)"})

        t = asyncio.create_task(tache())
        self.transcriptions.add(t)
        t.add_done_callback(self.transcriptions.discard)

    async def attendre_transcriptions(self, delai: float) -> None:
        if self.transcriptions:
            await asyncio.wait(set(self.transcriptions), timeout=delai)

    # ------------------------------------------------------ interruption

    def interrompre(self) -> None:
        """Bouton « stop » : arrête la réponse en cours et la voix."""
        self.abandonner_anticipe()
        if self.en_cours is not None:
            tache, arret = self.en_cours
            self.cerveau.interrompre(arret)
            tache.cancel()
        self.fin_voix = 0.0

    def abandonner_anticipe(self) -> None:
        if self.anticipe is not None:
            tache, _, arret, _, _ = self.anticipe
            self.cerveau.interrompre(arret)  # coupe net la réponse préparée pour rien
            tache.cancel()
            self.anticipe = None

    async def _executer(self, coro, arret: threading.Event) -> None:
        tache = asyncio.create_task(coro)
        self.en_cours = (tache, arret)
        try:
            await tache
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
        finally:
            self.en_cours = None

    async def fin_de_parole(self) -> None:
        """Attend que la voix soit jouée, puis annonce qu'on écoute (ou qu'on est prêt)."""
        while self.parle():
            await asyncio.sleep(0.05)
        await self.envoyer({"type": "etat", "etat": "ecoute" if self.voix_active else "pret"})

    # ------------------------------------------------------ déroulement

    async def boucle(self) -> None:
        """Traite, dans l'ordre, les messages écrits et les phrases dites."""
        while True:
            genre, donnee, fin_phrase, numero = await self.file.get()
            if genre == "texte":
                self.abandonner_anticipe()
                contenu, lire, fichiers = donnee
                arret = threading.Event()
                await self._executer(self.repondre(contenu=contenu, lire=lire, fichiers=fichiers, arret=arret),
                                     arret)
                await self.fin_de_parole()
            elif genre == "provisoire":
                # La personne s'est peut-être arrêtée de parler : on prépare déjà la réponse.
                self.abandonner_anticipe()
                feu, arret, ids = asyncio.Event(), threading.Event(), []
                tache = asyncio.create_task(self.repondre(audio=donnee, fin_phrase=fin_phrase, ids=ids,
                                                          feu_vert=feu, arret=arret))
                self.anticipe = (tache, feu, arret, donnee, ids)
            elif genre == "reprise":  # finalement non : elle continue sa phrase
                if self.anticipe is not None:
                    log.info("  (réponse préparée à l'avance abandonnée : vous continuez)")
                self.abandonner_anticipe()
            elif genre == "phrase":
                if self.anticipe is not None:
                    tache, feu, arret, audio_anticipe, ids = self.anticipe
                    if len(audio_anticipe) == len(donnee) and self.file.empty():
                        self.anticipe = None
                        ids.append(numero)
                        feu.set()  # c'était bien la fin de la phrase : la réponse est déjà prête
                        self.en_cours = (tache, arret)
                        try:
                            await tache
                        except asyncio.CancelledError:
                            pass
                        finally:
                            self.en_cours = None
                        await self.fin_de_parole()
                        continue
                    self.abandonner_anticipe()
                audio, ids = donnee, [numero]
                while not self.file.empty():  # phrases arrivées entre-temps : regroupées
                    g, suite, f, n = self.file.get_nowait()
                    if g == "phrase":
                        audio, fin_phrase = np.concatenate([audio, suite]), f
                        ids.append(n)
                    elif g == "texte":
                        self.file.put_nowait((g, suite, f, n))
                        break
                arret = threading.Event()
                await self._executer(self.repondre(audio=audio, fin_phrase=fin_phrase, ids=ids, arret=arret),
                                     arret)
                await self.fin_de_parole()

    async def recevoir(self) -> None:
        """Lit ce qu'envoie la page : texte, micro, commandes."""
        while True:
            try:
                message = await self.ws.receive()
            except (fastapi.WebSocketDisconnect, RuntimeError):
                return
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                if not self.voix_active or self.parle() or self.en_cours is not None:
                    self.decoupeur.reinitialiser()  # l'assistant parle ou réfléchit : on n'écoute pas
                    continue
                for p in self.decoupeur.ajouter(pcm16_vers_float(message["bytes"])):
                    numero = self._nouvel_id()
                    await self.envoyer({"type": "utilisateur", "id": numero, "texte": "…"})
                    await self.envoyer({"type": "etat", "etat": "reflexion"})
                    self.file.put_nowait(("phrase", p, time.monotonic() - self.app.reglages.pause_s, numero))
                continue
            try:
                donnees = json.loads(message.get("text") or "{}")
            except json.JSONDecodeError:
                continue
            genre = donnees.get("type")
            if genre == "texte":
                fichiers = self.app.prendre_fichiers(donnees.get("fichiers") or [])
                contenu = str(donnees.get("contenu", "")).strip()
                if contenu or fichiers:
                    self.file.put_nowait(("texte", (contenu, bool(donnees.get("lire")), fichiers), None, None))
            elif genre == "voix":
                self.voix_active = bool(donnees.get("active"))
                self.decoupeur.reinitialiser()
                if not self.voix_active:
                    self.abandonner_anticipe()
                await self.envoyer({"type": "etat", "etat": "ecoute" if self.voix_active else "pret"})
            elif genre == "stop":
                self.interrompre()
                await self.envoyer({"type": "silence"})
            elif genre == "lecture_finie":
                self.fin_voix = min(self.fin_voix, time.monotonic())


def decrire_outil(nom: str, args: dict) -> str:
    if nom == "lire_document":
        return f"Lecture du document « {args.get('nom', '')} »…"
    if nom == "chercher_dans_documents":
        return f"Recherche de « {args.get('requete', '')} » dans les documents…"
    if nom == "lister_documents":
        return "Consultation de la liste des documents…"
    return f"Outil {nom}…"


async def deroule(app: "Application", ws: fastapi.WebSocket, conv: Conversation, profil: Profil) -> None:
    await ws.accept()
    session = Session(app, ws, conv, profil)
    await session.envoyer({"type": "config", "taux": app.voix.taux})
    # Le cerveau lit les consignes (et l'historique) pendant que la personne tape ou parle.
    asyncio.get_running_loop().run_in_executor(None, prechauffer_sans_erreur, session)
    await session.envoyer({"type": "etat", "etat": "pret"})
    boucle = asyncio.create_task(session.boucle())
    try:
        await session.recevoir()
    finally:
        session.interrompre()
        boucle.cancel()
        await session.attendre_transcriptions(30)
        app.conversations.enregistrer(session.conv)


def prechauffer_sans_erreur(session: Session) -> None:
    try:
        session.prechauffer()
    except Exception as exc:  # noqa: BLE001
        log.warning("Lecture à l'avance des consignes impossible : %s", exc)
