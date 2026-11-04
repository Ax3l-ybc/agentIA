"""Application web : l'agent vocal « directeur de mémoire » construit sur Gradbot."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import pathlib

import fastapi
import gradbot

from . import coffre as coffre_mod
from . import consignes, moteurs, ollama, serveur_voix
from .reglages import Reglages

log = logging.getLogger("agent")

DOSSIER_STATIC = pathlib.Path(__file__).parent / "static"

MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]

OUTILS = [
    gradbot.ToolDef(
        "lister_notes",
        "Donne la liste à jour des notes du mémoire de l'étudiant.",
        json.dumps({"type": "object", "properties": {}, "required": []}),
    ),
    gradbot.ToolDef(
        "lire_note",
        "Lit le contenu d'une note du mémoire (lecture seule). Le nom peut être approximatif.",
        json.dumps(
            {
                "type": "object",
                "properties": {"nom": {"type": "string", "description": "Nom de la note"}},
                "required": ["nom"],
            }
        ),
    ),
    gradbot.ToolDef(
        "chercher_dans_notes",
        "Cherche des mots-clés dans toutes les notes et renvoie les passages trouvés.",
        json.dumps(
            {
                "type": "object",
                "properties": {"requete": {"type": "string", "description": "Mots-clés"}},
                "required": ["requete"],
            }
        ),
    ),
    gradbot.ToolDef(
        "rediger_fiche_recap",
        "Rédige et enregistre la fiche récapitulative de la séance dans le dossier recaps. "
        "À utiliser uniquement quand l'étudiant le demande.",
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "consignes": {
                        "type": "string",
                        "description": "Précisions éventuelles de l'étudiant pour la fiche",
                    }
                },
                "required": [],
            }
        ),
    ),
]


def date_en_francais(d: datetime.datetime) -> str:
    return f"{d.day} {MOIS[d.month - 1]} {d.year} à {d:%Hh%M}"


class Seance:
    """Mémoire d'une séance : transcription et notes consultées."""

    def __init__(self):
        self.debut = datetime.datetime.now()
        self.echanges: list[list[str]] = []  # [rôle, texte]
        self.notes_lues: set[str] = set()
        self.fiche_en_cours: asyncio.Task | None = None

    def ajouter(self, role: str, texte: str) -> None:
        texte = texte.strip()
        if not texte:
            return
        if self.echanges and self.echanges[-1][0] == role:
            self.echanges[-1][1] += " " + texte
        else:
            self.echanges.append([role, texte])

    def transcription(self) -> str:
        noms = {"etudiant": "Étudiant", "directeur": "Directeur"}
        return "\n".join(f"{noms[r]} : {t}" for r, t in self.echanges)

    def etudiant_a_parle(self) -> bool:
        return any(r == "etudiant" for r, _ in self.echanges)


class Agent:
    def __init__(self, reglages: Reglages):
        self.r = reglages
        self.coffre = coffre_mod.Coffre(reglages.dossier_notes, reglages.dossier_recaps)
        self.modele_llm = reglages.ollama_modele
        self.config_gradbot = gradbot.config.Config(
            llm={
                "model": self.modele_llm,
                "base_url": f"{reglages.ollama_adresse}/v1",
                "api_key": "ollama",
            },
            gradium={"api_key": "local", "base_url": f"http://127.0.0.1:{reglages.port}/api"},
        )

    def utiliser_modele(self, nom: str) -> None:
        self.modele_llm = nom
        self.config_gradbot.llm.model = nom

    # ----------------------------------------------------------- configuration

    def config_seance(self) -> gradbot.SessionConfig:
        index = self.coffre.index()
        texte = None
        if self.coffre.taille_totale() <= self.r.seuil_notes_entieres:
            texte = self.coffre.tout_le_texte()
        return gradbot.SessionConfig(
            instructions=consignes.construire(self.r.sujet_memoire, index, texte),
            language=gradbot.Lang.Fr,
            tools=OUTILS,
            assistant_speaks_first=True,
            silence_timeout_s=self.r.silence_relance_s,
            flush_duration_s=0.6,
        )

    # ------------------------------------------------------------------ fiches

    async def rediger_fiche(self, seance: Seance, precisions: str = "") -> pathlib.Path:
        if not seance.etudiant_a_parle():
            raise ValueError("La séance n'a pas encore commencé : rien à résumer.")
        if self.coffre.taille_totale() <= self.r.seuil_notes_entieres:
            notes = self.coffre.tout_le_texte()
        else:
            notes = "\n".join(
                f"=== NOTE : {nom} ===\n{self.coffre.lire(nom, 4000).get('contenu', '')}"
                for nom in sorted(seance.notes_lues)
            )
        demande = (
            f"SUJET DU MÉMOIRE : {self.r.sujet_memoire or 'non précisé'}\n\n"
            f"NOTES CONSULTÉES :\n{notes[:24000] or '(aucune)'}\n\n"
            f"TRANSCRIPTION DE LA SÉANCE :\n{seance.transcription()[-24000:]}\n"
        )
        if precisions:
            demande += f"\nPRÉCISIONS DE L'ÉTUDIANT POUR LA FICHE : {precisions}\n"
        corps = await asyncio.to_thread(
            ollama.discuter,
            self.r.ollama_adresse,
            self.modele_llm,
            [
                {"role": "system", "content": consignes.CONSIGNE_FICHE},
                {"role": "user", "content": demande},
            ],
            self.r.ollama_contexte,
        )
        maintenant = datetime.datetime.now()
        liens = ", ".join(f'"[[{n}]]"' for n in sorted(seance.notes_lues))
        contenu = (
            "---\n"
            "type: recap-seance\n"
            f"date: {maintenant:%Y-%m-%d %H:%M}\n"
            f"notes_consultees: [{liens}]\n"
            "---\n\n"
            f"# Séance du {date_en_francais(seance.debut)}\n\n"
            f"{corps}\n\n"
            "---\n\n"
            "<details><summary>Transcription de la séance</summary>\n\n"
            f"{seance.transcription()}\n\n</details>\n"
        )
        chemin = await asyncio.to_thread(self.coffre.ecrire_recap, contenu, maintenant)
        log.info("Fiche enregistrée : %s", chemin)
        return chemin

    def lancer_fiche(self, seance: Seance, ws: fastapi.WebSocket, precisions: str = "") -> bool:
        """Lance la rédaction en arrière-plan. Faux si une rédaction est déjà en cours."""
        if seance.fiche_en_cours and not seance.fiche_en_cours.done():
            return False

        async def tache():
            await _envoyer(ws, {"type": "fiche", "etat": "en_cours"})
            try:
                chemin = await self.rediger_fiche(seance, precisions)
                await _envoyer(ws, {"type": "fiche", "etat": "ok", "fichier": chemin.name})
            except Exception as exc:  # noqa: BLE001
                log.exception("Échec de la fiche")
                await _envoyer(ws, {"type": "fiche", "etat": "erreur", "message": str(exc)})

        seance.fiche_en_cours = asyncio.create_task(tache())
        return True

    # ------------------------------------------------------------------ outils

    async def sur_outil(self, outil: gradbot.ToolHandle, seance: Seance, ws) -> None:
        nom, args = outil.name, outil.args
        log.info("Outil %s %s", nom, args)
        if nom == "lister_notes":
            await outil.send_json({"notes": [n.relatif for n in self.coffre.lister()]})
        elif nom == "lire_note":
            resultat = await asyncio.to_thread(self.coffre.lire, str(args.get("nom", "")))
            if "note" in resultat:
                seance.notes_lues.add(resultat["note"])
            await outil.send_json(resultat)
        elif nom == "chercher_dans_notes":
            resultat = await asyncio.to_thread(self.coffre.chercher, str(args.get("requete", "")))
            await outil.send_json(resultat)
        elif nom == "rediger_fiche_recap":
            if not seance.etudiant_a_parle():
                await outil.send_json({"erreur": "La séance vient de commencer, rien à résumer."})
            elif self.lancer_fiche(seance, ws, str(args.get("consignes", ""))):
                await outil.send_json(
                    {"statut": "Rédaction lancée. La fiche sera enregistrée dans le dossier recaps."}
                )
            else:
                await outil.send_json({"statut": "Une fiche est déjà en cours de rédaction."})
        else:
            await outil.send_error(f"Outil inconnu : {nom}")


async def _envoyer(ws: fastapi.WebSocket, donnees: dict) -> None:
    with contextlib.suppress(Exception):
        await ws.send_json(donnees)


async def boucle_seance(agent: Agent, ws: fastapi.WebSocket) -> None:
    """Session vocale (adaptée de gradbot.websocket.handle_session).

    En plus de relayer l'audio, elle garde la transcription de la séance et
    accepte le message {"type": "fiche"} envoyé par le bouton de la page.
    """
    await ws.accept()
    seance = Seance()
    cfg = agent.config_gradbot
    taches_outils: set[asyncio.Task] = set()
    try:
        debut = await ws.receive_json()
        if debut.get("type") != "start":
            await ws.close(code=4000)
            return
        entree, sortie = await gradbot.run(
            **cfg.client_kwargs,
            session_config=agent.config_seance(),
            input_format=gradbot.AudioFormat.OggOpus,
            output_format=cfg.audio_format,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Impossible de démarrer la séance")
        await _envoyer(ws, {"type": "error", "message": f"Démarrage impossible : {exc}"})
        with contextlib.suppress(Exception):
            await ws.close()
        return

    arret = asyncio.Event()

    async def boucle_entree():
        while not arret.is_set():
            try:
                brut = await ws.receive()
            except (fastapi.WebSocketDisconnect, RuntimeError):
                break
            if brut.get("type") == "websocket.disconnect":
                break
            if brut.get("bytes") is not None:
                await entree.send_audio(brut["bytes"])
                continue
            if brut.get("text") is None:
                continue
            donnees = json.loads(brut["text"])
            if donnees.get("type") == "stop":
                break
            if donnees.get("type") == "fiche":
                if not seance.etudiant_a_parle():
                    await _envoyer(
                        ws,
                        {"type": "fiche", "etat": "erreur", "message": "Rien à résumer pour l'instant."},
                    )
                elif not agent.lancer_fiche(seance, ws):
                    await _envoyer(ws, {"type": "fiche", "etat": "en_cours"})
        arret.set()
        with contextlib.suppress(Exception):
            await entree.close()

    async def boucle_sortie():
        while not arret.is_set():
            try:
                msg = await sortie.receive()
            except Exception as exc:  # noqa: BLE001
                log.exception("Erreur de session")
                await _envoyer(ws, {"type": "error", "message": str(exc)})
                break
            if msg is None:
                break
            if msg.msg_type == "tool_call":
                outil = gradbot.ToolHandle(msg.tool_call_handle, msg.tool_call)

                async def executer(o=outil):
                    try:
                        await agent.sur_outil(o, seance, ws)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("Outil %s en échec", o.name)
                        with contextlib.suppress(Exception):
                            await o.send_error(str(exc))

                t = asyncio.create_task(executer())
                taches_outils.add(t)
                t.add_done_callback(taches_outils.discard)
                continue
            if msg.msg_type == "stt_text" and msg.text:
                seance.ajouter("etudiant", msg.text)
            elif msg.msg_type == "tts_text" and msg.text:
                seance.ajouter("directeur", gradbot.schemas.sanitize(msg.text))
            schema = gradbot.schemas.from_msg(msg)
            if schema is None:
                continue
            try:
                await ws.send_json(schema.model_dump())
                if msg.msg_type == "audio":
                    await ws.send_bytes(msg.data)
            except Exception:  # noqa: BLE001
                break
        arret.set()

    try:
        await asyncio.gather(boucle_sortie(), boucle_entree())
    finally:
        for t in taches_outils:
            t.cancel()
        # Une fiche en cours de rédaction est menée à son terme.
        if seance.fiche_en_cours and not seance.fiche_en_cours.done():
            with contextlib.suppress(Exception):
                await seance.fiche_en_cours
        with contextlib.suppress(Exception):
            await ws.close()


def creer_app(reglages: Reglages, moteur_ecoute=None, moteur_voix=None, modele_llm=None):
    """Construit l'application. Les moteurs peuvent être fournis (tests)."""
    agent = Agent(reglages)

    @contextlib.asynccontextmanager
    async def cycle_de_vie(app):
        if modele_llm:
            agent.utiliser_modele(modele_llm)
        else:
            nom = await asyncio.to_thread(
                ollama.preparer, reglages.ollama_adresse, reglages.ollama_modele, reglages.ollama_contexte
            )
            agent.utiliser_modele(nom)
        serveur_voix.MOTEURS["ecoute"] = moteur_ecoute or await asyncio.to_thread(
            moteurs.MoteurEcoute, reglages.stt_modele, reglages.materiel
        )
        serveur_voix.MOTEURS["voix"] = moteur_voix or await asyncio.to_thread(
            moteurs.MoteurVoix,
            reglages.voix_modele,
            reglages.voix_nom,
            reglages.voix_vitesse,
            reglages.materiel,
        )
        n = len(agent.coffre.lister())
        log.info("%s note(s) trouvée(s) dans %s (lecture seule).", n, reglages.dossier_notes)
        log.info("Les fiches iront dans %s.", reglages.dossier_recaps)
        log.info("=== Prêt ! Ouvrez http://localhost:%s dans votre navigateur. ===", reglages.port)
        app.state.pret = True
        yield

    app = fastapi.FastAPI(title="Directeur de mémoire", lifespan=cycle_de_vie)
    app.state.pret = False
    app.state.agent = agent
    app.include_router(serveur_voix.routeur)

    @app.websocket("/ws/chat")
    async def ws_chat(ws: fastapi.WebSocket):
        await boucle_seance(agent, ws)

    @app.get("/api/infos")
    async def infos():
        return {
            "notes": len(agent.coffre.lister()),
            "dossier_recaps": str(reglages.dossier_recaps),
            "modele": agent.modele_llm,
        }

    gradbot.routes.setup(app, config=agent.config_gradbot, static_dir=DOSSIER_STATIC)
    return app
