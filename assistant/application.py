"""Le serveur de la page web : assistants, conversations et discussion (voir session.py)."""

from __future__ import annotations

import logging
import pathlib
import re

import fastapi
import fastapi.responses

from .conversations import Conversation, Conversations
from .profils import Profil, Profils
from .reglages import Reglages

log = logging.getLogger("assistant")

DOSSIER_STATIC = pathlib.Path(__file__).parent / "static"


class Application:
    """Ce que partagent toutes les conversations : le cerveau, la voix, les réglages."""

    def __init__(self, reglages: Reglages, cerveau, voix, detecteur_factory, reparer=None):
        self.reglages = reglages
        self.cerveau = cerveau
        self.voix = voix
        self.nouveau_detecteur = detecteur_factory  # un détecteur de parole par conversation
        self.reparer = reparer  # redémarre llama.cpp s'il s'arrête
        self.profils = Profils(reglages.dossier_assistants)
        self.conversations = Conversations(reglages.dossier_conversations)


def creer_app(app_etat: Application) -> fastapi.FastAPI:
    from .session import deroule

    app = fastapi.FastAPI()

    def erreur(message: str, code: int = 400):
        return fastapi.responses.JSONResponse({"erreur": message}, status_code=code)

    @app.get("/")
    async def index():
        return fastapi.responses.FileResponse(DOSSIER_STATIC / "index.html",
                                              headers={"Cache-Control": "no-store"})

    @app.get("/api/infos")
    async def infos():
        r = app_etat.reglages
        return {
            "modele": r.llamacpp_modele.name,
            "voix": list(getattr(app_etat.voix, "voices", [])),
            "voix_disponible": bool(getattr(app_etat.voix, "voices", [])),
            "dossier_assistants": str(r.dossier_assistants),
            "dossier_conversations": str(r.dossier_conversations),
        }

    # ------------------------------------------------------------ assistants

    @app.get("/api/assistants")
    async def assistants():
        return [p.en_dict() for p in app_etat.profils.lister()]

    def profil_depuis(donnees: dict, ident: str = "") -> Profil:
        return Profil(
            id=ident,
            nom=str(donnees.get("nom") or "").strip(),
            consignes=str(donnees.get("consignes") or ""),
            description=str(donnees.get("description") or "").strip(),
            voix=str(donnees.get("voix") or "").strip(),
            dossier_documents=str(donnees.get("dossier_documents") or "").strip(),
        )

    @app.post("/api/assistants")
    async def creer_assistant(donnees: dict = fastapi.Body(...)):
        try:
            return app_etat.profils.enregistrer(profil_depuis(donnees)).en_dict()
        except ValueError as exc:
            return erreur(str(exc))

    @app.put("/api/assistants/{ident}")
    async def modifier_assistant(ident: str, donnees: dict = fastapi.Body(...)):
        if app_etat.profils.lire(ident) is None:
            return erreur("Assistant introuvable.", 404)
        try:
            return app_etat.profils.enregistrer(profil_depuis(donnees, ident)).en_dict()
        except ValueError as exc:
            return erreur(str(exc))

    @app.delete("/api/assistants/{ident}")
    async def supprimer_assistant(ident: str):
        try:
            app_etat.profils.supprimer(ident)
        except ValueError as exc:
            return erreur(str(exc))
        return {"ok": True}

    # ---------------------------------------------------------- conversations

    @app.get("/api/conversations")
    async def conversations():
        return app_etat.conversations.lister()

    @app.get("/api/conversations/{ident}")
    async def conversation(ident: str):
        try:
            conv = app_etat.conversations.charger(ident)
        except ValueError:
            conv = None
        if conv is None:
            return erreur("Conversation introuvable.", 404)
        return conv.__dict__

    @app.patch("/api/conversations/{ident}")
    async def renommer(ident: str, donnees: dict = fastapi.Body(...)):
        try:
            app_etat.conversations.renommer(ident, str(donnees.get("titre") or ""))
        except ValueError as exc:
            return erreur(str(exc), 404)
        return {"ok": True}

    @app.delete("/api/conversations/{ident}")
    async def supprimer(ident: str):
        try:
            app_etat.conversations.supprimer(ident)
        except ValueError as exc:
            return erreur(str(exc))
        return {"ok": True}

    @app.post("/api/conversations")
    async def nouvelle(donnees: dict = fastapi.Body(...)):
        ident = str(donnees.get("assistant") or "")
        profil = app_etat.profils.lire(ident) or app_etat.profils.par_defaut()
        return app_etat.conversations.creer(profil.id).__dict__

    # ------------------------------------------------------------ discussion

    @app.websocket("/ws/{ident}")
    async def discussion(ws: fastapi.WebSocket, ident: str, assistant: str = ""):
        if not re.fullmatch(r"[\w-]+", ident):
            await ws.close(code=4000)
            return
        conv = app_etat.conversations.charger(ident)
        if conv is None:  # nouvelle conversation (enregistrée au premier message)
            profil = app_etat.profils.lire(assistant) or app_etat.profils.par_defaut()
            conv = Conversation(id=ident, assistant=profil.id)
        else:
            profil = app_etat.profils.lire(conv.assistant) or app_etat.profils.par_defaut()
        log.info("Conversation « %s » avec « %s ».", conv.titre, profil.nom)
        await deroule(app_etat, ws, conv, profil)

    return app
