"""Le serveur de la page web : assistants, conversations et discussion (voir session.py)."""

from __future__ import annotations

import logging
import pathlib
import re
import secrets

import fastapi
import fastapi.responses

from .conversations import Conversation, Conversations
from .profils import Profil, Profils
from .reglages import Reglages
from .utilisateurs import Utilisateur, Utilisateurs

log = logging.getLogger("assistant")

DOSSIER_STATIC = pathlib.Path(__file__).parent / "static"


class Application:
    """Ce que partagent toutes les conversations : le cerveau, la voix, les réglages."""

    def __init__(self, reglages: Reglages, cerveau, voix, detecteur_factory, reparer=None):
        """`cerveau` : le gestionnaire des cerveaux (cerveaux.py), ou un seul cerveau (tests)."""
        from .cerveaux import GestionCerveaux

        self.reglages = reglages
        self.gestion = cerveau if isinstance(cerveau, GestionCerveaux) else None
        self._cerveau = None if self.gestion else cerveau
        self.voix = voix
        self.nouveau_detecteur = detecteur_factory  # un détecteur de parole par conversation
        self.reparer = self.gestion.reparer if self.gestion else reparer  # redémarre llama.cpp
        self.profils = Profils(reglages.dossier_assistants)
        self.conversations = Conversations(reglages.dossier_conversations)
        self.utilisateurs = Utilisateurs(reglages.dossier_utilisateurs)
        self.pieces: dict[str, dict] = {}  # fichiers joints, en attente du message qui les envoie
        self.reunions: dict = {}  # comptes rendus de réunion en cours (voir reunion.py)

    @property
    def cerveau(self):
        """Le cerveau qui écoute : voix, retranscriptions."""
        return self.gestion.cerveau if self.gestion else self._cerveau

    @property
    def redacteur(self):
        """Le cerveau qui rédige : messages écrits, comptes rendus (le même s'il n'y en a qu'un)."""
        return self.gestion.redacteur if self.gestion else self._cerveau

    def description_cerveau(self) -> str:
        return self.gestion.description() if self.gestion else self.reglages.llamacpp_modele.name

    def deposer_fichier(self, nom: str, donnees: bytes) -> dict:
        """Lit un fichier joint et garde son texte jusqu'à l'envoi du message."""
        from .fichiers import extraire_texte, tronquer

        texte = extraire_texte(nom, donnees)
        texte, tronque = tronquer(texte)
        ident = secrets.token_hex(6)
        self.pieces[ident] = {"nom": nom, "texte": texte, "tronque": tronque}
        while len(self.pieces) > 50:  # fichiers jamais envoyés : on oublie les plus vieux
            self.pieces.pop(next(iter(self.pieces)))
        return {"id": ident, "nom": nom, "caracteres": len(texte), "tronque": tronque}

    def prendre_fichiers(self, idents: list) -> list[dict]:
        """Les fichiers joints au message. Ensemble, ils tiennent dans la mémoire du modèle."""
        from .fichiers import CARACTERES_MAX, tronquer

        fichiers = [self.pieces.pop(str(i)) for i in idents if str(i) in self.pieces]
        if len(fichiers) > 1:
            part = CARACTERES_MAX // len(fichiers)
            for f in fichiers:
                f["texte"], coupe = tronquer(f["texte"], part)
                f["tronque"] = f["tronque"] or coupe
        return fichiers


def creer_app(app_etat: Application) -> fastapi.FastAPI:
    from .session import deroule

    app = fastapi.FastAPI()

    def erreur(message: str, code: int = 400):
        return fastapi.responses.JSONResponse({"erreur": message}, status_code=code)

    @app.get("/")
    async def index():
        return fastapi.responses.FileResponse(DOSSIER_STATIC / "index.html",
                                              headers={"Cache-Control": "no-store"})

    @app.get("/icone.png")
    async def icone():
        return fastapi.responses.FileResponse(DOSSIER_STATIC / "icone.png")

    @app.get("/api/infos")
    async def infos():
        r = app_etat.reglages
        return {
            "modele": app_etat.description_cerveau(),
            "reglage_cerveau": app_etat.gestion is not None,
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

    # ---------------------------------------------------------------- cerveau

    @app.get("/api/cerveau")
    async def cerveau():
        import asyncio

        from .cerveaux import resume_page

        if app_etat.gestion is None:
            return erreur("Réglage du cerveau indisponible.", 404)
        return await asyncio.to_thread(resume_page, app_etat.gestion)

    @app.get("/api/cerveau/etat")
    async def etat_cerveau():
        if app_etat.gestion is None:
            return erreur("Réglage du cerveau indisponible.", 404)
        return {**app_etat.gestion.etat, "description": app_etat.gestion.description()}

    @app.post("/api/cerveau")
    async def changer_cerveau(donnees: dict = fastapi.Body(...)):
        from .modeles import Choix

        gestion = app_etat.gestion
        if gestion is None:
            return erreur("Réglage du cerveau indisponible.", 404)
        if gestion.etat.get("etape") in ("telechargement", "redemarrage", "demarrage"):
            return erreur("Un changement de cerveau est déjà en cours.")
        choix = Choix(ecoute=str(donnees.get("ecoute") or ""), redacteur=donnees.get("redacteur") or None,
                      carte_graphique=bool(donnees.get("carte_graphique")))
        try:
            gestion.appliquer(choix)
        except ValueError as exc:
            return erreur(str(exc))
        return {"ok": True}

    # --------------------------------------------------------- fichiers joints

    @app.post("/api/fichiers")
    async def deposer(req: fastapi.Request, nom: str):
        from .fichiers import FichierIllisible

        donnees = await req.body()
        try:
            return app_etat.deposer_fichier(pathlib.Path(nom).name, donnees)
        except FichierIllisible as exc:
            return erreur(str(exc))

    # ------------------------------------------------------ comptes rendus de réunion

    def assistant_valide(ident: str) -> str:
        return (app_etat.profils.lire(ident) or app_etat.profils.par_defaut()).id

    @app.post("/api/reunions")
    async def importer_reunion(req: fastapi.Request, nom: str, assistant: str = "", utilisateur: str = ""):
        """Un fichier audio de réunion : transcription puis compte rendu, en arrière-plan."""
        import asyncio

        from .reunion import Reunion

        reunion = Reunion(app_etat, assistant_valide(assistant), pathlib.Path(nom).name,
                          utilisateur_valide(utilisateur))
        app_etat.reunions[reunion.id] = reunion
        reunion.import_ = asyncio.create_task(reunion.importer(await req.body()))
        return reunion.etat()

    @app.get("/api/reunions/{ident}")
    async def etat_reunion(ident: str):
        reunion = app_etat.reunions.get(ident)
        if reunion is None:
            return erreur("Réunion introuvable.", 404)
        return reunion.etat()

    @app.websocket("/ws-reunion")
    async def enregistrer_reunion(ws: fastapi.WebSocket, assistant: str = "", utilisateur: str = ""):
        """Enregistrement en direct : la page envoie le micro (PCM 16 bits, 16 kHz)."""
        import json

        from .moteurs import pcm16_vers_float
        from .reunion import Reunion

        await ws.accept()
        reunion = Reunion(app_etat, assistant_valide(assistant), "enregistrement", utilisateur_valide(utilisateur))
        app_etat.reunions[reunion.id] = reunion
        log.info("Enregistrement d'une réunion (%s).", reunion.id)
        await ws.send_text(json.dumps({"id": reunion.id}))
        annulee = False
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    reunion.ajouter_audio(pcm16_vers_float(message["bytes"]))
                elif '"fin"' in (message.get("text") or ""):
                    break
                elif '"annuler"' in (message.get("text") or ""):
                    annulee = True
                    break
        except (fastapi.WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if annulee:
                reunion.tache.cancel()
                reunion.etape, reunion.message = "annulee", "Enregistrement annulé."
                log.info("Enregistrement de réunion annulé.")
                return
            # Même si la page est fermée, ce qui a été enregistré donne un compte rendu.
            if reunion.etape == "enregistrement":
                reunion.etape = "transcription"
                reunion.message = "Transcription des derniers passages…"
            reunion.terminer()
            log.info("Fin de l'enregistrement (%s).", reunion.etat()["duree"])

    # ---------------------------------------------------------- conversations

    @app.get("/api/conversations")
    async def conversations(utilisateur: str | None = None):
        return app_etat.conversations.lister(utilisateur)

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

    @app.get("/api/conversations/{ident}/word")
    async def exporter_word(ident: str, message: int | None = None):
        """La conversation entière (ou une seule réponse, `message`) en document Word."""
        from .fichiers import document_word, nom_de_fichier

        try:
            conv = app_etat.conversations.charger(ident)
        except ValueError:
            conv = None
        if conv is None:
            return erreur("Conversation introuvable.", 404)
        profil = app_etat.profils.lire(conv.assistant)
        nom_assistant = profil.nom if profil else "Assistant"
        if message is not None:
            if not 0 <= message < len(conv.messages):
                return erreur("Message introuvable.", 404)
            sections = [("", conv.messages[message]["texte"])]
            titre = conv.titre
        else:
            sections = []
            for m in conv.messages:
                qui = "Moi" if m["role"] == "utilisateur" else nom_assistant
                texte = m["texte"] or "(message vocal)"
                if m.get("fichiers"):
                    texte += "\n\n" + "\n".join(f"*Fichier joint : {f['nom']}*" for f in m["fichiers"])
                sections.append((qui, texte))
            titre = conv.titre
        donnees = document_word(titre, sections)
        nom = nom_de_fichier(titre)
        return fastapi.responses.Response(
            donnees, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{nom}"'})

    @app.post("/api/conversations")
    async def nouvelle(donnees: dict = fastapi.Body(...)):
        ident = str(donnees.get("assistant") or "")
        profil = app_etat.profils.lire(ident) or app_etat.profils.par_defaut()
        return app_etat.conversations.creer(profil.id, utilisateur_valide(donnees.get("utilisateur"))).__dict__

    # ---------------------------------------------------------- utilisateurs

    def utilisateur_valide(ident) -> str:
        return ident if ident and app_etat.utilisateurs.lire(str(ident)) else ""

    @app.get("/api/utilisateurs")
    async def utilisateurs():
        return [u.en_dict() for u in app_etat.utilisateurs.lister()]

    def utilisateur_depuis(donnees: dict, ident: str = "") -> Utilisateur:
        return Utilisateur(id=ident, **{k: str(donnees.get(k) or "") for k in
                                        ("prenom", "role", "presentation", "besoins", "preferences", "autres")})

    @app.post("/api/utilisateurs")
    async def creer_utilisateur(donnees: dict = fastapi.Body(...)):
        try:
            return app_etat.utilisateurs.enregistrer(utilisateur_depuis(donnees)).en_dict()
        except ValueError as exc:
            return erreur(str(exc))

    @app.put("/api/utilisateurs/{ident}")
    async def modifier_utilisateur(ident: str, donnees: dict = fastapi.Body(...)):
        if app_etat.utilisateurs.lire(ident) is None:
            return erreur("Profil introuvable.", 404)
        try:
            return app_etat.utilisateurs.enregistrer(utilisateur_depuis(donnees, ident)).en_dict()
        except ValueError as exc:
            return erreur(str(exc))

    @app.delete("/api/utilisateurs/{ident}")
    async def supprimer_utilisateur(ident: str):
        try:
            app_etat.utilisateurs.supprimer(ident)
        except ValueError as exc:
            return erreur(str(exc), 404)
        return {"ok": True}

    # ------------------------------------------------------------ discussion

    @app.websocket("/ws/{ident}")
    async def discussion(ws: fastapi.WebSocket, ident: str, assistant: str = "", utilisateur: str = ""):
        if not re.fullmatch(r"[\w-]+", ident):
            await ws.close(code=4000)
            return
        conv = app_etat.conversations.charger(ident)
        if conv is None:  # nouvelle conversation (enregistrée au premier message)
            profil = app_etat.profils.lire(assistant) or app_etat.profils.par_defaut()
            conv = Conversation(id=ident, assistant=profil.id, utilisateur=utilisateur_valide(utilisateur))
        else:
            profil = app_etat.profils.lire(conv.assistant) or app_etat.profils.par_defaut()
        personne = app_etat.utilisateurs.lire(conv.utilisateur or utilisateur)
        log.info("Conversation « %s » avec « %s »%s.", conv.titre, profil.nom,
                 f" (pour {personne.prenom})" if personne else "")
        await deroule(app_etat, ws, conv, profil, personne)

    return app
