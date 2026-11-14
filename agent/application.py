"""Application web du directeur de mémoire (Gemma 4 dans llama.cpp + voix Phonon).

Ce module garde tout ce qui est propre au directeur de mémoire : les notes (lecture seule),
les consignes construites à partir du mode de conduite, les outils et la fiche de séance.
Le déroulement de la conversation à la voix est dans seance.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import pathlib
import re

import fastapi
import fastapi.responses
import fastapi.staticfiles

from . import coffre as coffre_mod
from . import consignes
from .cerveau import CerveauLlamaCpp
from .reglages import Reglages

log = logging.getLogger("agent")

DOSSIER_STATIC = pathlib.Path(__file__).parent / "static"


def outil(nom: str, description: str, proprietes: dict | None = None, requis: list | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": nom,
            "description": description,
            "parameters": {"type": "object", "properties": proprietes or {}, "required": requis or []},
        },
    }


OUTILS = [
    outil("lister_notes", "Donne la liste à jour des notes du mémoire de l'étudiant."),
    outil("lire_note", "Lit le contenu d'une note du mémoire (lecture seule). Le nom peut être approximatif.",
          {"nom": {"type": "string", "description": "Nom de la note"}}, ["nom"]),
    outil("chercher_dans_notes", "Cherche des mots-clés dans toutes les notes et renvoie les passages trouvés.",
          {"requete": {"type": "string", "description": "Mots-clés"}}, ["requete"]),
    outil("lire_fiche", "Lit une fiche de séance précédente (par sa date ou son sujet).",
          {"nom": {"type": "string", "description": "Date ou sujet de la fiche"}}, ["nom"]),
    outil("rediger_fiche_recap",
          "Rédige et enregistre la fiche de clôture de la séance (selon le modèle du mode de "
          "conduite) dans le dossier des fiches. À utiliser quand l'étudiant demande la fiche "
          "ou dit qu'on termine.",
          {"consignes": {"type": "string", "description": "Précisions éventuelles de l'étudiant pour la fiche"}}),
]


class Seance:
    """Mémoire d'une séance : transcription et notes consultées."""

    def __init__(self):
        self.debut = datetime.datetime.now()
        self.echanges: list[list[str]] = []  # [rôle, texte]
        self.notes_lues: set[str] = set()
        self.fiche_en_cours: asyncio.Task | None = None

    def ajouter(self, role: str, texte: str) -> list[str]:
        """Ajoute une réplique ; renvoie l'échange (modifiable plus tard)."""
        echange = [role, texte.strip()]
        self.echanges.append(echange)
        return echange

    def transcription(self) -> str:
        noms = {"etudiant": "Étudiant", "directeur": "Directeur"}
        lignes = []
        for r, t in self.echanges:
            if t:
                propre = re.sub(r"\s+([,.…])", r"\1", t)
                lignes.append(f"{noms[r]} : {propre}")
        return "\n".join(lignes)

    def etudiant_a_parle(self) -> bool:
        return any(r == "etudiant" and t for r, t in self.echanges)


class Agent:
    def __init__(self, reglages: Reglages, cerveau: CerveauLlamaCpp, voix, detecteur_factory):
        self.r = reglages
        self.coffre = coffre_mod.Coffre(reglages.dossier_notes, reglages.dossier_recaps)
        self.cerveau = cerveau
        self.voix = voix
        self.nouveau_detecteur = detecteur_factory  # un détecteur de parole par séance
        self.reparer = None  # redémarre llama.cpp s'il s'arrête (fourni par __main__)

    # ----------------------------------------------------------- consignes

    def mode_de_conduite(self) -> str:
        from .reglages import RACINE

        return consignes.lire_mode(self.r.mode_de_conduite or RACINE / "mode_de_conduite.md")

    def consignes_seance(self) -> str:
        texte = None
        if self.coffre.taille_totale() <= self.r.seuil_notes_entieres:
            texte = self.coffre.tout_le_texte()
        return consignes.construire(
            self.mode_de_conduite(),
            self.r.sujet_memoire,
            self.coffre.index(),
            texte,
            self.coffre.fiches_recentes(2, 4000),
            [p.stem for p in self.coffre.lister_fiches()],
        )

    def prechauffer(self) -> None:
        """Fait lire les consignes au cerveau à l'avance : le début de séance sera rapide."""
        import time

        debut = time.monotonic()
        messages = [{"role": "system", "content": self.consignes_seance()},
                    {"role": "user", "content": consignes.DEBUT_SEANCE}]
        for _ in self.cerveau.discuter_en_flux(messages, OUTILS, espace=0, max_mots=1):
            pass
        log.info("Consignes, notes et fiches lues par le cerveau en %.0f s.", time.monotonic() - debut)

    # ------------------------------------------------------------- outils

    def executer_outil(self, nom: str, args: dict, seance: Seance, lancer_fiche) -> dict:
        log.info("Outil %s %s", nom, args)
        if nom == "lister_notes":
            return {"notes": [n.relatif for n in self.coffre.lister()]}
        if nom == "lire_note":
            resultat = self.coffre.lire(str(args.get("nom", "")))
            if "note" in resultat:
                seance.notes_lues.add(resultat["note"])
            return resultat
        if nom == "chercher_dans_notes":
            return self.coffre.chercher(str(args.get("requete", "")))
        if nom == "lire_fiche":
            return self.coffre.lire_fiche(str(args.get("nom", "")))
        if nom == "rediger_fiche_recap":
            if not seance.etudiant_a_parle():
                return {"erreur": "La séance vient de commencer, rien à résumer."}
            if lancer_fiche(str(args.get("consignes", ""))):
                return {"statut": "Rédaction lancée. La fiche sera enregistrée dans le dossier des fiches."}
            return {"statut": "Une fiche est déjà en cours de rédaction."}
        return {"erreur": f"Outil inconnu : {nom}"}

    # -------------------------------------------------------------- fiche

    def rediger_fiche(self, seance: Seance, precisions: str = "") -> pathlib.Path:
        if not seance.etudiant_a_parle():
            raise ValueError("La séance n'a pas encore commencé : rien à résumer.")
        if self.coffre.taille_totale() <= self.r.seuil_notes_entieres:
            notes = self.coffre.tout_le_texte()
        else:
            notes = "\n".join(
                f"=== NOTE : {nom} ===\n{self.coffre.lire(nom, 4000).get('contenu', '')}"
                for nom in sorted(seance.notes_lues)
            )
        fiches = "\n\n".join(
            f"=== FICHE : {f['nom']} ===\n{f['contenu']}" for f in self.coffre.fiches_recentes(3)
        )
        aujourd_hui = datetime.date.today()
        demande = (
            f"FICHES DES SÉANCES PRÉCÉDENTES :\n{fiches or '(aucune : première séance)'}\n\n"
            f"MES NOTES :\n{notes[:24000] or '(aucune consultée)'}\n\n"
            f"TRANSCRIPTION DE LA SÉANCE D'AUJOURD'HUI :\n{seance.transcription()[-24000:]}\n"
        )
        if precisions:
            demande += f"\nPRÉCISIONS POUR LA FICHE : {precisions}\n"
        systeme = (
            "# MODE DE CONDUITE\n" + self.mode_de_conduite() + "\n\n"
            + consignes.CONSIGNE_FICHE.format(date=f"{aujourd_hui:%Y-%m-%d}")
        )
        reponse = self.cerveau.rediger(
            [{"role": "system", "content": systeme}, {"role": "user", "content": demande}]
        )
        nom, contenu = consignes.extraire_fiche(reponse, aujourd_hui)
        chemin = self.coffre.ecrire_recap(contenu, datetime.datetime.now(), nom)
        log.info("Fiche enregistrée : %s", chemin)
        return chemin


def creer_app(agent: Agent) -> fastapi.FastAPI:
    from .seance import deroule_seance

    app = fastapi.FastAPI(title="Directeur de mémoire")
    app.state.agent = agent

    @app.websocket("/ws/seance")
    async def ws_seance(ws: fastapi.WebSocket):
        await deroule_seance(agent, ws)

    @app.get("/api/infos")
    async def infos():
        return {
            "notes": len(agent.coffre.lister()),
            "dossier_recaps": str(agent.r.dossier_recaps),
            "modele": agent.r.llamacpp_modele.name,
        }

    app.mount("/static", fastapi.staticfiles.StaticFiles(directory=DOSSIER_STATIC), name="static")

    @app.get("/")
    async def index():
        return fastapi.responses.FileResponse(DOSSIER_STATIC / "index.html")

    return app


async def envoyer(ws: fastapi.WebSocket, donnees: dict) -> None:
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps(donnees, ensure_ascii=False))
