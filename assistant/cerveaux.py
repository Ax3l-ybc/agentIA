"""Le gestionnaire des cerveaux : un modèle d'écoute, et si on le choisit, un modèle de rédaction.

- L'**écoute** (Gemma 4 E2B ou E4B) converse à la voix, retranscrit les phrases dites et les
  réunions. Sans rédacteur, il répond aussi à l'écrit.
- La **rédaction** (Gemma 4 12B, 26B-A4B ou 31B), si elle est choisie, répond aux messages
  écrits et rédige les comptes rendus. Elle tourne dans un second llama.cpp, sans module audio.

Le choix se fait dans la page (panneau « Cerveau ») et est gardé dans
%USERPROFILE%\\MonAssistant\\cerveau.json. Changer de choix télécharge les fichiers qui
manquent, puis redémarre llama.cpp (une à deux minutes), sans fermer le programme. Si le
nouveau choix ne démarre pas (mémoire insuffisante…), l'ancien est remis en route.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

from . import modeles
from .cerveau import ErreurCerveau, ServeurLlamaCpp
from .modeles import PAR_ID, Choix
from .reglages import DOSSIER_DONNEES, Reglages

log = logging.getLogger("assistant.cerveaux")


class GestionCerveaux:
    def __init__(self, r: Reglages, choix: Choix | None = None, fabrique=ServeurLlamaCpp):
        self.r = r
        self.fabrique = fabrique  # (pour les tests)
        self.choix = choix or modeles.lire_choix()
        self.ecoute: ServeurLlamaCpp | None = None
        self.redaction: ServeurLlamaCpp | None = None
        self.verrou = threading.Lock()
        self.etat = {"etape": "arret", "message": "", "progression": None}
        self.pret = threading.Event()

    # ------------------------------------------------------------- accès

    @property
    def cerveau(self):
        """Le cerveau qui écoute (voix, transcriptions)."""
        return self.ecoute.cerveau if self.ecoute else None

    @property
    def redacteur(self):
        """Le cerveau qui rédige (messages écrits, comptes rendus)."""
        return self.redaction.cerveau if self.redaction else self.cerveau

    def description(self) -> str:
        texte = PAR_ID[self.choix.ecoute].nom
        if self.choix.redacteur:
            texte += f" + {PAR_ID[self.choix.redacteur].nom}"
        return texte + (" (carte graphique)" if self.choix.carte_graphique else "")

    # ---------------------------------------------------------- démarrage

    def _serveurs(self, choix: Choix) -> tuple[ServeurLlamaCpp, ServeurLlamaCpp | None]:
        r = self.r
        couches = 99 if choix.carte_graphique else r.llamacpp_couches_gpu
        m = PAR_ID[choix.ecoute]
        ecoute = self.fabrique(r, DOSSIER_DONNEES / "llama-server.log", modele=m.chemin(),
                               mmproj=m.chemin_mmproj(), port=r.llamacpp_port, couches_gpu=couches, nom="écoute")
        redaction = None
        if choix.redacteur:
            red = PAR_ID[choix.redacteur]
            redaction = self.fabrique(r, DOSSIER_DONNEES / "llama-server-redaction.log", modele=red.chemin(),
                                      mmproj=None, port=r.llamacpp_port + 1, couches_gpu=couches,
                                      nom="rédaction")
        return ecoute, redaction

    def demarrer(self, strict: bool = False) -> None:
        """Démarre le choix. Au lancement du programme, si le rédacteur ne démarre pas, on continue
        sans lui ; lors d'un changement fait dans la page (`strict`), c'est un échec."""
        with self.verrou:
            self.etat = {"etape": "demarrage", "message": "Démarrage du cerveau…", "progression": None}
            ecoute, redaction = self._serveurs(self.choix)
            ecoute.demarrer()  # sans écoute, rien ne marche : l'erreur remonte
            self.ecoute = ecoute
            if redaction is not None:
                try:
                    redaction.demarrer()
                    self.redaction = redaction
                except ErreurCerveau as exc:
                    if strict:
                        redaction.arreter()
                        raise
                    log.error("Le cerveau de rédaction n'a pas démarré (%s) : l'écoute rédige aussi.", exc)
                    redaction.arreter()
                    self.choix = dataclasses.replace(self.choix, redacteur=None)
            self.etat = {"etape": "pret", "message": f"Cerveau prêt : {self.description()}.", "progression": None}
            self.pret.set()

    def arreter(self) -> None:
        for s in (self.redaction, self.ecoute):
            if s is not None:
                s.arreter()
        self.ecoute = self.redaction = None

    def reparer(self) -> None:
        """Redémarre un llama.cpp qui s'est arrêté."""
        for s in (self.ecoute, self.redaction):
            if s is not None:
                s.reparer()

    # ------------------------------------------------------- changer de choix

    def appliquer(self, nouveau: Choix) -> None:
        """Télécharge ce qui manque puis redémarre avec `nouveau` (dans un fil à part)."""
        nouveau.valider()
        threading.Thread(target=self._appliquer, args=(nouveau,), daemon=True).start()

    def _telecharger(self, choix: Choix) -> None:
        fichiers = []
        for ident in filter(None, (choix.ecoute, choix.redacteur)):
            fichiers += modeles.a_telecharger(PAR_ID[ident])
        if choix.carte_graphique:
            from . import llamacpp_gpu, materiel

            if llamacpp_gpu.serveur_installe() is None:
                genre = materiel.resume().get("genre_carte")
                if not genre:
                    raise ErreurCerveau("Aucune carte graphique utilisable n'a été trouvée.")
                self.etat = {"etape": "telechargement", "progression": None,
                             "message": "Téléchargement de llama.cpp pour la carte graphique…"}
                llamacpp_gpu.installer(genre, afficher=lambda texte: None)
        for i, (adresse, cible) in enumerate(fichiers, 1):
            nom = cible.name

            def progression(recus, total, nom=nom, i=i):
                self.etat = {"etape": "telechargement",
                             "message": f"Téléchargement {i}/{len(fichiers)} : {nom}",
                             "progression": {"recus": recus, "total": total}}

            progression(0, 0)
            modeles.telecharger_fichier(adresse, cible, progression)

    def _appliquer(self, nouveau: Choix) -> None:
        ancien = self.choix
        try:
            self.pret.clear()
            self._telecharger(nouveau)
            self.etat = {"etape": "redemarrage", "progression": None,
                         "message": "Redémarrage du cerveau (une à deux minutes)…"}
            with self.verrou:
                self.arreter()
            self.choix = nouveau
            self.demarrer(strict=True)
            modeles.ecrire_choix(self.choix)
            log.info("Nouveau cerveau : %s.", self.description())
        except Exception as exc:  # noqa: BLE001
            log.exception("Le nouveau cerveau n'a pas pu démarrer")
            message = f"Le nouveau choix n'a pas pu être appliqué : {exc}"
            with self.verrou:
                self.arreter()
            self.choix = ancien
            try:
                self.demarrer()
                message += " L'ancien cerveau a été remis en route."
            except Exception as exc2:  # noqa: BLE001
                message += f" L'ancien cerveau n'a pas redémarré non plus ({exc2}) : relancez le programme."
            self.etat = {"etape": "erreur", "message": message, "progression": None}
        finally:
            self.pret.set()

    def attendre(self, delai: float = 600) -> bool:
        """Attend que le cerveau soit prêt (après un changement). Renvoie False si trop long."""
        return self.pret.wait(delai)


def resume_page(gestion: GestionCerveaux) -> dict:
    """Tout ce qu'affiche le panneau « Cerveau » de la page."""
    from . import llamacpp_gpu, materiel

    infos = materiel.resume()
    ecoute = PAR_ID[gestion.choix.ecoute]
    catalogue = []
    for m in modeles.CATALOGUE:
        autre = None if m.ecoute else ecoute
        catalogue.append({**m.en_dict(),
                          "place": modeles.adequation(m, infos["ram_go"], infos["vram_go"], autre)})
    conseil = modeles.conseil(infos["ram_go"], infos["vram_go"])
    return {
        "materiel": infos,
        "catalogue": catalogue,
        "choix": dataclasses.asdict(gestion.choix),
        "conseil": dataclasses.asdict(conseil),
        "description": gestion.description(),
        "carte_graphique_installee": llamacpp_gpu.serveur_installe() is not None,
        "etat": gestion.etat,
        "temps": time.time(),
    }
