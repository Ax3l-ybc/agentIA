"""Point d'entrée : `python -m assistant` (lancé par lancer.bat).

1. le cerveau (Gemma 4 E2B dans llama.cpp, qui lit le texte et écoute la voix) ;
2. la voix (Phonon) ;
3. la page web, ouverte dans le navigateur.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser


def premier_port_libre(depart: int, essais: int = 20) -> int | None:
    """Premier port libre à partir de `depart` sur 127.0.0.1."""
    for port in range(depart, depart + essais):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for bavard in ("httpx", "httpcore", "huggingface_hub", "uvicorn.access", "faster_whisper"):
        logging.getLogger(bavard).setLevel(logging.WARNING)
    log = logging.getLogger("assistant")

    from .reglages import DOSSIER_DONNEES, ErreurReglages, charger

    try:
        reglages = charger()
    except ErreurReglages as exc:
        log.error("PROBLÈME DE RÉGLAGES :\n%s", exc)
        return 1
    if reglages.jeton_hf:
        os.environ["HF_TOKEN"] = reglages.jeton_hf

    port = premier_port_libre(reglages.port)
    if port is None:
        log.error("PROBLÈME : les ports %s à %s sont tous occupés.", reglages.port, reglages.port + 19)
        return 1
    reglages.port = port
    port_cerveau = premier_port_libre(reglages.llamacpp_port)
    if port_cerveau is not None:
        reglages.llamacpp_port = port_cerveau

    from .cerveau import ErreurCerveau
    from .cerveaux import GestionCerveaux
    from .modeles import Choix

    gestion = GestionCerveaux(reglages)
    log.info("Étape 1/3 : le cerveau (%s).", gestion.description())
    try:
        gestion.demarrer()
    except ErreurCerveau as exc:
        if gestion.choix == Choix():
            log.error("PROBLÈME AVEC LE CERVEAU :\n%s", exc)
            return 1
        # Le cerveau choisi dans la page ne démarre plus : on revient au choix de base.
        log.error("Le cerveau choisi (%s) n'a pas démarré :\n%s\nRetour au cerveau de base.",
                  gestion.description(), exc)
        gestion.arreter()
        gestion.choix = Choix()
        try:
            gestion.demarrer()
        except ErreurCerveau as exc2:
            log.error("PROBLÈME AVEC LE CERVEAU :\n%s", exc2)
            return 1
    try:
        return lancer_application(reglages, gestion, log)
    finally:
        gestion.arreter()


def charger_voix(reglages, log):
    """La voix (Kokoro par défaut, ou Phonon), ou None : l'assistant marche alors à l'écrit."""
    from .moteurs import MoteurKokoro, MoteurVoix
    from .reglages import DOSSIER_DONNEES

    try:
        if reglages.voix_moteur == "phonon":
            return MoteurVoix(reglages.voix_modele, "", reglages.voix_vitesse)
        return MoteurKokoro(DOSSIER_DONNEES / "voix" / "kokoro", reglages.voix_vitesse)
    except Exception as exc:  # noqa: BLE001
        texte = str(exc)
        if any(m in texte for m in ("401", "403", "gated", "Gated", "token", "Unauthorized", "restricted")):
            log.warning(
                "LA VOIX N'A PAS PU ÊTRE TÉLÉCHARGÉE (Hugging Face a refusé).\n"
                "L'assistant marche quand même, à l'écrit. Pour la voix : acceptez les conditions\n"
                "du modèle Phonon sur sa page Hugging Face et indiquez votre jeton dans config.yaml\n"
                "(voir le README). Détail : %s", texte)
        else:
            log.warning("La voix n'a pas pu être chargée (%s) : l'assistant marche à l'écrit.", texte)
        return None


def detecteur_simple():
    """Détection de parole simplifiée (volume du son), si Silero ne se charge pas."""
    return lambda bloc: 0.9 if float(abs(bloc).mean()) > 0.01 else 0.1


class SansVoix:
    """Quand Phonon n'est pas disponible : l'assistant marche à l'écrit seulement."""

    taux = 24000
    voices: list[str] = []

    def parler(self, texte, arret, voix=None):
        return iter(())


def lancer_application(reglages, gestion, log) -> int:
    import uvicorn

    from .application import Application, creer_app
    from .moteurs import DetecteurSilero

    log.info("Étape 2/3 : la voix (%s).", "Phonon" if reglages.voix_moteur == "phonon" else "Kokoro")
    voix = charger_voix(reglages, log) or SansVoix()
    # Le détecteur de parole sert à la voix et aux comptes rendus de réunion.
    try:
        DetecteurSilero()
        detecteur = DetecteurSilero
    except Exception as exc:  # noqa: BLE001
        log.warning("Détecteur de parole indisponible (%s) : détection simplifiée.", exc)
        detecteur = detecteur_simple
    etat = Application(reglages, gestion, voix, detecteur)
    noms = ", ".join(p.nom for p in etat.profils.lister())
    log.info("Assistants disponibles : %s (dossier %s).", noms, reglages.dossier_assistants)
    log.info("Les conversations sont gardées dans %s.", reglages.dossier_conversations)

    log.info("Étape 3/3 : la page web.")
    app = creer_app(etat)
    adresse = f"http://127.0.0.1:{reglages.port}"

    def ouvrir_navigateur():
        for _ in range(120):  # on attend que la page réponde vraiment
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(adresse + "/api/infos", timeout=2):
                    pass
            except OSError:
                continue
            log.info("=== Prêt ! La page s'ouvre : %s ===", adresse)
            log.info("Si elle ne s'ouvre pas, copiez cette adresse dans votre navigateur.")
            if not webbrowser.open(adresse):
                log.warning("Impossible d'ouvrir le navigateur tout seul : ouvrez %s", adresse)
            return
        log.error("La page ne répond pas. Essayez d'ouvrir %s dans votre navigateur.", adresse)

    threading.Thread(target=ouvrir_navigateur, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=reglages.port, log_level="warning",
                ws_ping_interval=None, ws_ping_timeout=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
