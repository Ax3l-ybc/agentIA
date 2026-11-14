"""Point d'entrée : `python -m agent` (lancé par lancer.bat).

1. le cerveau qui écoute (Gemma 4 E2B dans llama.cpp) ;
2. la voix (Phonon) ;
3. lecture à l'avance des consignes, notes et fiches ;
4. la page web, ouverte dans le navigateur.
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
    for bavard in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub", "uvicorn.access",
                   "faster_whisper"):
        logging.getLogger(bavard).setLevel(logging.WARNING)
    log = logging.getLogger("agent")

    from .reglages import ErreurReglages, charger

    try:
        reglages = charger()
    except ErreurReglages as exc:
        log.error("PROBLÈME DE RÉGLAGES :\n%s", exc)
        return 1
    log.info("Réglages lus. Notes : %s", reglages.dossier_notes)
    if reglages.jeton_hf:
        os.environ["HF_TOKEN"] = reglages.jeton_hf
    else:
        log.warning("Aucun jeton Hugging Face dans config.yaml : la voix ne pourra pas être téléchargée.")

    port = premier_port_libre(reglages.port)
    if port is None:
        log.error("PROBLÈME : les ports %s à %s sont tous occupés.", reglages.port, reglages.port + 19)
        return 1
    if port != reglages.port:
        log.info("Le port %s est déjà pris : utilisation du port %s.", reglages.port, port)
        reglages.port = port
    port_cerveau = premier_port_libre(reglages.llamacpp_port)
    if port_cerveau is not None:
        reglages.llamacpp_port = port_cerveau

    from .cerveau import ErreurCerveau, ServeurLlamaCpp
    from .reglages import DOSSIER_DONNEES

    log.info("Étape 1/3 : le cerveau qui écoute (Gemma 4 dans llama.cpp, %s).",
             reglages.llamacpp_modele.name)
    serveur = ServeurLlamaCpp(reglages, DOSSIER_DONNEES / "llama-server.log")
    try:
        serveur.demarrer()
    except ErreurCerveau as exc:
        log.error("PROBLÈME AVEC LE CERVEAU :\n%s", exc)
        return 1

    try:
        return lancer_application(reglages, serveur, log)
    finally:
        serveur.arreter()


def lancer_application(reglages, serveur, log) -> int:
    from .moteurs import DetecteurSilero, MoteurVoix

    log.info("Étape 2/3 : la voix (Phonon).")
    try:
        voix = MoteurVoix(reglages.voix_modele, reglages.voix_nom, reglages.voix_vitesse)
        DetecteurSilero()  # vérifie que le détecteur de parole se charge
    except Exception as exc:  # noqa: BLE001
        texte = str(exc)
        if any(m in texte for m in ("401", "403", "gated", "Gated", "token", "Unauthorized", "restricted")):
            log.error(
                "TÉLÉCHARGEMENT DE LA VOIX REFUSÉ PAR HUGGING FACE.\n"
                "Vérifiez que vous avez accepté les conditions du modèle Phonon sur sa page\n"
                "Hugging Face et que « jeton_huggingface » est bien rempli dans config.yaml.\n"
                "Détail : %s",
                texte,
            )
        else:
            log.exception("Impossible de charger la voix")
        return 1

    import uvicorn

    from .application import Agent, creer_app

    agent = Agent(reglages, serveur.cerveau, voix, DetecteurSilero)
    agent.reparer = serveur.reparer
    n = len(agent.coffre.lister())
    log.info("%s note(s) trouvée(s) dans %s (lecture seule).", n, reglages.dossier_notes)
    log.info("Les fiches iront dans %s.", reglages.dossier_recaps)

    log.info("Étape 3/3 : le directeur lit vos consignes, notes et fiches "
             "(une à deux minutes la première fois)…")
    try:
        agent.prechauffer()
    except Exception as exc:  # noqa: BLE001
        log.warning("Lecture à l'avance impossible (%s) : le premier tour sera plus lent.", exc)

    app = creer_app(agent)
    adresse = f"http://127.0.0.1:{reglages.port}"

    def ouvrir_navigateur():
        # On attend que la page réponde vraiment avant d'ouvrir le navigateur.
        for _ in range(120):
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
