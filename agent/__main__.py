"""Point d'entrée : `python -m agent` (lancé par lancer.bat)."""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for bavard in ("httpx", "huggingface_hub", "uvicorn.access"):
        logging.getLogger(bavard).setLevel(logging.WARNING)
    log = logging.getLogger("agent")

    from .reglages import ErreurReglages, charger

    try:
        reglages = charger()
    except ErreurReglages as exc:
        log.error("PROBLÈME DE RÉGLAGES :\n%s", exc)
        return 1

    log.info("Réglages lus. Notes : %s", reglages.dossier_notes)
    port = premier_port_libre(reglages.port)
    if port is None:
        log.error(
            "PROBLÈME : les ports %s à %s sont tous occupés. Fermez les autres fenêtres "
            "« Directeur de mémoire » ou redémarrez l'ordinateur.",
            reglages.port,
            reglages.port + 19,
        )
        return 1
    if port != reglages.port:
        log.info(
            "Le port %s est déjà pris (l'agent est peut-être déjà ouvert dans une autre "
            "fenêtre) : utilisation du port %s.",
            reglages.port,
            port,
        )
        reglages.port = port
    if reglages.jeton_hf:
        os.environ["HF_TOKEN"] = reglages.jeton_hf
    else:
        log.warning("Aucun jeton Hugging Face dans config.yaml : la voix ne pourra pas être téléchargée.")

    from . import moteurs, ollama

    log.info("Étape 1/3 : le cerveau (Ollama). Au premier lancement, environ 5 Go à télécharger.")
    try:
        modele = ollama.preparer(
            reglages.ollama_adresse, reglages.ollama_modele, reglages.ollama_contexte
        )
    except ollama.ErreurOllama as exc:
        log.error("PROBLÈME AVEC OLLAMA :\n%s", exc)
        return 1
    log.info("Cerveau prêt : %s", modele)

    try:
        log.info(
            "Étape 2/3 : l'écoute (Kyutai). Au premier lancement, environ 2,4 Go à télécharger ; "
            "le chargement prend ensuite une à deux minutes."
        )
        ecoute = moteurs.MoteurEcoute(reglages.stt_modele, reglages.materiel)
        log.info("Étape 3/3 : la voix (Phonon). Environ 120 Mo au premier lancement.")
        voix = moteurs.MoteurVoix(
            reglages.voix_modele, reglages.voix_nom, reglages.voix_vitesse, reglages.materiel
        )
    except Exception as exc:  # noqa: BLE001
        texte = str(exc)
        if any(m in texte for m in ("401", "403", "gated", "Gated", "token", "Unauthorized")):
            log.error(
                "TÉLÉCHARGEMENT REFUSÉ PAR HUGGING FACE.\n"
                "Vérifiez que vous avez accepté les conditions du modèle Phonon sur sa page\n"
                "Hugging Face et que « jeton_huggingface » est bien rempli dans config.yaml.\n"
                "Détail : %s",
                texte,
            )
        else:
            log.exception("Impossible de charger l'écoute ou la voix")
        return 1

    import uvicorn

    from .application import creer_app

    app = creer_app(reglages, moteur_ecoute=ecoute, moteur_voix=voix, modele_llm=modele)

    def ouvrir_navigateur():
        for _ in range(60):
            time.sleep(0.5)
            if app.state.pret:
                time.sleep(0.5)
                webbrowser.open(f"http://localhost:{reglages.port}")
                return

    threading.Thread(target=ouvrir_navigateur, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=reglages.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
