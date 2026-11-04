"""Point d'entrée : `python -m agent` (lancé par lancer.bat)."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import webbrowser


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

    if reglages.jeton_hf:
        os.environ["HF_TOKEN"] = reglages.jeton_hf

    from . import moteurs, ollama

    # 1. Le cerveau (Ollama)
    try:
        modele = ollama.preparer(
            reglages.ollama_adresse, reglages.ollama_modele, reglages.ollama_contexte
        )
    except ollama.ErreurOllama as exc:
        log.error("PROBLÈME AVEC OLLAMA :\n%s", exc)
        return 1
    log.info("Cerveau prêt : %s", modele)

    # 2. L'écoute et la voix (téléchargées au premier lancement)
    try:
        ecoute = moteurs.MoteurEcoute(reglages.stt_modele, reglages.materiel)
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
