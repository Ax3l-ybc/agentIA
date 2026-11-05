"""Point d'entrée : `python -m agent` (lancé par lancer.bat)."""

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


def choisir_moteur_ecoute(reglages) -> str:
    """« auto » : Kyutai avec une carte NVIDIA, Whisper sur processeur seul."""
    if reglages.stt_moteur in ("kyutai", "whisper"):
        return reglages.stt_moteur
    if reglages.materiel == "cpu":
        return "whisper"
    import torch

    return "kyutai" if torch.cuda.is_available() else "whisper"


def charger_ecoute(reglages, moteurs, log):
    if choisir_moteur_ecoute(reglages) == "kyutai":
        log.info(
            "Étape 2/3 : l'écoute (Kyutai). Au premier lancement, environ 2,4 Go à télécharger ; "
            "le chargement prend ensuite une à deux minutes."
        )
        return moteurs.MoteurEcoute(reglages.stt_modele, reglages.materiel)
    log.info(
        "Étape 2/3 : l'écoute (Whisper « %s », car pas de carte NVIDIA). "
        "Environ 500 Mo au premier lancement.",
        reglages.whisper_modele,
    )
    # Les titres des notes et le sujet aident Whisper à reconnaître votre vocabulaire.
    titres = sorted({p.stem for p in reglages.dossier_notes.rglob("*.md")})[:40]
    invite = ". ".join(filter(None, [reglages.sujet_memoire, ", ".join(titres)]))
    return moteurs.MoteurEcouteWhisper(
        reglages.whisper_modele, invite=invite[:600], pause_s=reglages.whisper_pause_s
    )


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
        ecoute = charger_ecoute(reglages, moteurs, log)
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
            log.info("Si elle ne s'ouvre pas, copiez cette adresse dans Chrome ou Edge.")
            if not webbrowser.open(adresse):
                log.warning("Impossible d'ouvrir le navigateur tout seul : ouvrez %s", adresse)
            return
        log.error("La page ne répond pas. Essayez d'ouvrir %s dans votre navigateur.", adresse)

    threading.Thread(target=ouvrir_navigateur, daemon=True).start()
    # Tout est local : pas besoin des pings de contrôle des WebSockets, qui
    # coupaient la connexion quand le modèle prenait du retard.
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=reglages.port,
        log_level="warning",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
