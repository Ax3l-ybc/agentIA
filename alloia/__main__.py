"""AlloIA : `python -m alloia` (lancé par jetson/lancer.sh).

Boucle : veille → quelqu'un parle dans le combiné → appel → documentation → veille.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time

log = logging.getLogger("alloia")


def configurer_journal() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for bavard in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub",
                   "uvicorn.access", "faster_whisper"):
        logging.getLogger(bavard).setLevel(logging.WARNING)


def ecouter_clavier(boucle: asyncio.AbstractEventLoop, evenement: asyncio.Event) -> None:
    """Entrée au clavier = décrocher / raccrocher (pratique pour les essais)."""
    if not sys.stdin or not sys.stdin.isatty():
        return

    def lire():
        for _ in sys.stdin:
            boucle.call_soon_threadsafe(evenement.set)

    threading.Thread(target=lire, daemon=True).start()


async def demarrer_serveur_voix(port: int):
    import fastapi
    import uvicorn

    from agent import serveur_voix

    app = fastapi.FastAPI()
    app.include_router(serveur_voix.routeur)
    serveur = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", lifespan="off",
            ws_ping_interval=None, ws_ping_timeout=None,
        )
    )
    tache = asyncio.create_task(serveur.serve())
    while not serveur.started:
        if tache.done():
            tache.result()
        await asyncio.sleep(0.05)
    return serveur, tache


def charger_moteurs(r) -> None:
    from agent import moteurs, serveur_voix

    log.info("Chargement de l'écoute (Whisper « %s »)…", r.whisper_modele)
    serveur_voix.MOTEURS["ecoute"] = moteurs.MoteurEcouteWhisper(
        r.whisper_modele,
        invite="FabLab, découpe laser, imprimante 3D, fraiseuse numérique, contreplaqué, PLA, Arduino",
        pause_s=r.pause_s,
    )
    log.info("Chargement de la voix (Phonon)…")
    serveur_voix.MOTEURS["voix"] = moteurs.MoteurVoix(r.voix_modele, r.voix_nom, r.voix_vitesse)


async def boucle_principale(r, combine, modele_llm: str) -> None:
    from . import doc
    from .appel import Appel, attendre_decroche

    consignes = r.mode_de_conduite.read_text(encoding="utf-8-sig")
    modele_doc = r.modele_doc.read_text(encoding="utf-8-sig")
    clavier = asyncio.Event()
    ecouter_clavier(asyncio.get_running_loop(), clavier)
    redactions: set[asyncio.Task] = set()

    while True:
        log.info("=== En attente d'un appel : parlez dans le combiné (ou appuyez sur Entrée). ===")
        cause = await attendre_decroche(combine, r.parole_pour_decrocher_s, clavier)
        log.info("Décroché (%s) : début de l'appel.", cause)
        # Le mode de conduite est relu à chaque appel : une modification s'applique tout de suite.
        consignes = r.mode_de_conduite.read_text(encoding="utf-8-sig")
        try:
            conv = await Appel(r, modele_llm, consignes).derouler(combine, clavier)
        except Exception:  # noqa: BLE001
            log.exception("L'appel s'est mal terminé")
            await asyncio.sleep(2)
            continue
        modele_doc = r.modele_doc.read_text(encoding="utf-8-sig")
        tache = asyncio.create_task(
            asyncio.to_thread(
                doc.rediger, conv, r.ollama_adresse, modele_llm, r.ollama_contexte,
                modele_doc, r.dossier_docs,
            )
        )
        redactions.add(tache)
        tache.add_done_callback(redactions.discard)
        await asyncio.sleep(2)  # le temps de reposer le combiné


async def principal(args) -> int:
    from agent import ollama
    from agent.__main__ import premier_port_libre
    from agent.reglages import ErreurReglages

    from . import audio
    from .reglages import charger

    try:
        r = charger()
    except ErreurReglages as exc:
        log.error("PROBLÈME DE RÉGLAGES :\n%s", exc)
        return 1
    if r.jeton_hf:
        os.environ["HF_TOKEN"] = r.jeton_hf

    if args.test_audio:
        try:
            return await tester_audio(r)
        except RuntimeError as exc:
            log.error("PROBLÈME DE SON :\n%s", exc)
            return 1

    log.info("Étape 1/3 : le cerveau (Ollama, %s).", r.ollama_modele)
    try:
        modele = await asyncio.to_thread(
            ollama.preparer, r.ollama_adresse, r.ollama_modele, r.ollama_contexte,
            "alloia", r.ollama_temperature,
        )
        await asyncio.to_thread(ollama.precharger, r.ollama_adresse, modele)
    except ollama.ErreurOllama as exc:
        log.error("PROBLÈME AVEC OLLAMA :\n%s", exc)
        return 1
    log.info("Étape 2/3 : l'écoute et la voix.")
    try:
        await asyncio.to_thread(charger_moteurs, r)
    except Exception as exc:  # noqa: BLE001
        texte = str(exc)
        if any(m in texte for m in ("GatedRepo", "Authentication", "401", "403", "restricted")):
            log.error(
                "TÉLÉCHARGEMENT DE LA VOIX REFUSÉ PAR HUGGING FACE.\n"
                "%s\n"
                "Vérifiez : 1) que vous avez accepté les conditions sur "
                "https://huggingface.co/gradium/phonon-40m-fr (avec le même compte que le jeton) ;\n"
                "2) que le jeton est bien écrit dans alloia.yaml, ligne « jeton_huggingface: »,"
                " entre apostrophes.",
                "Aucun jeton n'est indiqué dans alloia.yaml." if not r.jeton_hf
                else "Un jeton est indiqué dans alloia.yaml, mais il n'a pas accès au modèle.",
            )
        else:
            log.exception("Impossible de charger l'écoute ou la voix")
        return 1
    r.port = premier_port_libre(r.port) or r.port
    serveur, _ = await demarrer_serveur_voix(r.port)
    log.info("Étape 3/3 : le combiné.")
    try:
        combine = audio.CombineUSB(r.audio_entree, r.audio_sortie, r.volume)
        await combine.demarrer()
    except RuntimeError as exc:
        log.error("PROBLÈME DE SON :\n%s", exc)
        serveur.should_exit = True
        return 1
    log.info("Les documentations seront écrites dans %s", r.dossier_docs)
    try:
        await boucle_principale(r, combine, modele)
    finally:
        await combine.arreter()
        serveur.should_exit = True
    return 0


async def tester_audio(r) -> int:
    """Joue un bip, enregistre 4 secondes et indique le niveau du micro."""
    import numpy as np

    from . import audio

    combine = audio.CombineUSB(r.audio_entree, r.audio_sortie, r.volume)
    await combine.demarrer()
    t = np.arange(int(0.5 * 48000)) / 48000
    bip = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    combine.jouer((bip * 32767).astype("<i2").tobytes())
    log.info("Vous devez entendre un bip. Parlez maintenant dans le micro pendant 4 secondes…")
    await asyncio.sleep(0.8)
    combine.vider_micro()
    niveaux, fin = [], time.monotonic() + 4
    while time.monotonic() < fin:
        trame = await combine.micro.get()
        niveaux.append(float(np.sqrt(np.mean(trame**2))))
    await combine.arreter()
    maxi = max(niveaux) if niveaux else 0.0
    log.info("Niveau du micro : moyen %.3f, maximum %.3f", float(np.mean(niveaux or [0])), maxi)
    if maxi < 0.01:
        log.warning("Le micro semble muet : vérifiez le branchement et le volume d'entrée (alsamixer).")
    else:
        log.info("Le micro fonctionne.")
    return 0


def main() -> int:
    configurer_journal()
    parser = argparse.ArgumentParser(prog="alloia")
    parser.add_argument("--lister-audio", action="store_true", help="liste les cartes son")
    parser.add_argument("--test-audio", action="store_true", help="teste le micro et le haut-parleur")
    args = parser.parse_args()
    if args.lister_audio:
        from .audio import lister_peripheriques

        print(lister_peripheriques())
        return 0
    try:
        return asyncio.run(principal(args))
    except KeyboardInterrupt:
        log.info("Arrêt d'AlloIA.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
