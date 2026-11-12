"""AlloIA (version « omni ») : `python -m alloia` (lancé par jetson/lancer.sh).

Boucle : veille → quelqu'un parle dans le combiné → appel → documentation → veille.
Le cerveau (Gemma 4 par défaut) écoute directement la voix ; Phonon ou Ollama parle.
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

# Versions minimales d'Ollama pour les modèles qui écoutent l'audio.
VERSION_MINI = {"gemma4": (0, 30, 9), "lfm3": (0, 40, 13)}


def configurer_journal() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for bavard in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub", "faster_whisper"):
        logging.getLogger(bavard).setLevel(logging.WARNING)


def ecouter_clavier(boucle: asyncio.AbstractEventLoop, evenement: asyncio.Event) -> None:
    """Entrée au clavier = décrocher / raccrocher (pratique pour les essais)."""
    if not sys.stdin or not sys.stdin.isatty():
        return

    def lire():
        for _ in sys.stdin:
            boucle.call_soon_threadsafe(evenement.set)

    threading.Thread(target=lire, daemon=True).start()


def preparer_cerveau(r) -> None:
    """Vérifie Ollama, sa version et le modèle ; le télécharge et le charge en mémoire."""
    from agent import ollama

    from . import ollama_audio

    ollama.verifier(r.ollama_adresse)
    version = ollama_audio.version(r.ollama_adresse)
    famille = r.ollama_modele.split(":")[0].rstrip("0123456789.-")
    mini = next((v for k, v in VERSION_MINI.items() if r.ollama_modele.startswith(k)), None)
    if mini and version < mini:
        raise ollama.ErreurOllama(
            f"Ollama {'.'.join(map(str, version))} est trop ancien pour {r.ollama_modele} "
            f"(il faut {'.'.join(map(str, mini))} ou plus récent).\n"
            "Mettez-le à jour avec : curl -fsSL https://ollama.com/install.sh | sh"
        )
    ollama.telecharger(r.ollama_adresse, r.ollama_modele)
    caps = ollama_audio.capacites(r.ollama_adresse, r.ollama_modele)
    if "audio" not in caps:
        raise ollama.ErreurOllama(
            f"Le modèle {r.ollama_modele} ne sait pas écouter l'audio ({famille}). "
            "Choisissez gemma4:e2b ou lfm3-omni dans alloia.yaml."
        )
    log.info("Chargement du cerveau en mémoire…")
    ollama.precharger(r.ollama_adresse, r.ollama_modele, r.ollama_contexte)


def preparer_voix(r):
    from . import ollama_audio, voix

    if r.voix_moteur == "ollama":
        from agent import ollama

        ollama.telecharger(r.ollama_adresse, r.voix_ollama_modele)
        if "speech" not in ollama_audio.capacites(r.ollama_adresse, r.voix_ollama_modele):
            raise ollama.ErreurOllama(f"Le modèle {r.voix_ollama_modele} ne sait pas parler.")
        log.info("Voix : %s (%s), servie par Ollama.", r.voix_ollama_modele, r.voix_ollama_nom)
        return voix.VoixOllama(r.ollama_adresse, r.voix_ollama_modele, r.voix_ollama_nom)
    return voix.VoixPhonon(r.voix_modele, r.voix_nom, r.voix_vitesse)


async def boucle_principale(r, combine, voix_agent, detecteur) -> None:
    from . import doc
    from .appel import Appel, attendre_decroche

    clavier = asyncio.Event()
    ecouter_clavier(asyncio.get_running_loop(), clavier)
    redactions: set[asyncio.Task] = set()

    while True:
        log.info("=== En attente d'un appel : parlez dans le combiné (ou appuyez sur Entrée). ===")
        cause = await attendre_decroche(combine, detecteur, r.parole_pour_decrocher_s, clavier)
        log.info("Décroché (%s) : début de l'appel.", cause)
        # Le mode de conduite est relu à chaque appel : une modification s'applique tout de suite.
        consignes = r.mode_de_conduite.read_text(encoding="utf-8-sig")
        try:
            conv = await Appel(r, consignes, voix_agent, detecteur).derouler(combine, clavier)
        except Exception:  # noqa: BLE001
            log.exception("L'appel s'est mal terminé")
            await asyncio.sleep(2)
            continue
        modele_doc = r.modele_doc.read_text(encoding="utf-8-sig")
        tache = asyncio.create_task(
            asyncio.to_thread(
                doc.rediger, conv, r.ollama_adresse, r.ollama_modele, r.ollama_contexte,
                modele_doc, r.dossier_docs, r.ollama_reflexion,
            )
        )
        redactions.add(tache)
        tache.add_done_callback(redactions.discard)
        await asyncio.sleep(2)  # le temps de reposer le combiné


async def principal(args) -> int:
    from agent import ollama
    from agent.reglages import ErreurReglages

    from . import audio
    from .phrases import detecteur_silero
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

    log.info("Étape 1/3 : le cerveau qui écoute (Ollama, %s).", r.ollama_modele)
    try:
        await asyncio.to_thread(preparer_cerveau, r)
    except ollama.ErreurOllama as exc:
        log.error("PROBLÈME AVEC OLLAMA :\n%s", exc)
        return 1

    log.info("Étape 2/3 : la voix et le détecteur de parole.")
    try:
        voix_agent = await asyncio.to_thread(preparer_voix, r)
        detecteur = detecteur_silero()
    except ollama.ErreurOllama as exc:
        log.error("PROBLÈME AVEC LA VOIX :\n%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        texte = str(exc)
        if any(m in texte for m in ("GatedRepo", "Authentication", "401", "403", "restricted")):
            log.error(
                "TÉLÉCHARGEMENT DE LA VOIX REFUSÉ PAR HUGGING FACE.\n%s\n"
                "Vérifiez : 1) que vous avez accepté les conditions sur "
                "https://huggingface.co/gradium/phonon-40m-fr (avec le même compte que le jeton) ;\n"
                "2) le jeton dans alloia.perso.yaml (ou alloia.yaml), ligne « jeton_huggingface: ».",
                "Aucun jeton n'est indiqué." if not r.jeton_hf
                else "Un jeton est indiqué, mais il n'a pas accès au modèle.",
            )
        else:
            log.exception("Impossible de charger la voix")
        return 1

    log.info("Étape 3/3 : le combiné.")
    try:
        combine = audio.CombineUSB(r.audio_entree, r.audio_sortie, r.volume)
        await combine.demarrer()
    except RuntimeError as exc:
        log.error("PROBLÈME DE SON :\n%s", exc)
        return 1
    log.info("Les documentations seront écrites dans %s", r.dossier_docs)
    try:
        await boucle_principale(r, combine, voix_agent, detecteur)
    finally:
        await combine.arreter()
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
