"""Bouton du socle : le combiné posé appuie dessus ; le soulever lance l'appel, le reposer l'arrête.

Branchement conseillé sur le connecteur 40 broches de la Jetson (numérotation de la carte) :

    broche 17 (3,3 V) ── résistance 10 kΩ ──┬── broche 15 (entrée)
                                            │
                                  bouton (contacts C et NO)
                                            │
    broche 14 (masse) ──────────────────────┘

Combiné posé : bouton fermé, la broche lit 0. Combiné soulevé : la broche lit 1.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import warnings
from typing import Callable

log = logging.getLogger("alloia")


class ErreurBouton(RuntimeError):
    pass


PAS_LE_DROIT = (
    "Pas le droit de lire les broches de la carte. Relancez ./jetson/installer.sh, puis "
    "redémarrez la Jetson (l'autorisation ne compte qu'après un redémarrage)."
)


def ouvrir_broche(broche: int) -> Callable[[], int]:
    """Prépare une broche du connecteur 40 broches en entrée ; renvoie une fonction qui la lit."""
    try:
        import Jetson.GPIO as GPIO  # noqa: N814
    except ImportError as exc:
        raise ErreurBouton(
            "La bibliothèque Jetson.GPIO manque : relancez ./jetson/installer.sh"
        ) from exc
    except Exception as exc:  # noqa: BLE001  (pas une Jetson, pas les droits…)
        if "permission" in str(exc).lower():
            raise ErreurBouton(PAS_LE_DROIT) from exc
        raise ErreurBouton(f"Impossible d'utiliser les broches de la carte : {exc}") from exc
    try:
        GPIO.setmode(GPIO.BOARD)
        with warnings.catch_warnings(record=True) as avertissements:
            warnings.simplefilter("always")
            GPIO.setup(broche, GPIO.IN)
        # Lancé avec sudo, Jetson.GPIO vérifie que la broche est bien réglée en entrée
        # (« pinmux ») et donne la commande qui corrige : on l'affiche.
        for a in avertissements:
            if "pinmux" in str(a.message) and "/dev/mem" not in str(a.message):
                log.warning("Réglage de la broche %d sur la carte :%s", broche, a.message)
        GPIO.input(broche)
    except PermissionError as exc:
        raise ErreurBouton(PAS_LE_DROIT) from exc
    except Exception as exc:  # noqa: BLE001
        raise ErreurBouton(f"La broche {broche} n'est pas utilisable : {exc}") from exc
    return lambda: int(GPIO.input(broche))


class BoutonSocle:
    """Surveille le bouton dans un fil à part et prévient la boucle asyncio des changements.

    Anti-rebond : un nouvel état n'est retenu que s'il dure `anti_rebond_s`, pour qu'un
    combiné reposé un peu brusquement ne raccroche et redécroche pas.
    """

    def __init__(self, lire: Callable[[], int], valeur_pose: int = 0,
                 anti_rebond_s: float = 0.15, periode_s: float = 0.01):
        self.lire = lire
        self.valeur_pose = int(valeur_pose)
        self.anti_rebond_s = anti_rebond_s
        self.periode_s = periode_s
        self.decroche = False
        self.evt_decroche = asyncio.Event()
        self.evt_pose = asyncio.Event()
        self._arret = threading.Event()
        self._fil: threading.Thread | None = None

    def _lire_decroche(self) -> bool | None:
        try:
            return int(self.lire()) != self.valeur_pose
        except Exception as exc:  # noqa: BLE001
            log.warning("Lecture du bouton impossible : %s", exc)
            return None

    def _changer(self, decroche: bool) -> None:
        self.decroche = decroche
        if decroche:
            self.evt_pose.clear()
            self.evt_decroche.set()
        else:
            self.evt_decroche.clear()
            self.evt_pose.set()

    def demarrer(self) -> None:
        boucle = asyncio.get_running_loop()
        etat = bool(self._lire_decroche())
        self._changer(etat)

        def surveiller():
            nonlocal etat
            candidat, depuis = etat, time.monotonic()
            while not self._arret.is_set():
                v = self._lire_decroche()
                maintenant = time.monotonic()
                if v is None:
                    time.sleep(1)
                    continue
                if v != candidat:
                    candidat, depuis = v, maintenant
                elif candidat != etat and maintenant - depuis >= self.anti_rebond_s:
                    etat = candidat
                    boucle.call_soon_threadsafe(self._changer, etat)
                time.sleep(self.periode_s)

        self._fil = threading.Thread(target=surveiller, daemon=True)
        self._fil.start()

    def arreter(self) -> None:
        self._arret.set()


async def attendre_decroche_bouton(bouton: BoutonSocle, evenement_clavier: asyncio.Event) -> str:
    """Attend qu'on soulève le combiné (ou qu'on appuie sur Entrée). Renvoie la cause."""
    while True:
        if bouton.decroche:
            return "bouton"
        if evenement_clavier.is_set():
            evenement_clavier.clear()
            return "clavier"
        try:
            await asyncio.wait_for(bouton.evt_decroche.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass


def tester(broche: int, valeur_pose: int, duree_s: float = 120) -> int:
    """Affiche l'état du bouton à chaque changement (./jetson/lancer.sh --test-bouton)."""
    try:
        lire = ouvrir_broche(broche)
    except ErreurBouton as exc:
        log.error("PROBLÈME DE BOUTON :\n%s", exc)
        return 1

    def decrire(v: int) -> str:
        return "combiné POSÉ (en veille)" if v == valeur_pose else "combiné DÉCROCHÉ (appel)"

    log.info("Test du bouton sur la broche %d pendant %d s : posez et soulevez le combiné "
             "(Ctrl+C pour arrêter).", broche, duree_s)
    precedent = lire()
    log.info("  broche %d = %d  ->  %s", broche, precedent, decrire(precedent))
    changements, fin = 0, time.monotonic() + duree_s
    try:
        while time.monotonic() < fin:
            v = lire()
            if v != precedent:
                changements += 1
                log.info("  broche %d = %d  ->  %s", broche, v, decrire(v))
                precedent = v
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    if changements == 0:
        log.warning(
            "La broche n'a jamais changé. Vérifiez les fils (broches 17, 15 et 14) et la résistance.\n"
            "Si tout est bon, la broche n'est peut-être pas réglée en entrée sur la carte : relancez "
            "le test avec sudo pour le savoir :\n"
            "    sudo .venv/bin/python -m alloia --test-bouton"
        )
    else:
        log.info("Le bouton fonctionne. Si « posé » et « décroché » sont inversés, mettez "
                 "valeur_pose: %d dans la partie « bouton » de alloia.perso.yaml.", 1 - valeur_pose)
    return 0
