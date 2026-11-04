"""Serveur vocal local qui parle le protocole WebSocket de Gradium.

Gradbot est conçu pour le service en ligne de Gradium (écoute + voix). Ici, on
lui fournit à la place un petit serveur local, à la même adresse logique
(`<base>/speech/asr` et `<base>/speech/tts`), qui répond avec :
- Kyutai STT pour la transcription en streaming et la détection de fin de tour ;
- Phonon (phonon-40m-fr) pour la synthèse vocale.
Rien ne sort de l'ordinateur.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import uuid

import fastapi
import numpy as np

from . import moteurs

log = logging.getLogger("agent.voix")

routeur = fastapi.APIRouter()

# Les moteurs sont branchés au démarrage par l'application.
MOTEURS: dict = {"ecoute": None, "voix": None}

# Une seule session d'écoute à la fois (le modèle garde un état interne).
_generation_ecoute = 0


# ---------------------------------------------------------------------------
# Écoute (speech/asr)
# ---------------------------------------------------------------------------


def _probas_vers_vad(probas: list[float] | None, horizons=(0.5, 1.0, 2.0, 3.0)) -> list[dict]:
    if not probas:
        probas = [1.0] * len(horizons)
    while len(probas) < 3:  # Gradbot lit la 3e valeur
        probas = probas + [probas[-1]]
    return [
        {"horizon_s": horizons[min(i, len(horizons) - 1)], "inactivity_prob": p}
        for i, p in enumerate(probas)
    ]


class _AssembleurMots:
    """Regroupe les morceaux de mots de Kyutai en mots entiers."""

    def __init__(self, delai_s: float):
        self.delai_s = delai_s
        self.mot = ""
        self.debut_s: float | None = None  # moment où le mot en cours a commencé
        self.vides = 0

    def pousser(self, jeton, morceau, temps_s: float) -> list[dict]:
        sortie = []
        if jeton == moteurs.JETON_DEBUT_MOT:
            sortie += self.terminer()
            self.debut_s = temps_s
            self.vides = 0
        elif morceau is not None:
            if morceau.startswith("▁") and self.mot:
                sortie += self.terminer()
            if self.debut_s is None:
                self.debut_s = temps_s
            self.mot += morceau.replace("▁", " ")
            self.vides = 0
        else:
            self.vides += 1
            if self.vides >= 2:
                sortie += self.terminer()
        return sortie

    def terminer(self) -> list[dict]:
        mot = self.mot.strip()
        debut = self.debut_s or 0.0
        self.mot = ""
        self.debut_s = None
        if not mot:
            return []
        return [
            {
                "type": "text",
                "text": mot,
                "start_s": max(0.0, debut - self.delai_s),
                "stream_id": 0,
            }
        ]


@routeur.websocket("/api/speech/asr")
async def ecoute_ws(ws: fastapi.WebSocket):
    global _generation_ecoute
    moteur = MOTEURS["ecoute"]
    await ws.accept()
    try:
        setup = json.loads(await ws.receive_text())
    except Exception:
        await ws.close()
        return
    if setup.get("type") != "setup":
        await ws.send_json({"type": "error", "code": 400, "message": "setup attendu"})
        await ws.close()
        return

    _generation_ecoute += 1
    ma_generation = _generation_ecoute
    await asyncio.to_thread(moteur.reinitialiser)
    await ws.send_json(
        {
            "type": "ready",
            "model_name": "kyutai-stt-local",
            "sample_rate": moteurs.TAUX_ENTREE,
            "frame_size": moteurs.TRAME_ENTREE,
            "delay_in_frames": moteur.delai_s / 0.08,
            "text_stream_names": ["text"],
            "request_id": uuid.uuid4().hex,
        }
    )

    tampon = np.zeros(0, dtype=np.float32)
    assembleur = _AssembleurMots(moteur.delai_s)
    pas_idx = 0

    async def traiter(trame: np.ndarray):
        nonlocal pas_idx
        jeton, morceau, probas = await asyncio.to_thread(moteur.pas, trame)
        pas_idx += 1
        temps = pas_idx * 0.08
        messages = assembleur.pousser(jeton, morceau, temps) if jeton is not None else []
        if probas and probas[min(2, len(probas) - 1)] > 0.5:
            messages += assembleur.terminer()  # fin de tour probable : on livre le mot
        for m in messages:
            await ws.send_json(m)
        await ws.send_json(
            {
                "type": "step",
                "step_idx": pas_idx,
                "step_duration_s": 0.08,
                "total_duration_s": temps,
                "vad": _probas_vers_vad(probas),
            }
        )

    try:
        while ma_generation == _generation_ecoute:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is None:
                continue
            donnees = json.loads(message["text"])
            if donnees.get("type") == "audio":
                pcm = moteurs.pcm16_vers_float(base64.b64decode(donnees["audio"]))
                tampon = np.concatenate([tampon, pcm])
                while len(tampon) >= moteurs.TRAME_ENTREE and ma_generation == _generation_ecoute:
                    trame, tampon = tampon[: moteurs.TRAME_ENTREE], tampon[moteurs.TRAME_ENTREE :]
                    await traiter(trame)
            elif donnees.get("type") == "end_of_stream":
                for m in assembleur.terminer():
                    await ws.send_json(m)
                await ws.send_json({"type": "end_of_stream"})
                break
    except fastapi.WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Erreur dans la session d'écoute")
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Voix (speech/tts)
# ---------------------------------------------------------------------------

_FIN_PHRASE = re.compile(r"([.!?…;:])(\s+)")
_A_NETTOYER = re.compile(r"[*_#`>|~]|\[\[|\]\]")


def nettoyer_pour_voix(texte: str) -> str:
    texte = _A_NETTOYER.sub(" ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def couper_phrases(tampon: str, premiere: bool) -> tuple[list[str], str]:
    """Sépare les phrases complètes du reste du texte en attente."""
    phrases = []
    while True:
        m = _FIN_PHRASE.search(tampon)
        if not m:
            break
        phrases.append(tampon[: m.end(1)])
        tampon = tampon[m.end() :]
    # Phrase très longue sans ponctuation forte : on coupe à une virgule.
    seuil = 60 if premiere and not phrases else 220
    if len(tampon) > seuil and "," in tampon[20:]:
        i = tampon.rindex(",")
        phrases.append(tampon[: i + 1])
        tampon = tampon[i + 1 :].lstrip()
    return phrases, tampon


@routeur.websocket("/api/speech/tts")
async def voix_ws(ws: fastapi.WebSocket):
    moteur = MOTEURS["voix"]
    await ws.accept()
    try:
        setup = json.loads(await ws.receive_text())
    except Exception:
        await ws.close()
        return
    await ws.send_json(
        {
            "type": "ready",
            "model_name": "phonon-local",
            "sample_rate": moteurs.TAUX_SORTIE,
            "frame_size": moteurs.TRAME_SORTIE,
            "audio_stream_names": ["audio"],
            "text_stream_names": ["text"],
            "request_id": uuid.uuid4().hex,
            **({"client_req_id": setup["client_req_id"]} if setup.get("client_req_id") else {}),
        }
    )

    file_phrases: asyncio.Queue = asyncio.Queue()
    arret = threading.Event()
    boucle = asyncio.get_running_loop()

    async def recevoir():
        tampon, premiere = "", True
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    arret.set()
                    break
                if message.get("text") is None:
                    continue
                donnees = json.loads(message["text"])
                if donnees.get("type") == "text":
                    tampon += donnees.get("text", "")
                    phrases, tampon = couper_phrases(tampon, premiere)
                    for p in phrases:
                        premiere = False
                        await file_phrases.put(p)
                elif donnees.get("type") == "end_of_stream":
                    if tampon.strip():
                        await file_phrases.put(tampon)
                    break
        except (fastapi.WebSocketDisconnect, RuntimeError):
            arret.set()
        finally:
            await file_phrases.put(None)

    async def parler():
        temps = 0.0
        reech = moteurs.Reechantillonneur(moteur.taux, moteurs.TAUX_SORTIE)
        reste = np.zeros(0, dtype=np.float32)
        while not arret.is_set():
            phrase = await file_phrases.get()
            if phrase is None:
                break
            texte = nettoyer_pour_voix(phrase)
            if not re.search(r"\w", texte):
                continue
            debut_phrase = temps
            morceaux: asyncio.Queue = asyncio.Queue()

            def produire(texte=texte):
                try:
                    for m in moteur.parler(texte, arret):
                        boucle.call_soon_threadsafe(morceaux.put_nowait, m)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Erreur de synthèse vocale")
                    boucle.call_soon_threadsafe(morceaux.put_nowait, exc)
                finally:
                    boucle.call_soon_threadsafe(morceaux.put_nowait, None)

            fil = threading.Thread(target=produire, daemon=True)
            fil.start()
            while True:
                m = await morceaux.get()
                if m is None or isinstance(m, Exception):
                    break
                reste = np.concatenate([reste, reech(m)])
                while len(reste) >= moteurs.TRAME_SORTIE:
                    trame, reste = reste[: moteurs.TRAME_SORTIE], reste[moteurs.TRAME_SORTIE :]
                    temps = await _envoyer_audio(ws, trame, temps)
            if len(reste):  # fin de phrase : on complète la dernière trame
                trame = np.concatenate(
                    [reste, np.zeros(moteurs.TRAME_SORTIE - len(reste), dtype=np.float32)]
                )
                reste = np.zeros(0, dtype=np.float32)
                temps = await _envoyer_audio(ws, trame, temps)
            await _envoyer_mots(ws, texte, debut_phrase, temps)
        if not arret.is_set():
            await ws.send_json({"type": "end_of_stream"})

    try:
        await asyncio.gather(recevoir(), parler())
    except Exception:
        arret.set()
        log.debug("Session de voix interrompue", exc_info=True)
    finally:
        arret.set()
        try:
            await ws.close()
        except Exception:
            pass


async def _envoyer_audio(ws: fastapi.WebSocket, trame: np.ndarray, temps: float) -> float:
    fin = temps + len(trame) / moteurs.TAUX_SORTIE
    await ws.send_json(
        {
            "type": "audio",
            "audio": base64.b64encode(moteurs.float_vers_pcm16(trame)).decode("ascii"),
            "start_s": temps,
            "stop_s": fin,
            "stream_id": 0,
        }
    )
    return fin


async def _envoyer_mots(ws: fastapi.WebSocket, texte: str, debut: float, fin: float) -> None:
    """Envoie les mots prononcés avec un horodatage réparti sur la phrase."""
    mots = texte.split()
    total = sum(len(m) + 1 for m in mots) or 1
    t = debut
    for mot in mots:
        duree = (fin - debut) * (len(mot) + 1) / total
        await ws.send_json(
            {"type": "text", "text": mot, "start_s": t, "stop_s": t + duree, "stream_id": 0}
        )
        t += duree
