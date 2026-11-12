"""Dialogue avec Ollama pour AlloIA-omni : un modèle qui écoute directement l'audio.

- `discuter_en_flux` : conversation (avec un message vocal en WAV), réponse en flux ;
- `transcrire` : retranscription d'un message vocal (pour l'historique et la doc) ;
- `parler_en_flux` : synthèse vocale par Ollama (modèles « speech », ex. lfm3-omni).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
import uuid
import wave
from typing import Iterator

import numpy as np

from agent import ollama as ollama_base

log = logging.getLogger("alloia.ollama")

ErreurOllama = ollama_base.ErreurOllama


def wav_16k(audio: np.ndarray, taux: int = 16000) -> bytes:
    """Encode un signal float32 mono en WAV 16 bits."""
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(taux)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return tampon.getvalue()


def _post(adresse: str, chemin: str, corps: bytes, type_contenu: str, delai: float = 600):
    req = urllib.request.Request(adresse + chemin, data=corps, headers={"Content-Type": type_contenu})
    try:
        return urllib.request.urlopen(req, timeout=delai)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ErreurOllama(f"Ollama a refusé la requête {chemin} ({exc.code}) : {detail}") from exc


def version(adresse: str) -> tuple[int, ...]:
    with urllib.request.urlopen(adresse + "/api/version", timeout=5) as r:
        texte = json.load(r).get("version", "0")
    return tuple(int(x) for x in texte.split("-")[0].split(".") if x.isdigit())


def capacites(adresse: str, modele: str) -> set[str]:
    corps = json.dumps({"model": modele}).encode()
    with _post(adresse, "/api/show", corps, "application/json", delai=30) as r:
        return set(json.load(r).get("capabilities") or [])


def discuter_en_flux(
    adresse: str,
    modele: str,
    messages: list[dict],
    outils: list[dict] | None = None,
    contexte: int = 8192,
    temperature: float = 0.4,
    reflexion: bool | None = False,
) -> Iterator[dict]:
    """Envoie la conversation et produit les morceaux de réponse.

    Chaque élément est {"texte": "..."} ou {"outil": nom, "arguments": {...}}.
    """
    corps = {
        "model": modele,
        "messages": messages,
        "stream": True,
        "keep_alive": -1,
        "options": {"num_ctx": contexte, "temperature": temperature},
    }
    if reflexion is not None:
        corps["think"] = reflexion
    if outils:
        corps["tools"] = outils
    with _post(adresse, "/api/chat", json.dumps(corps).encode(), "application/json") as r:
        for ligne in r:
            if not ligne.strip():
                continue
            morceau = json.loads(ligne)
            if "error" in morceau:
                raise ErreurOllama(morceau["error"])
            message = morceau.get("message") or {}
            if message.get("content"):
                yield {"texte": message["content"]}
            for appel in message.get("tool_calls") or []:
                fonction = appel.get("function") or {}
                yield {"outil": fonction.get("name", ""), "arguments": fonction.get("arguments") or {}}
            if morceau.get("done"):
                break


def message_vocal(wav: bytes, consigne: str) -> dict:
    """Message « utilisateur » contenant un enregistrement (Ollama le reconnaît comme audio)."""
    return {"role": "user", "content": consigne, "images": [base64.b64encode(wav).decode("ascii")]}


def transcrire(adresse: str, modele: str, wav: bytes, langue: str = "fr") -> str:
    """Retranscrit un message vocal avec le même modèle (/v1/audio/transcriptions)."""
    frontiere = uuid.uuid4().hex
    parties = []
    for nom, valeur in (("model", modele), ("language", langue), ("response_format", "json")):
        parties.append(
            f'--{frontiere}\r\nContent-Disposition: form-data; name="{nom}"\r\n\r\n{valeur}\r\n'.encode()
        )
    parties.append(
        f'--{frontiere}\r\nContent-Disposition: form-data; name="file"; filename="maker.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n"
    )
    parties.append(f"--{frontiere}--\r\n".encode())
    with _post(
        adresse, "/v1/audio/transcriptions", b"".join(parties),
        f"multipart/form-data; boundary={frontiere}", delai=300,
    ) as r:
        return (json.load(r).get("text") or "").strip()


def parler_en_flux(adresse: str, modele: str, texte: str, voix: str = "") -> Iterator[bytes]:
    """Synthèse vocale par Ollama : PCM 16 bits mono 24 kHz, en flux."""
    corps = {"model": modele, "input": texte, "response_format": "pcm"}
    if voix:
        corps["voice"] = voix
    with _post(adresse, "/v1/audio/speech", json.dumps(corps).encode(), "application/json") as r:
        while True:
            bloc = r.read(4800)
            if not bloc:
                break
            yield bloc
