"""Tests d'AlloIA (version « omni ») sans Jetson : combiné simulé, faux Ollama."""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import re

import fastapi
import numpy as np
import pytest
from fastapi.responses import StreamingResponse

from alloia import doc, ollama_audio
from alloia.appel import Appel, Conversation, attendre_decroche
from alloia.phrases import DecoupeurPhrases
from alloia.reglages import RACINE, Reglages, charger, fusionner
from alloia.voix import VoixSimulee
from tests.test_agent import lancer_serveur, port_libre

TRANSCRIPTIONS = [
    "J'ai fabriqué une lampe en contreplaqué de 3 mm à la découpe laser.",
    "Le plus dur a été de régler la puissance du laser. C'est tout, au revoir.",
]

DOC = """```markdown
---
date: 2026-11-12
type: documentation-projet
projet: Lampe en contreplaqué
---

# Lampe en contreplaqué

## Présentation
Une lampe.
```"""


def ton(secondes: float, taux: int = 24000) -> np.ndarray:
    t = np.arange(int(secondes * taux)) / taux
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def detecteur_energie(bloc: np.ndarray) -> float:
    return 0.9 if float(np.abs(bloc).mean()) > 0.01 else 0.1


def faux_ollama(journal: list):
    """Imite Ollama : /api/chat (flux NDJSON, audio dans « images »), transcription, doc."""
    app = fastapi.FastAPI()
    transcriptions = list(TRANSCRIPTIONS)

    def ndjson(morceaux):
        for m in morceaux:
            yield json.dumps({"message": {"role": "assistant", **m}, "done": False}) + "\n"
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}) + "\n"

    @app.post("/api/chat")
    async def chat(req: fastapi.Request):
        corps = await req.json()
        journal.append(corps)
        if not corps.get("stream"):  # rédaction de la documentation
            return {"message": {"role": "assistant", "content": DOC}, "done": True}
        dernier = corps["messages"][-1]
        historique = " ".join(str(m.get("content")) for m in corps["messages"])
        if "décrocher" in dernier["content"]:
            morceaux = [{"content": "Allô ! Ici AlloIA. "}, {"content": "Quel projet viens-tu de terminer ?"}]
        elif dernier.get("images") and "lampe" in historique:
            # Deuxième message vocal (au revoir) : remerciement puis outil « raccrocher ».
            morceaux = [{"content": "Merci beaucoup, à bientôt !"},
                        {"content": "", "tool_calls": [{"function": {"name": "raccrocher", "arguments": {}}}]}]
        else:
            morceaux = [{"content": "Génial ! Quel réglage de laser as-tu utilisé ?"}]
        return StreamingResponse(ndjson(morceaux), media_type="application/x-ndjson")

    @app.post("/v1/audio/transcriptions")
    async def transcrire(req: fastapi.Request):
        corps = (await req.body()).decode("latin-1")
        champ = lambda nom: re.search(rf'name="{nom}"\r\n\r\n([^\r]*)', corps).group(1)  # noqa: E731
        assert "RIFF" in corps  # le fichier WAV est bien joint
        journal.append({"transcription": champ("model"), "langue": champ("language")})
        return {"text": transcriptions.pop(0) if transcriptions else ""}

    return app


def test_reglages_et_fichier_perso(tmp_path):
    r = charger()
    assert r.ollama_modele == "gemma4:e2b" and r.ollama_reflexion is False
    assert r.voix_moteur == "phonon"
    base = tmp_path / "alloia.yaml"
    base.write_text("ollama:\n  modele: 'gemma4:e2b'\naudio:\n  entree: ''\n", encoding="utf-8")
    perso = tmp_path / "alloia.perso.yaml"
    perso.write_text("jeton_huggingface: 'hf_x'\naudio:\n  entree: 'KT USB'\n", encoding="utf-8")
    r = charger(base, perso)
    assert r.jeton_hf == "hf_x" and r.audio_entree == "KT USB" and r.ollama_modele == "gemma4:e2b"
    assert fusionner({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_decoupeur_phrases():
    d = DecoupeurPhrases(detecteur_energie, pause_s=0.8)
    silence = np.zeros(1920, np.float32)
    voix = ton(0.08)
    sorties = []
    for trame in [silence] * 10 + [voix] * 25 + [silence] * 15:
        sorties += d.ajouter(trame)
    assert len(sorties) == 1 and 2.5 <= len(sorties[0]) / 16000 <= 3.4
    # Un bruit très bref n'est pas une phrase.
    d.reinitialiser()
    assert sum((d.ajouter(t) for t in [voix] * 2 + [silence] * 15), []) == []


def test_wav_et_message_vocal():
    wav = ollama_audio.wav_16k(np.zeros(1600, np.float32))
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    m = ollama_audio.message_vocal(wav, "consigne")
    assert base64.b64decode(m["images"][0]) == wav and m["content"] == "consigne"


def test_nom_et_ecriture_doc(tmp_path):
    contenu = doc.nettoyer(DOC)
    assert contenu.startswith("---") and "```" not in contenu
    nom = doc.nom_fichier(contenu, datetime.datetime(2026, 11, 12, 14, 5))
    assert nom == "2026-11-12_14h05_lampe-en-contreplaque.md"
    p1, p2 = doc.ecrire(tmp_path, nom, "a"), doc.ecrire(tmp_path, nom, "b")
    assert p1 != p2 and p1.read_text() == "a"


def test_appel_trop_court_sans_doc(tmp_path):
    conv = Conversation(debut=datetime.datetime.now())
    conv.ajouter("alloia", "Allô !")
    assert doc.rediger(conv, "http://x", "m", 100, "", tmp_path) is None


def test_appel_complet(tmp_path):
    from alloia import audio

    journal: list = []
    port = port_libre()
    faux = lancer_serveur(faux_ollama(journal), port)
    r = Reglages(dossier_docs=tmp_path / "docs", ollama_adresse=f"http://127.0.0.1:{port}",
                 pause_s=0.8, fin_apres_silence_s=30)
    consignes = (RACINE / "alloia_conduite.md").read_text(encoding="utf-8")

    async def scenario():
        combine = audio.CombineSimule([(1.0, ton(1.0)), (3.0, ton(2.0)), (3.0, ton(2.0))])
        await combine.demarrer()
        clavier = asyncio.Event()
        try:
            cause = await asyncio.wait_for(attendre_decroche(combine, detecteur_energie, 0.6, clavier), 10)
            appel = Appel(r, consignes, VoixSimulee(), detecteur_energie)
            conv = await asyncio.wait_for(appel.derouler(combine, clavier), 60)
        finally:
            await combine.arreter()
        return cause, conv, combine

    try:
        cause, conv, combine = asyncio.run(scenario())
        chemin = doc.rediger(conv, r.ollama_adresse, "gemma4:e2b", 8192, "MODELE", r.dossier_docs, False)
    finally:
        faux.should_exit = True

    assert cause == "voix"
    assert conv.raison_fin == "l'agent a raccroché"
    texte = conv.transcription()
    assert texte.startswith("AlloIA : Allô ! Ici AlloIA.")
    assert "Maker : J'ai fabriqué une lampe en contreplaqué" in texte and "à bientôt" in texte
    assert float(np.abs(combine.audio_joue()).max()) > 0.05  # la voix est sortie

    flux = [c for c in journal if c.get("stream")]
    # Le premier message vocal est envoyé en audio (WAV) et le mode réflexion est coupé.
    vocal = flux[1]["messages"][-1]
    assert base64.b64decode(vocal["images"][0])[:4] == b"RIFF" and flux[1]["think"] is False
    assert flux[1]["tools"][0]["function"]["name"] == "raccrocher"
    assert "téléphone de la documentation du FabLab" in flux[0]["messages"][0]["content"]
    # Au tour suivant, la phrase précédente est dans l'historique sous forme de texte.
    assert any("contreplaqué de 3 mm" in str(m.get("content")) for m in flux[2]["messages"][:-1])
    assert {"transcription": "gemma4:e2b", "langue": "fr"} in journal

    assert chemin is not None and chemin.name.endswith("_lampe-en-contreplaque.md")
    contenu = chemin.read_text(encoding="utf-8")
    assert contenu.startswith("---") and "Transcription de l'appel" in contenu
    demande_doc = next(c for c in journal if c.get("stream") is False)
    assert "MODELE" in demande_doc["messages"][0]["content"] and demande_doc["think"] is False
