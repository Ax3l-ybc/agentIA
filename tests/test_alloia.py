"""Tests d'AlloIA sans Jetson : combiné simulé, faux moteurs, faux Ollama."""

from __future__ import annotations

import asyncio
import datetime
import json
import pathlib

import fastapi
import numpy as np
import pytest
from fastapi.responses import StreamingResponse

from agent import moteurs, serveur_voix
from alloia import doc
from alloia.appel import Conversation
from alloia.reglages import RACINE, Reglages, charger
from tests.test_agent import lancer_serveur, port_libre

PHRASES = [
    "J'ai fabriqué une lampe en contreplaqué de 3 mm à la découpe laser.",
    "Le plus dur a été de régler la puissance du laser. C'est tout, au revoir.",
]

DOC = """```markdown
---
date: 2026-11-10
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


def faux_ollama(journal: list):
    app = fastapi.FastAPI()

    def sse(blocs):
        for b in blocs:
            yield "data: " + json.dumps(
                {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "f",
                 "choices": [{"index": 0, "delta": b[0], "finish_reason": b[1]}]}
            ) + "\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def completions(req: fastapi.Request):
        corps = await req.json()
        journal.append(corps)
        dernier = corps["messages"][-1]
        contenu = str(dernier.get("content", ""))
        # Après un outil, Gradbot relance le modèle avec un message vide.
        if dernier["role"] == "tool" or any(m["role"] == "tool" for m in corps["messages"][-3:]):
            blocs = [({"role": "assistant", "content": "Merci beaucoup, à bientôt !"}, None), ({}, "stop")]
        elif "au revoir" in contenu.lower():
            appel = {"index": 0, "id": "a1", "type": "function",
                     "function": {"name": "raccrocher", "arguments": "{}"}}
            blocs = [({"role": "assistant", "tool_calls": [appel]}, None), ({}, "tool_calls")]
        elif "[start]" in contenu:
            blocs = [({"role": "assistant", "content": "Allô ! Ici AlloIA. Quel projet viens-tu de terminer ?"}, None), ({}, "stop")]
        else:
            blocs = [({"role": "assistant", "content": "Génial ! Quel réglage de laser as-tu utilisé ?"}, None), ({}, "stop")]
        return StreamingResponse(sse(blocs), media_type="text/event-stream")

    @app.post("/api/chat")
    async def chat(req: fastapi.Request):
        journal.append(await req.json())
        return {"message": {"role": "assistant", "content": DOC}}

    return app


def test_reglages_par_defaut():
    r = charger()
    assert r.ollama_modele == "ministral-3:3b" and r.mode_de_conduite.name == "alloia_conduite.md"
    assert r.micro_coupe_quand_agent_parle is True


def test_nom_et_ecriture_doc(tmp_path):
    contenu = doc.nettoyer(DOC)
    assert contenu.startswith("---") and "```" not in contenu
    nom = doc.nom_fichier(contenu, datetime.datetime(2026, 11, 10, 14, 5))
    assert nom == "2026-11-10_14h05_lampe-en-contreplaque.md"
    p1 = doc.ecrire(tmp_path, nom, "a")
    p2 = doc.ecrire(tmp_path, nom, "b")
    assert p1 != p2 and p1.read_text() == "a"


def test_appel_trop_court_sans_doc(tmp_path):
    conv = Conversation(debut=datetime.datetime.now())
    conv.ajouter("alloia", "Allô !")
    assert doc.rediger(conv, "http://x", "m", 100, "", tmp_path) is None


def test_appel_complet(tmp_path):
    pytest.importorskip("gradbot")
    from alloia import audio
    from alloia.__main__ import demarrer_serveur_voix
    from alloia.appel import Appel, attendre_decroche

    phrases = list(PHRASES)

    def transcrire(_audio):
        return phrases.pop(0) if phrases else ""

    serveur_voix.MOTEURS["ecoute"] = moteurs.EcouteParPhrases(
        lambda b: 0.9 if float(np.abs(b).mean()) > 0.01 else 0.1, transcrire, pause_s=0.8
    )
    serveur_voix.MOTEURS["voix"] = moteurs.MoteurVoixSimulation()
    journal: list = []
    port_llm = port_libre()
    faux = lancer_serveur(faux_ollama(journal), port_llm)
    r = Reglages(dossier_docs=tmp_path / "docs", ollama_adresse=f"http://127.0.0.1:{port_llm}",
                 port=port_libre(), fin_apres_silence_s=30)
    consignes = (RACINE / "alloia_conduite.md").read_text(encoding="utf-8")

    async def scenario():
        serveur, _ = await demarrer_serveur_voix(r.port)
        combine = audio.CombineSimule([
            (1.0, ton(1.0)),   # « Allô ? » : décroche
            (4.0, ton(2.5)),   # première phrase, après la salutation
            (4.0, ton(2.5)),   # deuxième phrase, avec « au revoir »
        ])
        await combine.demarrer()
        clavier = asyncio.Event()
        try:
            cause = await asyncio.wait_for(attendre_decroche(combine, 0.6, clavier), 10)
            conv = await asyncio.wait_for(
                Appel(r, "faux", consignes).derouler(combine, clavier), 60
            )
        finally:
            await combine.arreter()
            serveur.should_exit = True
        return cause, conv, combine

    try:
        cause, conv, combine = asyncio.run(scenario())
        chemin = doc.rediger(conv, r.ollama_adresse, "faux", 8192, "MODELE", r.dossier_docs)
    finally:
        faux.should_exit = True

    assert cause == "voix"
    assert conv.raison_fin == "l'agent a raccroché"
    texte = conv.transcription()
    assert "Allô ! Ici AlloIA" in texte and "contreplaqué de 3 mm" in texte and "à bientôt" in texte
    assert float(np.abs(combine.audio_joue()).max()) > 0.05  # la voix est sortie du haut-parleur
    # Les consignes d'AlloIA ont bien été données au cerveau.
    assert "téléphone de la documentation du FabLab" in json.dumps(journal[0], ensure_ascii=False)
    assert chemin is not None and chemin.name.endswith("_lampe-en-contreplaque.md")
    contenu = chemin.read_text(encoding="utf-8")
    assert contenu.startswith("---") and "Transcription de l'appel" in contenu
    demande_doc = next(c for c in journal if "options" in c)
    assert "MODELE" in demande_doc["messages"][0]["content"]
    assert "contreplaqué de 3 mm" in demande_doc["messages"][1]["content"]
