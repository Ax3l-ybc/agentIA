"""Tests automatiques (sans vrais modèles, sans Ollama).

Lancer :  .venv\\Scripts\\python -m pytest tests
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import socket
import threading
import time

import fastapi
import numpy as np
import pytest

from agent import coffre as coffre_mod
from agent import moteurs, reglages, serveur_voix

# ---------------------------------------------------------------------------
# Outils de test
# ---------------------------------------------------------------------------


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def empreinte(dossier: pathlib.Path) -> dict:
    return {
        str(p.relative_to(dossier)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(dossier.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def coffre_test(tmp_path):
    notes = tmp_path / "Coffre" / "Mémoire"
    (notes / "Chapitres").mkdir(parents=True)
    (notes / ".obsidian").mkdir()
    (notes / "Problématique.md").write_text(
        "# Problématique\nComment les tiers-lieux transforment-ils le travail rural ?\n",
        encoding="utf-8",
    )
    (notes / "Chapitres" / "Méthodologie.md").write_text(
        "# Méthodologie\n## Terrain\nEntretiens semi-directifs avec 12 coworkers.\n",
        encoding="utf-8",
    )
    (notes / ".obsidian" / "cache.md").write_text("ignoré", encoding="utf-8")
    recaps = tmp_path / "Coffre" / "Recaps"
    return notes, recaps


# ---------------------------------------------------------------------------
# Coffre : lecture seule
# ---------------------------------------------------------------------------


def test_lister_et_lire(coffre_test):
    notes, recaps = coffre_test
    c = coffre_mod.Coffre(notes, recaps)
    assert [n.relatif for n in c.lister()] == ["Chapitres/Méthodologie", "Problématique"]
    assert "tiers-lieux" in c.lire("problematique")["contenu"]  # sans accent
    assert c.lire("methodo")["note"] == "Chapitres/Méthodologie"
    assert "erreur" in c.lire("cuisine japonaise")
    assert c.chercher("entretiens coworkers")["resultats"][0]["note"] == "Chapitres/Méthodologie"
    assert c.index()[0]["intertitres"] == ["Méthodologie", "Terrain"]


def test_recaps_dans_notes_refuse(coffre_test):
    notes, _ = coffre_test
    with pytest.raises(ValueError):
        coffre_mod.Coffre(notes, notes / "Recaps")
    with pytest.raises(ValueError):
        coffre_mod.Coffre(notes, notes)


def test_fiche_jamais_ecrasee(coffre_test):
    notes, recaps = coffre_test
    c = coffre_mod.Coffre(notes, recaps)
    avant = empreinte(notes)
    p1 = c.ecrire_recap("un")
    p2 = c.ecrire_recap("deux", quand=None)
    assert p1.parent == recaps.resolve() and p1 != p2
    assert p1.read_text(encoding="utf-8") == "un"
    assert empreinte(notes) == avant


def test_reglages(tmp_path, coffre_test):
    notes, recaps = coffre_test
    fichier = tmp_path / "config.yaml"
    fichier.write_text(
        f"dossier_notes: '{notes}'\ndossier_recaps: '{recaps}'\nollama:\n  modele: 'x'\n",
        encoding="utf-8",
    )
    r = reglages.charger(fichier)
    assert r.ollama_modele == "x" and recaps.is_dir()
    fichier.write_text(
        f"dossier_notes: '{notes}'\ndossier_recaps: '{notes / 'sous'}'\n", encoding="utf-8"
    )
    with pytest.raises(reglages.ErreurReglages):
        reglages.charger(fichier)


# ---------------------------------------------------------------------------
# Audio et découpage du texte
# ---------------------------------------------------------------------------


def test_reechantillonneur_continu():
    r = moteurs.Reechantillonneur(24000, 48000)
    x = np.sin(np.linspace(0, 20, 24000)).astype(np.float32)
    morceaux = np.concatenate([r(x[i : i + 1000]) for i in range(0, len(x), 1000)])
    assert abs(len(morceaux) - 48000) <= 2
    assert np.max(np.abs(np.diff(morceaux))) < 0.01  # aucun saut entre morceaux


def test_couper_phrases():
    phrases, reste = serveur_voix.couper_phrases("Bonjour. Comment vas-tu ? Je", premiere=True)
    assert phrases == ["Bonjour.", "Comment vas-tu ?"] and reste == "Je"


def test_assembleur_mots():
    a = serveur_voix._AssembleurMots(delai_s=0.5)
    sortie = []
    for jeton, morceau in [(0, None), (5, "▁Bon"), (5, "jour"), (0, None), (5, "▁à"), (3, None), (3, None)]:
        sortie += a.pousser(jeton, morceau, 1.0)
    assert [m["text"] for m in sortie] == ["Bonjour", "à"]


def test_seance_ponctuation():
    from agent.application import Seance

    s = Seance()
    for mot in ["Pour", "cela", ",", "il", "faut"]:
        s.ajouter("etudiant", mot)
    assert s.transcription() == "Étudiant : Pour cela, il faut"


# ---------------------------------------------------------------------------
# Bout en bout : Gradbot + serveur vocal local + faux LLM
# ---------------------------------------------------------------------------


def faux_ollama_app(journal: list):
    """Imite l'API d'Ollama (OpenAI /v1 en streaming + /api/chat)."""
    from fastapi.responses import StreamingResponse

    app = fastapi.FastAPI()

    def sse(morceaux):
        for m in morceaux:
            yield "data: " + json.dumps(m) + "\n\n"
        yield "data: [DONE]\n\n"

    def bloc(delta, fin=None):
        return {
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "faux",
            "choices": [{"index": 0, "delta": delta, "finish_reason": fin}],
        }

    @app.post("/v1/chat/completions")
    async def completions(req: fastapi.Request):
        corps = await req.json()
        journal.append(corps)
        messages = corps["messages"]
        dernier = messages[-1]
        if dernier["role"] == "tool":
            texte = "Très bien, la fiche est en cours de rédaction."
            m = [bloc({"role": "assistant", "content": texte}), bloc({}, "stop")]
        elif dernier["role"] == "user" and "fiche" in str(dernier.get("content", "")).lower():
            appel = {
                "index": 0, "id": "appel1", "type": "function",
                "function": {"name": "rediger_fiche_recap", "arguments": "{}"},
            }
            m = [bloc({"role": "assistant", "tool_calls": [appel]}), bloc({}, "tool_calls")]
        else:
            texte = "Bonjour. Quelle est précisément ta problématique ?"
            m = [bloc({"role": "assistant", "content": texte}), bloc({}, "stop")]
        return StreamingResponse(sse(m), media_type="text/event-stream")

    @app.post("/api/chat")
    async def chat(req: fastapi.Request):
        corps = await req.json()
        journal.append(corps)
        return {"message": {"role": "assistant", "content": "## En bref\nSéance de test."}}

    return app


def lancer_serveur(app, port):
    import uvicorn

    serveur = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    fil = threading.Thread(target=serveur.run, daemon=True)
    fil.start()
    for _ in range(100):
        if serveur.started:
            return serveur
        time.sleep(0.05)
    raise RuntimeError("serveur non démarré")


def audio_ogg_opus(secondes_parole: float, secondes_silence: float) -> list[bytes]:
    import sphn

    ecrivain = sphn.OpusStreamWriter(24000)
    t = np.arange(int(secondes_parole * 24000)) / 24000
    parole = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    signal = np.concatenate([parole, np.zeros(int(secondes_silence * 24000), np.float32)])
    paquets = []
    for i in range(0, len(signal), 480):
        donnees = ecrivain.append_pcm(signal[i : i + 480])
        if donnees:
            paquets.append(donnees)
    return paquets


def test_bout_en_bout(coffre_test):
    pytest.importorskip("gradbot")
    pytest.importorskip("sphn")
    import websockets

    from agent.application import creer_app

    notes, recaps = coffre_test
    avant = empreinte(notes)
    journal: list = []
    port_llm, port_app = port_libre(), port_libre()
    faux = lancer_serveur(faux_ollama_app(journal), port_llm)
    r = reglages.Reglages(
        dossier_notes=notes,
        dossier_recaps=recaps,
        ollama_adresse=f"http://127.0.0.1:{port_llm}",
        port=port_app,
    )
    reglages.verifier_dossiers(r)
    app = creer_app(
        r,
        moteur_ecoute=moteurs.MoteurEcouteSimulation("Fais la fiche de la séance"),
        moteur_voix=moteurs.MoteurVoixSimulation(),
        modele_llm="faux",
    )
    serveur = lancer_serveur(app, port_app)

    async def scenario():
        recus = []
        async with websockets.connect(f"ws://127.0.0.1:{port_app}/ws/chat", max_size=None) as ws:
            await ws.send(json.dumps({"type": "start"}))

            async def lire():
                async for m in ws:
                    if isinstance(m, str):
                        recus.append(json.loads(m))
                    else:
                        recus.append({"type": "_audio", "n": len(m)})

            lecteur = asyncio.create_task(lire())
            await asyncio.sleep(2.0)  # salutation du directeur
            for paquet in audio_ogg_opus(1.5, 4.0):
                await ws.send(paquet)
                await asyncio.sleep(0.02)
            for _ in range(200):
                if any(m.get("type") == "fiche" and m.get("etat") == "ok" for m in recus):
                    break
                await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.sleep(0.3)
            lecteur.cancel()
        return recus

    try:
        recus = asyncio.run(scenario())
    finally:
        serveur.should_exit = True
        faux.should_exit = True

    types = [m["type"] for m in recus]
    assert "_audio" in types, types  # la voix a parlé
    textes_agent = " ".join(m["text"] for m in recus if m["type"] == "agent_text")
    assert "problématique" in textes_agent
    textes_etudiant = " ".join(m["text"] for m in recus if m["type"] == "user_text")
    assert "fiche" in textes_etudiant.lower()
    assert any(m.get("type") == "fiche" and m.get("etat") == "ok" for m in recus), recus[-20:]

    # Le LLM a bien reçu les notes dans ses consignes.
    assert "tiers-lieux" in json.dumps(journal[0], ensure_ascii=False)
    fiches = list(recaps.glob("*.md"))
    assert len(fiches) == 1
    contenu = fiches[0].read_text(encoding="utf-8")
    assert "## En bref" in contenu and "type: recap-seance" in contenu
    assert "Fais la fiche" in contenu  # transcription incluse
    # Les notes n'ont pas bougé d'un octet.
    assert empreinte(notes) == avant
