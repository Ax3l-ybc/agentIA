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


def test_reglages_apostrophe_dans_le_texte(tmp_path, coffre_test):
    notes, recaps = coffre_test
    fichier = tmp_path / "config.yaml"
    fichier.write_text(
        f"dossier_notes: '{notes}'\ndossier_recaps: '{recaps}'\n"
        "sujet_memoire: 'Service d'appui à la pédagogie'   # commentaire\n",
        encoding="utf-8",
    )
    assert reglages.charger(fichier).sujet_memoire == "Service d'appui à la pédagogie"


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
        date = __import__("datetime").date.today().isoformat()
        texte = (
            f"`{date}_fiche_cadrage-terrain.md`\n\n```markdown\n---\ndate: {date}\n"
            "type: fiche-seance\nprojet: memoire\n---\n\n"
            f"# Séance du {date} — cadrage du terrain\n\n## Objet de la séance\nTest.\n```\n"
        )
        return {"message": {"role": "assistant", "content": texte}}

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
    assert fiches[0].name.endswith("_fiche_cadrage-terrain.md")
    assert contenu.startswith("---\ndate:") and "type: fiche-seance" in contenu
    assert "```" not in contenu
    # La fiche a été demandée avec le mode de conduite et la transcription.
    demande_fiche = next(c for c in journal if "options" in c)
    assert "MODE DE CONDUITE" in demande_fiche["messages"][0]["content"]
    assert "Fais la fiche" in demande_fiche["messages"][1]["content"]
    # Les notes n'ont pas bougé d'un octet.
    assert empreinte(notes) == avant


def test_ollama_preparer_affiche_la_progression(caplog):
    from fastapi.responses import StreamingResponse

    from agent import ollama

    app = fastapi.FastAPI()
    cree = {}

    @app.get("/api/version")
    async def version():
        return {"version": "0.0"}

    @app.get("/api/tags")
    async def tags():
        return {"models": []}

    @app.post("/api/pull")
    async def pull():
        def flux():
            yield json.dumps({"status": "pulling manifest"}) + "\n"
            for fait in range(0, 5_000_000_001, 500_000_000):
                yield json.dumps({"digest": "a", "total": 5_000_000_000, "completed": fait}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"

        return StreamingResponse(flux(), media_type="application/x-ndjson")

    @app.post("/api/create")
    async def create(req: fastapi.Request):
        cree.update(await req.json())
        return {"status": "success"}

    port = port_libre()
    serveur = lancer_serveur(app, port)
    try:
        with caplog.at_level("INFO", logger="agent.ollama"):
            nom = ollama.preparer(f"http://127.0.0.1:{port}", "qwen2.5:7b", 16384)
    finally:
        serveur.should_exit = True
    assert nom == ollama.NOM_MODELE_AGENT
    assert cree["from"] == "qwen2.5:7b" and cree["parameters"]["num_ctx"] == 16384
    assert "100 %" in caplog.text and "5.0 Go" in caplog.text


def test_premier_port_libre():
    from agent.__main__ import premier_port_libre

    with socket.socket() as occupe:
        occupe.bind(("127.0.0.1", 0))
        occupe.listen()
        pris = occupe.getsockname()[1]
        libre = premier_port_libre(pris)
        assert libre is not None and libre != pris


def _ecoute_par_phrases_test(transcrites):
    def prob(bloc):
        return 0.9 if float(np.abs(bloc).mean()) > 0.01 else 0.1

    def transcrire(audio):
        transcrites.append(len(audio) / 16000)
        return "Ma problématique porte sur les tiers-lieux."

    return moteurs.EcouteParPhrases(prob, transcrire, pause_s=0.8)


def test_ecoute_par_phrases():
    transcrites: list = []
    e = _ecoute_par_phrases_test(transcrites)
    t = np.arange(1920) / 24000
    voix = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    silence = np.zeros(1920, np.float32)
    sorties = [e.pas_mots(silence) for _ in range(10)]  # 0,8 s de silence
    sorties += [e.pas_mots(voix) for _ in range(25)]  # 2 s de parole
    sorties += [e.pas_mots(silence) for _ in range(15)]  # 1,2 s de pause
    mots = [m for ms, _ in sorties for m in ms]
    assert [m for m, _ in mots] == "Ma problématique porte sur les tiers-lieux.".split()
    assert abs(mots[0][1] - 0.8) < 0.1  # début de la phrase
    assert 2.0 <= transcrites[0] <= 3.2  # la phrase (avec un peu d'avant et la pause)
    probas = [p[2] for _, p in sorties]
    assert probas[0] == 1.0 and probas[15] == 0.05  # silence puis parole
    i_mots = next(i for i, (ms, _) in enumerate(sorties) if ms)
    assert probas[i_mots - 1] == 0.6  # pause avant la transcription : ni parole ni fin de tour
    assert probas[i_mots] == 1.0  # fin de tour annoncée avec les mots


def test_nettoyer_transcription():
    assert moteurs.nettoyer_transcription(" Sous-titres réalisés par la communauté d'Amara.org ") == ""
    assert moteurs.nettoyer_transcription(" Bonjour. ") == "Bonjour."


def test_choix_moteur_ecoute():
    from agent.__main__ import choisir_moteur_ecoute

    r = reglages.Reglages(dossier_notes=pathlib.Path("."), dossier_recaps=pathlib.Path("r"))
    r.materiel = "cpu"
    assert choisir_moteur_ecoute(r) == "whisper"
    r.stt_moteur = "kyutai"
    assert choisir_moteur_ecoute(r) == "kyutai"


def _trames_voix(n):
    t = np.arange(1920) / 24000
    voix = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return [voix] * n


def test_ecoute_longue_tirade_par_morceaux():
    transcrites: list = []
    e = _ecoute_par_phrases_test(transcrites)
    silence = np.zeros(1920, np.float32)
    # 20 s de parole avec une micro-pause de 0,32 s toutes les 4 s, puis une vraie pause.
    trames = []
    for _ in range(5):
        trames += _trames_voix(46) + [silence] * 4
    sorties = [e.pas_mots(tr) for tr in trames]
    probas_pendant = [p[2] for _, p in sorties]
    assert 1.0 not in probas_pendant  # jamais « fin de tour » tant que vous parlez
    assert len(transcrites) >= 2  # des morceaux ont été livrés en cours de route
    assert all(d <= 8.5 for d in transcrites)
    fin = [e.pas_mots(silence) for _ in range(15)]
    assert any(ms for ms, _ in fin) and fin[-1][1][2] == 1.0
    assert abs(sum(transcrites) - 20.3) < 1.0  # rien n'est perdu, rien n'est doublé


def test_ecoute_phrase_conservee_si_redemarrage():
    transcrites: list = []
    e = _ecoute_par_phrases_test(transcrites)
    for tr in _trames_voix(25):  # 2 s de parole
        e.pas_mots(tr)
    e.reinitialiser()  # Gradbot redémarre l'écoute en pleine phrase
    for tr in _trames_voix(10):
        e.pas_mots(tr)
    sorties = [e.pas_mots(np.zeros(1920, np.float32)) for _ in range(15)]
    assert any(ms for ms, _ in sorties)
    assert transcrites[0] >= 2.8  # les 2 s d'avant le redémarrage sont incluses


def test_extraire_fiche():
    import datetime as dt

    from agent import consignes

    d = dt.date(2026, 11, 5)
    nom, contenu = consignes.extraire_fiche(
        "Nom : `2026-11-05_fiche_role-des-sap.md`\n```markdown\n---\ndate: 2026-11-05\n---\n# Séance\n```", d
    )
    assert nom == "2026-11-05_fiche_role-des-sap.md" and contenu == "---\ndate: 2026-11-05\n---\n# Séance\n"
    nom, contenu = consignes.extraire_fiche("---\ndate: x\n---\n# Séance du 2026-11-05 — Rôle des IGP\n", d)
    assert nom == "2026-11-05_fiche_role-des-igp.md" and contenu.startswith("---")


def test_consignes_mode_et_fiches(coffre_test, tmp_path):
    from agent import consignes

    notes, recaps = coffre_test
    c = coffre_mod.Coffre(notes, recaps)
    recaps.mkdir(parents=True, exist_ok=True)
    (recaps / "2026-10-01_fiche_ancienne.md").write_text("ancienne", encoding="utf-8")
    (recaps / "2026-11-01_fiche_recente.md").write_text("questions ouvertes : rôle des IGP", encoding="utf-8")
    assert [p.stem for p in c.lister_fiches()] == ["2026-11-01_fiche_recente", "2026-10-01_fiche_ancienne"]
    assert "IGP" in c.lire_fiche("recente")["contenu"]
    mode_fichier = tmp_path / "mode.md"
    mode_fichier.write_text("# Rôle\nTu es mon directeur, autour d'un café.", encoding="utf-8")
    texte = consignes.construire(
        consignes.lire_mode(mode_fichier), "", c.index(), None, c.fiches_recentes(1),
        [p.stem for p in c.lister_fiches()],
    )
    assert texte.index("autour d'un café") < texte.index("Adaptation à la conversation orale")
    assert "rôle des IGP" in texte and "2026-10-01_fiche_ancienne" in texte
    assert consignes.lire_mode(tmp_path / "absent.md") == consignes.MODE_PAR_DEFAUT.strip()


def test_mode_de_conduite_du_projet_est_charge():
    from agent import consignes, reglages as r

    texte = consignes.lire_mode(r.RACINE / "mode_de_conduite.md")
    assert "services d'appui à la pédagogie" in texte and "fiche-seance" in texte
