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
from agent import moteurs, reglages

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
        f"dossier_notes: '{notes}'\ndossier_recaps: '{recaps}'\n"
        "ollama:\n  modele: 'ancien réglage ignoré'\ncerveau:\n  taille_contexte: 8192\n"
        "  modele: 'gemma-4-E2B-it-Q3_K_M.gguf'\n",
        encoding="utf-8",
    )
    r = reglages.charger(fichier)
    assert r.llamacpp_contexte == 8192 and recaps.is_dir()
    assert r.llamacpp_modele == reglages.DOSSIER_DONNEES / "modeles" / "gemma-4-E2B-it-Q3_K_M.gguf"
    assert r.llamacpp_mmproj.name == "mmproj-F16.gguf" and r.llamacpp_couches_gpu == 0
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
# Audio, découpage du texte et des phrases
# ---------------------------------------------------------------------------


def test_reechantillonneur_continu():
    r = moteurs.Reechantillonneur(24000, 48000)
    x = np.sin(np.linspace(0, 20, 24000)).astype(np.float32)
    morceaux = np.concatenate([r(x[i : i + 1000]) for i in range(0, len(x), 1000)])
    assert abs(len(morceaux) - 48000) <= 2
    assert np.max(np.abs(np.diff(morceaux))) < 0.01  # aucun saut entre morceaux


def test_couper_phrases():
    from agent.texte_voix import couper_phrases, nettoyer_pour_voix

    phrases, reste = couper_phrases("Bonjour. Comment vas-tu ? Je", premiere=True)
    assert phrases == ["Bonjour.", "Comment vas-tu ?"] and reste == "Je"
    assert nettoyer_pour_voix("**Très** bien 🙂 [[Note]]") == "Très bien Note"


def detecteur_energie():
    return lambda bloc: 0.9 if float(np.abs(bloc).mean()) > 0.01 else 0.1


def ton(secondes: float, taux: int = 16000) -> np.ndarray:
    t = np.arange(int(secondes * taux)) / taux
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def test_decoupeur_phrases():
    from agent.phrases import DecoupeurPhrases

    d = DecoupeurPhrases(detecteur_energie(), pause_s=0.8)
    signal = np.concatenate([np.zeros(16000, np.float32), ton(2.0), np.zeros(24000, np.float32)])
    phrases = []
    for i in range(0, len(signal), 1600):
        phrases += d.ajouter(signal[i : i + 1600])
    assert len(phrases) == 1 and 2.5 <= len(phrases[0]) / 16000 <= 3.4
    d.reinitialiser()
    bref = np.concatenate([ton(0.1), np.zeros(24000, np.float32)])
    assert d.ajouter(bref) == []  # un bruit bref n'est pas une phrase


def test_seance_ponctuation():
    from agent.application import Seance

    s = Seance()
    s.ajouter("etudiant", "Pour cela , il faut")
    s.ajouter("directeur", "")
    assert s.transcription() == "Étudiant : Pour cela, il faut" and s.etudiant_a_parle()


def test_premier_port_libre():
    from agent.__main__ import premier_port_libre

    with socket.socket() as occupe:
        occupe.bind(("127.0.0.1", 0))
        occupe.listen()
        pris = occupe.getsockname()[1]
        libre = premier_port_libre(pris)
        assert libre is not None and libre != pris


# ---------------------------------------------------------------------------
# Bout en bout : page web simulée ↔ application ↔ faux llama.cpp
# ---------------------------------------------------------------------------

TRANSCRIPTIONS = [
    "Peux-tu relire ma note sur la problématique ?",
    "Fais-moi la fiche de la séance.",
]


def faux_llamacpp(journal: list):
    """Imite llama-server : conversation en flux (audio en input_audio), outils, deux espaces."""
    from fastapi.responses import StreamingResponse

    from agent.cerveau import CONSIGNE_TRANSCRIPTION

    app = fastapi.FastAPI()
    transcriptions = list(TRANSCRIPTIONS)
    messages_vocaux = [0]

    def sse(deltas):
        for d in deltas:
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": d}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    def appel(nom, args):
        return {"tool_calls": [{"index": 0, "id": f"id_{nom}",
                                "function": {"name": nom, "arguments": json.dumps(args)}}]}

    @app.post("/v1/chat/completions")
    async def chat(req: fastapi.Request):
        corps = await req.json()
        journal.append(corps)
        messages = corps["messages"]
        texte_complet = json.dumps(messages, ensure_ascii=False)
        if not corps.get("stream"):
            if CONSIGNE_TRANSCRIPTION in texte_complet:
                texte = transcriptions.pop(0) if transcriptions else ""
            elif "fiche récapitulative de clôture" in texte_complet:
                date = __import__("datetime").date.today().isoformat()
                texte = (f"`{date}_fiche_cadrage-terrain.md`\n```markdown\n---\ndate: {date}\n"
                         f"type: fiche-seance\n---\n\n# Séance du {date} — cadrage du terrain\n```")
            else:
                texte = "prêt"
            return {"choices": [{"message": {"role": "assistant", "content": texte}}],
                    "timings": {"prompt_n": 1, "cache_n": 0}}
        dernier = messages[-1]
        audio = isinstance(dernier["content"], list)
        if dernier["role"] == "tool" and dernier["tool_call_id"] == "id_lire_note":
            assert "tiers-lieux" in dernier["content"]  # le contenu de la note est bien fourni
            deltas = [{"content": "Dans ta note Problématique, tu parles des tiers-lieux. "},
                      {"content": "Qu'entends-tu par travail rural ?"}]
        elif dernier["role"] == "tool":
            deltas = [{"content": "C'est noté, la fiche est en cours de rédaction."}]
        elif audio:
            messages_vocaux[0] += 1
            if messages_vocaux[0] == 1:  # 1re phrase : il relit la note demandée
                deltas = [appel("lire_note", {"nom": "problematique"})]
            else:  # 2e phrase : « fais-moi la fiche »
                deltas = [appel("rediger_fiche_recap", {})]
        else:  # début de séance (ou lecture à l'avance des consignes)
            deltas = [{"content": "Bonjour ! Reprenons où nous en étions. "},
                      {"content": "Quelle est ta problématique aujourd'hui ?"}]
        return StreamingResponse(sse(deltas), media_type="text/event-stream")

    @app.get("/health")
    async def sante():
        return {"status": "ok"}

    return app


def lancer_serveur(app, port):
    import uvicorn

    serveur = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                            ws_ping_interval=None))
    fil = threading.Thread(target=serveur.run, daemon=True)
    fil.start()
    for _ in range(100):
        if serveur.started:
            return serveur
        time.sleep(0.05)
    raise RuntimeError("serveur non démarré")


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


def test_seance_complete(coffre_test):
    import websockets

    from agent.application import Agent, creer_app
    from agent.cerveau import CerveauLlamaCpp

    notes, recaps = coffre_test
    avant = empreinte(notes)
    journal: list = []
    port_llm, port_app = port_libre(), port_libre()
    faux = lancer_serveur(faux_llamacpp(journal), port_llm)
    r = reglages.Reglages(dossier_notes=notes, dossier_recaps=recaps, pause_s=0.8, port=port_app)
    reglages.verifier_dossiers(r)
    agent = Agent(r, CerveauLlamaCpp(f"http://127.0.0.1:{port_llm}"), moteurs.MoteurVoixSimulation(),
                  detecteur_energie)
    agent.prechauffer()
    serveur = lancer_serveur(creer_app(agent), port_app)

    async def parler(ws, signal):
        for i in range(0, len(signal), 1600):  # paquets de 100 ms, comme la page web
            await ws.send(moteurs.float_vers_pcm16(signal[i : i + 1600]))
            await asyncio.sleep(0.1)  # en temps réel

    async def scenario():
        recus, audio = [], [0]
        async with websockets.connect(f"ws://127.0.0.1:{port_app}/ws/seance", max_size=None) as ws:
            await ws.send(json.dumps({"type": "debut"}))

            async def attendre(condition, delai=30):
                fin = time.monotonic() + delai
                while time.monotonic() < fin:
                    if any(condition(m) for m in recus):
                        return
                    await asyncio.sleep(0.05)
                raise AssertionError(f"attendu non reçu ; derniers messages : {recus[-8:]}")

            async def lire():
                async for m in ws:
                    if isinstance(m, bytes):
                        audio[0] += len(m)
                    else:
                        recus.append(json.loads(m))

            lecteur = asyncio.create_task(lire())
            await attendre(lambda m: m.get("etat") == "ecoute")
            await parler(ws, np.concatenate([ton(1.5), np.zeros(16000, np.float32)]))
            await attendre(lambda m: "travail rural" in m.get("texte", ""))
            recus.clear()
            await attendre(lambda m: m.get("etat") == "ecoute")
            await parler(ws, np.concatenate([ton(1.5), np.zeros(16000, np.float32)]))
            await attendre(lambda m: m.get("type") == "fiche" and m.get("etat") == "ok", 60)
            fiche = next(m for m in recus if m.get("type") == "fiche" and m.get("etat") == "ok")
            await ws.send(json.dumps({"type": "stop"}))
            lecteur.cancel()
        return recus, audio[0], fiche

    try:
        recus, octets_audio, fiche = asyncio.run(scenario())
    finally:
        serveur.should_exit = True
        faux.should_exit = True

    assert octets_audio > 0  # la voix du directeur a été envoyée à la page
    assert fiche["fichier"].endswith("_fiche_cadrage-terrain.md")
    contenu = (recaps / fiche["fichier"]).read_text(encoding="utf-8")
    assert contenu.startswith("---") and "fiche-seance" in contenu
    # La retranscription des phrases de l'étudiant a été affichée.
    assert any(m.get("type") == "etudiant" and m.get("texte") == TRANSCRIPTIONS[1] for m in recus)

    flux = [c for c in journal if c.get("stream")]
    # Lecture à l'avance : consignes lues dans l'espace de conversation, un seul mot produit.
    assert flux[0]["id_slot"] == 0 and flux[0]["max_tokens"] == 1
    systeme = flux[0]["messages"][0]["content"]
    assert "MODE DE CONDUITE" in systeme and "tiers-lieux" in systeme
    # La séance réutilise exactement les mêmes consignes (pas de relecture).
    assert all(c["messages"][0]["content"] == systeme for c in flux)
    assert all(c["id_slot"] == 0 and c["cache_prompt"] for c in flux)
    vocal = next(c for c in flux if isinstance(c["messages"][-1]["content"], list))
    partie = next(p for p in vocal["messages"][-1]["content"] if p["type"] == "input_audio")
    assert __import__("base64").b64decode(partie["input_audio"]["data"])[:4] == b"RIFF"
    assert {o["function"]["name"] for o in vocal["tools"]} >= {"lire_note", "rediger_fiche_recap"}
    # Retranscriptions et fiche dans l'espace annexe (la conversation reste en mémoire).
    annexes = [c for c in journal if not c.get("stream")]
    assert annexes and all(c["id_slot"] == 1 for c in annexes)
    # Les notes n'ont pas bougé d'un octet.
    assert empreinte(notes) == avant

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
