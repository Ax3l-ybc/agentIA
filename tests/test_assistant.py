"""Tests de Mon Assistant, sans Gemma ni voix : faux llama.cpp, voix simulée."""

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
import uvicorn
from fastapi.responses import StreamingResponse

from assistant import moteurs
from assistant.application import Application, creer_app
from assistant.cerveau import CONSIGNE_TRANSCRIPTION, CerveauLlamaCpp
from assistant.conversations import Conversations
from assistant.documents import Documents
from assistant.profils import Profil, Profils, lire_fichier
from assistant.reglages import RACINE, Reglages, charger

# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lancer_serveur(app, port):
    serveur = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                            ws_ping_interval=None))
    threading.Thread(target=serveur.run, daemon=True).start()
    for _ in range(100):
        if serveur.started:
            return serveur
        time.sleep(0.05)
    raise RuntimeError("serveur non démarré")


def detecteur_energie():
    return lambda bloc: 0.9 if float(np.abs(bloc).mean()) > 0.01 else 0.1


def ton(secondes: float, taux: int = 16000) -> np.ndarray:
    t = np.arange(int(secondes * taux)) / taux
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def empreinte(dossier: pathlib.Path) -> dict:
    return {p.relative_to(dossier).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dossier.rglob("*")) if p.is_file()}


def faux_llamacpp(journal: list):
    """Imite llama-server : réponses en flux, outils, retranscription, mémoire des slots."""
    app = fastapi.FastAPI()

    def sse(deltas):
        for d in deltas:
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": d}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat(req: fastapi.Request):
        corps = await req.json()
        journal.append(corps)
        messages = corps["messages"]
        tout = json.dumps(messages, ensure_ascii=False)
        if not corps.get("stream"):
            texte = "Je voudrais un résumé de ma note." if CONSIGNE_TRANSCRIPTION in tout else "prêt"
            return {"choices": [{"message": {"role": "assistant", "content": texte}}]}
        dernier = messages[-1]
        if corps.get("max_tokens") == 1:
            deltas = [{"content": "Bon"}]
        elif dernier["role"] == "tool":
            deltas = [{"content": "D'après ta note Courses, il faut acheter du pain."}]
        elif isinstance(dernier["content"], list):  # message vocal
            deltas = [{"tool_calls": [{"index": 0, "id": "id1", "function": {
                "name": "lire_document", "arguments": json.dumps({"nom": "courses"})}}]}]
        elif "lent" in dernier["content"]:
            deltas = [{"content": f"Mot {i}. "} for i in range(400)]
        else:
            deltas = [{"content": "Bonjour ! "}, {"content": "Voici **trois** idées :\n- une\n- deux\n"},
                      {"content": "Laquelle te tente ?"}]
        return StreamingResponse(sse(deltas), media_type="text/event-stream")

    @app.post("/slots/{espace}")
    async def memoire(espace: int, action: str):
        journal.append({"memoire": action})
        return {"id_slot": espace}

    @app.get("/health")
    async def sante():
        return {"status": "ok"}

    return app


@pytest.fixture
def environnement(tmp_path):
    """Un assistant (avec un dossier de documents), un faux cerveau et le serveur de la page."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "courses.md").write_text("# Courses\n- pain\n- lait\n", encoding="utf-8")
    assistants = tmp_path / "assistants"
    assistants.mkdir()
    for f in (RACINE / "assistants").glob("*.md"):
        (assistants / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    journal: list = []
    port_llm, port_app = port_libre(), port_libre()
    faux = lancer_serveur(faux_llamacpp(journal), port_llm)
    r = Reglages(port=port_app, pause_s=0.8, dossier_assistants=assistants,
                 dossier_conversations=tmp_path / "conversations")
    etat = Application(r, CerveauLlamaCpp(f"http://127.0.0.1:{port_llm}"), moteurs.MoteurVoixSimulation(),
                       detecteur_energie)
    serveur = lancer_serveur(creer_app(etat), port_app)
    yield {"port": port_app, "journal": journal, "etat": etat, "docs": docs, "tmp": tmp_path}
    serveur.should_exit = True
    faux.should_exit = True


async def discuter(port, conv_id, assistant, actions, delai=30):
    """Ouvre la discussion, exécute `actions(ws, recus)` et renvoie les messages reçus."""
    import websockets

    recus, audio = [], [0]
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/{conv_id}?assistant={assistant}",
                                  max_size=None) as ws:
        async def lire():
            async for m in ws:
                if isinstance(m, bytes):
                    audio[0] += len(m)
                else:
                    recus.append(json.loads(m))

        lecteur = asyncio.create_task(lire())

        async def attendre(condition, d=delai):
            fin = time.monotonic() + d
            while time.monotonic() < fin:
                if any(condition(m) for m in recus):
                    return
                await asyncio.sleep(0.05)
            raise AssertionError(f"attendu non reçu ; derniers messages : {recus[-8:]}")

        await attendre(lambda m: m.get("etat") == "pret")
        await actions(ws, recus, attendre)
        lecteur.cancel()
    return recus, audio[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reglages_par_defaut_sans_fichier(tmp_path):
    r = charger(tmp_path / "absent.yaml")
    assert r.llamacpp_contexte == 16384 and r.anticipation_s == 0.4
    (tmp_path / "c.yaml").write_text("cerveau:\n  taille_contexte: 8192\nvoix:\n  vitesse: -0.1\n", encoding="utf-8")
    r = charger(tmp_path / "c.yaml")
    assert r.llamacpp_contexte == 8192 and r.voix_vitesse == -0.1


def test_config_fournie_est_lisible():
    r = charger(RACINE / "config.yaml")
    assert r.port == 8000 and r.dossier_assistants.name == "assistants"


def test_assistants_fournis():
    profils = Profils(RACINE / "assistants").lister()
    noms = {p.nom for p in profils}
    assert "Assistant du quotidien" in noms and "Coach d'écriture" in noms
    assert all(p.consignes and p.description for p in profils)


def test_profils_creer_modifier_supprimer(tmp_path):
    profils = Profils(tmp_path)
    p = profils.enregistrer(Profil(id="", nom="Coach d'écriture", consignes="Corrige mes textes.",
                                   description="Pour mes mails", voix="Hélène"))
    assert p.id == "coach-d-ecriture" and (tmp_path / "coach-d-ecriture.md").is_file()
    # Le fichier se relit (et se modifie) avec le Bloc-notes : en-tête simple + consignes.
    relu = lire_fichier(tmp_path / "coach-d-ecriture.md")
    assert relu.nom == "Coach d'écriture" and relu.consignes == "Corrige mes textes." and relu.voix == "Hélène"
    p2 = profils.enregistrer(Profil(id="", nom="Coach d'écriture", consignes="Autre."))
    assert p2.id == "coach-d-ecriture-2"  # jamais d'écrasement d'un autre assistant
    with pytest.raises(ValueError):
        profils.enregistrer(Profil(id="", nom="X", consignes="Y", dossier_documents=str(tmp_path / "absent")))
    profils.supprimer("coach-d-ecriture-2")
    assert not (tmp_path / "coach-d-ecriture-2.md").exists()
    assert len(list((tmp_path / "corbeille").glob("*.md"))) == 1  # rangé, pas effacé
    with pytest.raises(ValueError):
        profils.supprimer("coach-d-ecriture")  # on garde au moins un assistant
    assert profils.lire("../config") is None


def test_documents_lecture_seule(tmp_path):
    (tmp_path / "Projet.md").write_text("Le budget est de 300 euros.", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x.md").write_text("caché", encoding="utf-8")
    d = Documents(tmp_path)
    assert [rel for rel, _ in d.lister()] == ["Projet"]
    assert "300 euros" in d.lire("projet")["contenu"]
    assert d.chercher("budget")["resultats"][0]["document"] == "Projet"
    assert "erreur" in d.lire("inexistant")


def test_conversations(tmp_path):
    c = Conversations(tmp_path)
    conv = c.creer("assistant-du-quotidien")
    c.enregistrer(conv)
    assert c.lister() == []  # une conversation vide n'est pas gardée
    conv.ajouter("utilisateur", "Peux-tu m'aider à organiser un anniversaire surprise pour ma sœur ?")
    conv.ajouter("assistant", "Bien sûr !")
    c.enregistrer(conv)
    liste = c.lister()
    assert liste[0]["titre"].startswith("Peux-tu m'aider à organiser") and liste[0]["titre"].endswith("…")
    c.renommer(conv.id, "Anniversaire")
    assert c.charger(conv.id).titre == "Anniversaire"
    c.supprimer(conv.id)
    assert c.lister() == []


def test_discussion_ecrite_avec_lecture_a_voix_haute(environnement):
    env = environnement

    async def actions(ws, recus, attendre):
        await ws.send(json.dumps({"type": "texte", "contenu": "Donne-moi des idées de sortie", "lire": True}))
        await attendre(lambda m: m.get("type") == "fin")
        await attendre(lambda m: m.get("type") == "titre")

    recus, octets = asyncio.run(discuter(env["port"], "conv-ecrite", "assistant-du-quotidien", actions))
    morceaux = "".join(m["texte"] for m in recus if m.get("type") == "morceau")
    assert morceaux.startswith("Bonjour ! Voici **trois** idées")  # mot à mot, en Markdown
    assert octets > 0  # la réponse a aussi été lue à voix haute
    fin = next(m for m in recus if m.get("type") == "fin")
    assert fin["texte"].endswith("Laquelle te tente ?")
    conv = env["etat"].conversations.charger("conv-ecrite")
    assert conv.titre == "Donne-moi des idées de sortie"
    assert [m["role"] for m in conv.messages] == ["utilisateur", "assistant"]
    flux = [c for c in env["journal"] if c.get("stream") and c.get("max_tokens") != 1]
    systeme = flux[0]["messages"][0]["content"]
    assert systeme.startswith("Tu es mon assistant personnel") and "sans accès à internet" in systeme
    # Lecture à l'avance des consignes à l'ouverture : la même entrée que la vraie conversation.
    avance = next(c for c in env["journal"] if c.get("max_tokens") == 1)
    assert avance["messages"][0]["content"] == systeme


def test_reprendre_une_conversation(environnement):
    env = environnement

    async def premier(ws, recus, attendre):
        await ws.send(json.dumps({"type": "texte", "contenu": "Bonjour"}))
        await attendre(lambda m: m.get("type") == "fin")

    asyncio.run(discuter(env["port"], "conv-reprise", "coach-d-ecriture", premier))

    async def second(ws, recus, attendre):
        await ws.send(json.dumps({"type": "texte", "contenu": "Et ensuite ?"}))
        await attendre(lambda m: m.get("type") == "fin")

    asyncio.run(discuter(env["port"], "conv-reprise", "assistant-du-quotidien", second))
    conv = env["etat"].conversations.charger("conv-reprise")
    assert conv.assistant == "coach-d-ecriture"  # l'assistant de la conversation est gardé
    assert len(conv.messages) == 4
    dernier = [c for c in env["journal"] if c.get("stream") and c.get("max_tokens") != 1][-1]
    assert dernier["messages"][0]["content"].startswith("Tu es mon coach d'écriture")
    assert [m["content"] for m in dernier["messages"][1:]][:2] == ["Bonjour", dernier["messages"][2]["content"]]


def test_bouton_stop(environnement):
    env = environnement

    async def actions(ws, recus, attendre):
        await ws.send(json.dumps({"type": "texte", "contenu": "Réponds lentement"}))
        await attendre(lambda m: m.get("type") == "morceau")
        await ws.send(json.dumps({"type": "stop"}))
        await attendre(lambda m: m.get("type") == "fin")

    recus, _ = asyncio.run(discuter(env["port"], "conv-stop", "assistant-du-quotidien", actions))
    fin = next(m for m in recus if m.get("type") == "fin")
    assert fin["interrompue"] and "Mot 399" not in fin["texte"]


def test_conversation_a_la_voix_avec_documents(environnement):
    """À la voix, avec un dossier de documents : envoi anticipé, outil, retranscription."""
    env = environnement
    profils = env["etat"].profils
    p = profils.lire("assistant-du-quotidien")
    p.dossier_documents = str(env["docs"])
    profils.enregistrer(p)
    avant = empreinte(env["docs"])

    async def actions(ws, recus, attendre):
        await ws.send(json.dumps({"type": "voix", "active": True}))
        await attendre(lambda m: m.get("etat") == "ecoute")
        signal = np.concatenate([ton(1.5), np.zeros(24000, np.float32)])
        for i in range(0, len(signal), 1600):  # en temps réel, comme la page web
            await ws.send(moteurs.float_vers_pcm16(signal[i:i + 1600]))
            await asyncio.sleep(0.1)
        await attendre(lambda m: m.get("type") == "fin")
        await attendre(lambda m: m.get("type") == "utilisateur" and m.get("texte", "").startswith("Je voudrais"))

    recus, octets = asyncio.run(discuter(env["port"], "conv-voix", "assistant-du-quotidien", actions))
    assert octets > 0
    assert any(m.get("type") == "outil" and "courses" in m["texte"] for m in recus)
    fin = next(m for m in recus if m.get("type") == "fin")
    assert "acheter du pain" in fin["texte"]
    flux = [c for c in env["journal"] if c.get("stream") and c.get("max_tokens") != 1]
    # Les documents (petits) sont dans les consignes, et les outils sont proposés.
    assert "- pain" in flux[0]["messages"][0]["content"]
    assert {o["function"]["name"] for o in flux[0]["tools"]} == {"lister_documents", "lire_document",
                                                                 "chercher_dans_documents"}
    # La retranscription n'efface pas la mémoire de la conversation : sauvegarde puis restauration.
    for i, c in enumerate(env["journal"]):
        if "messages" in c and not c.get("stream"):
            assert env["journal"][i - 1].get("memoire") == "save"
            assert env["journal"][i + 1].get("memoire") == "restore"
    conv = env["etat"].conversations.charger("conv-voix")
    assert conv.messages[0]["mode"] == "voix" and conv.messages[0]["texte"].startswith("Je voudrais")
    assert conv.titre.startswith("Je voudrais")
    assert empreinte(env["docs"]) == avant  # les documents n'ont pas bougé d'un octet


def test_api_assistants(environnement):
    import urllib.request

    base = f"http://127.0.0.1:{environnement['port']}"

    def appel(methode, chemin, donnees=None):
        req = urllib.request.Request(base + chemin, method=methode,
                                     data=json.dumps(donnees).encode() if donnees is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    code, a = appel("POST", "/api/assistants", {"nom": "Chef cuisinier", "consignes": "Propose des recettes.",
                                                "voix": "Jules"})
    assert code == 200 and a["id"] == "chef-cuisinier"
    code, a = appel("PUT", "/api/assistants/chef-cuisinier", {**a, "description": "Recettes simples"})
    assert a["description"] == "Recettes simples"
    code, err = appel("POST", "/api/assistants", {"nom": "", "consignes": "x"})
    assert code == 400 and "nom" in err["erreur"]
    code, infos = appel("GET", "/api/infos")
    assert infos["voix_disponible"] and "Hélène" in infos["voix"]
    code, conv = appel("POST", "/api/conversations", {"assistant": "chef-cuisinier"})
    assert conv["assistant"] == "chef-cuisinier" and conv["messages"] == []
    code, _ = appel("DELETE", "/api/assistants/chef-cuisinier")
    assert code == 200


def test_decoupe_des_phrases_ecrites_pour_la_voix():
    from assistant.texte_voix import couper_phrases, nettoyer_pour_voix

    phrases, reste = couper_phrases("Voici :\n- un\n- deux, trois\nFin. Et", True)
    assert phrases == ["Voici :", "- un ", "- deux, trois ", "Fin."] and reste == "Et"
    assert nettoyer_pour_voix("**Gras** voir https://exemple.fr ok") == "Gras voir ok"
