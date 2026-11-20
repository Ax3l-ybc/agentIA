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
            if CONSIGNE_TRANSCRIPTION in tout:
                texte = "Je voudrais un résumé de ma note."
            elif "compte rendu d'une réunion" in tout:
                texte = "## Résumé\nPoint d'équipe.\n\n## Décisions\n- Commander du pain."
            elif "Prends des notes" in tout:
                texte = "- On parle du pain."
            else:
                texte = "prêt"
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
                 dossier_conversations=tmp_path / "conversations", dossier_utilisateurs=tmp_path / "personnes")
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
    assert noms == {"Assistant du quotidien", "Coach d'écriture"}
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


def test_raccourci_bureau():
    script = (RACINE / "assistant" / "raccourci.ps1").read_bytes()
    assert script.startswith(b"\xef\xbb\xbf")  # UTF-8 avec BOM : Windows PowerShell lit bien le « è »
    texte = script.decode("utf-8-sig")
    assert "Meilleur collègue.lnk" in texte and "lancer.bat" in texte
    assert (RACINE / "assistant" / "static" / "meilleur-collegue.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"
    assert "raccourci.ps1" in (RACINE / "installer.bat").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fichiers joints, export Word, comptes rendus de réunion
# ---------------------------------------------------------------------------

def pdf_simple(texte: str) -> bytes:
    """Un petit PDF d'une page contenant `texte` (sans bibliothèque)."""
    flux = f"BT /F1 12 Tf 72 720 Td ({texte}) Tj ET".encode("latin-1")
    objets = [b"<< /Type /Catalog /Pages 2 0 R >>",
              b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
              b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
              b"/Resources << /Font << /F1 5 0 R >> >> >>",
              b"<< /Length " + str(len(flux)).encode() + b" >>\nstream\n" + flux + b"\nendstream",
              b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    sortie, positions = b"%PDF-1.4\n", []
    for i, o in enumerate(objets, 1):
        positions.append(len(sortie))
        sortie += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    debut_xref = len(sortie)
    sortie += f"xref\n0 {len(objets) + 1}\n0000000000 65535 f \n".encode()
    sortie += b"".join(f"{p:010d} 00000 n \n".encode() for p in positions)
    sortie += f"trailer << /Size {len(objets) + 1} /Root 1 0 R >>\nstartxref\n{debut_xref}\n%%EOF".encode()
    return sortie


def docx_simple() -> bytes:
    import io

    import docx

    d = docx.Document()
    d.add_heading("Compte rendu", level=1)
    d.add_paragraph("Le budget est de 300 euros.")
    d.add_paragraph("Commander les badges", style="List Bullet")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Poste", "Coût"
    sortie = io.BytesIO()
    d.save(sortie)
    return sortie.getvalue()


def test_lire_pdf_word_texte():
    from assistant.fichiers import FichierIllisible, extraire_texte

    assert "Budget 2026 : 300 euros" in extraire_texte("rapport.pdf", pdf_simple("Budget 2026 : 300 euros"))
    texte = extraire_texte("cr.docx", docx_simple())
    assert "# Compte rendu" in texte and "300 euros" in texte and "- Commander les badges" in texte
    assert "Poste | Coût" in texte
    assert extraire_texte("note.txt", "Réunion à 14 h".encode("cp1252")) == "Réunion à 14 h"
    for nom, donnees in (("vieux.doc", b"x"), ("image.png", b"x"), ("vide.txt", b"  ")):
        with pytest.raises(FichierIllisible):
            extraire_texte(nom, donnees)


def test_export_word():
    import io

    import docx

    from assistant.fichiers import document_word, nom_de_fichier

    donnees = document_word("Mon titre", [("Moi", "Une question ?"),
                                          ("Assistant", "## Idées\n- **une** idée\n1. deux\n\nFin `code`.")])
    d = docx.Document(io.BytesIO(donnees))
    textes = [(p.style.name, p.text) for p in d.paragraphs]
    assert ("Title", "Mon titre") in textes and ("Heading 2", "Idées") in textes
    assert ("List Bullet", "une idée") in textes and ("List Number", "deux") in textes
    assert nom_de_fichier('Réunion: 12/05 "budget"') == "Reunion 12 05 budget.docx"
    assert nom_de_fichier("Combien me sera remboursé un dîner, et dans…") == "Combien me sera rembourse un diner, et dans.docx"


def test_message_avec_fichier_joint_et_export(environnement):
    import urllib.request

    env = environnement
    base = f"http://127.0.0.1:{env['port']}"
    req = urllib.request.Request(base + "/api/fichiers?nom=rapport.pdf", data=pdf_simple("Chiffre d'affaires : 42 k"),
                                 method="POST")
    with urllib.request.urlopen(req) as r:
        piece = json.load(r)
    assert piece["nom"] == "rapport.pdf" and not piece["tronque"]
    req = urllib.request.Request(base + "/api/fichiers?nom=photo.png", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as erreur:
        urllib.request.urlopen(req)
    assert "format non pris en charge" in json.load(erreur.value)["erreur"]

    async def actions(ws, recus, attendre):
        await ws.send(json.dumps({"type": "texte", "contenu": "", "fichiers": [piece["id"]]}))
        await attendre(lambda m: m.get("type") == "fin")

    recus, _ = asyncio.run(discuter(env["port"], "conv-fichier", "assistant-du-quotidien", actions))
    flux = [c for c in env["journal"] if c.get("stream") and c.get("max_tokens") != 1]
    envoye = flux[-1]["messages"][-1]["content"]
    assert "=== FICHIER JOINT : rapport.pdf ===" in envoye and "42 k" in envoye
    conv = env["etat"].conversations.charger("conv-fichier")
    assert conv.titre == "rapport.pdf" and conv.messages[0]["fichiers"][0]["nom"] == "rapport.pdf"
    fin = next(m for m in recus if m.get("type") == "fin")
    assert fin["index"] == 1
    # Export Word : une réponse, puis toute la conversation.
    with urllib.request.urlopen(base + "/api/conversations/conv-fichier/word?message=1") as r:
        assert r.headers["Content-Type"].startswith("application/vnd.openxmlformats")
        assert r.headers["Content-Disposition"] == 'attachment; filename="rapport.pdf.docx"'
        assert r.read()[:2] == b"PK"
    with urllib.request.urlopen(base + "/api/conversations/conv-fichier/word") as r:
        assert r.read()[:2] == b"PK"


def wav_reunion(secondes_parole=(3.0, 2.0), taux=48000) -> bytes:
    """Un faux enregistrement de réunion : de la « parole » séparée par des silences."""
    import io
    import wave

    morceaux = [np.zeros(taux, np.float32)]
    for s in secondes_parole:
        morceaux += [ton(s, taux), np.zeros(int(1.5 * taux), np.float32)]
    signal = np.concatenate(morceaux)
    sortie = io.BytesIO()
    with wave.open(sortie, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(taux)
        w.writeframes((signal * 32767).astype("<i2").tobytes())
    return sortie.getvalue()


def attendre_reunion(base, ident, delai=60):
    import urllib.request

    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        with urllib.request.urlopen(f"{base}/api/reunions/{ident}") as r:
            etat = json.load(r)
        if etat["etape"] in ("termine", "erreur"):
            return etat
        time.sleep(0.2)
    raise AssertionError(f"réunion non terminée : {etat}")


def test_decoder_et_decouper_une_reunion():
    from assistant.reunion import decoder_audio, decouper_en_parties, horodatage

    audio = decoder_audio(wav_reunion())
    assert abs(len(audio) / 16000 - 9.0) < 0.1  # rééchantillonné en 16 kHz mono
    assert horodatage(75) == "01:15" and horodatage(3725) == "1:02:05"
    parties = decouper_en_parties(["a" * 50] * 10, maximum=120)
    assert len(parties) == 5


def test_compte_rendu_depuis_un_fichier_audio(environnement):
    import urllib.request

    env = environnement
    base = f"http://127.0.0.1:{env['port']}"
    req = urllib.request.Request(base + "/api/reunions?nom=reunion.wav&assistant=coach-d-ecriture",
                                 data=wav_reunion(), method="POST")
    with urllib.request.urlopen(req) as r:
        ident = json.load(r)["id"]
    etat = attendre_reunion(base, ident)
    assert etat["etape"] == "termine", etat
    assert etat["total"] == 1  # deux phrases courtes : un seul morceau à retranscrire
    conv = env["etat"].conversations.charger(etat["conversation"])
    assert conv.assistant == "coach-d-ecriture" and conv.titre.startswith("Réunion du ")
    assert conv.messages[0]["fichiers"][0]["nom"] == "Transcription de la réunion"
    assert "Je voudrais un résumé" in conv.messages[0]["fichiers"][0]["texte"]
    compte_rendu = conv.messages[1]["texte"]
    assert compte_rendu.startswith("# Compte rendu") and "Commander du pain" in compte_rendu
    transcriptions = list((env["tmp"] / "reunions").glob("*_transcription.txt"))
    assert len(transcriptions) == 1 and "[00:0" in transcriptions[0].read_text(encoding="utf-8")


def test_compte_rendu_d_un_enregistrement_en_direct(environnement):
    import urllib.request

    import websockets

    env = environnement
    base = f"http://127.0.0.1:{env['port']}"

    async def enregistrer():
        async with websockets.connect(f"ws://127.0.0.1:{env['port']}/ws-reunion?assistant=") as ws:
            ident = json.loads(await ws.recv())["id"]
            signal = np.concatenate([ton(2.0), np.zeros(16000, np.float32), ton(1.0)])
            for i in range(0, len(signal), 1600):
                await ws.send(moteurs.float_vers_pcm16(signal[i:i + 1600]))
            await ws.send(json.dumps({"type": "fin"}))
            return ident

    ident = asyncio.run(enregistrer())
    etat = attendre_reunion(base, ident)
    assert etat["etape"] == "termine", etat
    conv = env["etat"].conversations.charger(etat["conversation"])
    assert conv.assistant == "assistant-du-quotidien" and "Commander du pain" in conv.messages[1]["texte"]


# ---------------------------------------------------------------------------
# Point 8 : choix du cerveau ; profils des personnes
# ---------------------------------------------------------------------------

def test_conseil_selon_le_pc():
    from assistant.modeles import conseil

    assert (conseil(16, 0).ecoute, conseil(16, 0).redacteur, conseil(16, 0).carte_graphique) == ("e2b", None, False)
    assert conseil(32, 0).ecoute == "e4b" and conseil(32, 0).redacteur is None
    assert conseil(64, 0).redacteur == "26b"  # sur processeur, seul 26B-A4B reste assez rapide
    nvidia_8 = conseil(16, 8)
    assert (nvidia_8.ecoute, nvidia_8.redacteur, nvidia_8.carte_graphique) == ("e4b", None, True)
    assert conseil(32, 16).redacteur == "12b"  # 7 Go (E4B) + 9 Go (12B) sur une carte de 16 Go
    assert conseil(64, 48).redacteur == "31b"


def test_choix_enregistre(tmp_path):
    from assistant.modeles import Choix, ecrire_choix, lire_choix

    assert lire_choix(tmp_path / "absent.json") == Choix()
    ecrire_choix(Choix("e4b", "12b", True), tmp_path / "c.json")
    assert lire_choix(tmp_path / "c.json") == Choix("e4b", "12b", True)
    (tmp_path / "c.json").write_text('{"ecoute": "12b"}')  # 12B n'écoute pas : choix refusé
    assert lire_choix(tmp_path / "c.json") == Choix()


def test_fichiers_llamacpp_pour_la_carte():
    from assistant.llamacpp_gpu import choisir_fichiers

    noms = ["llama-v0.9.0-bin-win-cpu-x64.zip", "llama-v0.9.0-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip", "llama-v0.9.0-bin-win-cuda-13.4-x64.zip",
            "cudart-llama-bin-win-cuda-13.4-x64.zip", "llama-v0.9.0-bin-win-cuda-13.4-arm64.zip",
            "llama-v0.9.0-bin-win-vulkan-x64.zip", "llama-v0.9.0-bin-win-vulkan-arm64.zip"]
    assert choisir_fichiers(noms, "nvidia") == ["llama-v0.9.0-bin-win-cuda-12.4-x64.zip",
                                                "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    assert choisir_fichiers(noms, "vulkan") == ["llama-v0.9.0-bin-win-vulkan-x64.zip"]


def test_telechargement_avec_reprise(tmp_path):
    from assistant.modeles import telecharger_fichier

    contenu = bytes(range(256)) * 4000

    class Serveur(__import__("http.server").server.BaseHTTPRequestHandler):
        def do_GET(self):
            debut = int((self.headers.get("Range") or "bytes=0-").split("=")[1].split("-")[0])
            self.send_response(206 if debut else 200)
            self.send_header("Content-Length", str(len(contenu) - debut))
            self.end_headers()
            self.wfile.write(contenu[debut:])

        def log_message(self, *a):
            pass

    import http.server

    serveur = http.server.HTTPServer(("127.0.0.1", 0), Serveur)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    cible = tmp_path / "modele.gguf"
    (tmp_path / "modele.gguf.partiel").write_bytes(contenu[:300000])  # téléchargement interrompu
    etapes = []
    telecharger_fichier(f"http://127.0.0.1:{serveur.server_address[1]}/m", cible, lambda r, t: etapes.append((r, t)))
    serveur.shutdown()
    assert cible.read_bytes() == contenu and not (tmp_path / "modele.gguf.partiel").exists()
    assert etapes[-1] == (len(contenu), len(contenu))


class FauxServeur:
    """Remplace ServeurLlamaCpp : un « llama.cpp » qui démarre (ou pas) à la demande."""

    pannes: set = set()
    adresses: dict = {}

    def __init__(self, r, journal, modele=None, mmproj=None, port=None, couches_gpu=0, nom=""):
        self.modele, self.mmproj, self.nom, self.couches_gpu = modele, mmproj, nom, couches_gpu
        self.cerveau = CerveauLlamaCpp(FauxServeur.adresses.get(nom, "http://127.0.0.1:1"))
        self.demarre = False

    def demarrer(self):
        from assistant.cerveau import ErreurCerveau

        if self.modele.name in FauxServeur.pannes:
            raise ErreurCerveau(f"pas assez de mémoire pour {self.modele.name}")
        self.demarre = True

    def arreter(self):
        self.demarre = False

    def reparer(self):
        pass


def test_gestion_des_cerveaux(tmp_path, monkeypatch):
    from assistant import modeles
    from assistant.cerveaux import GestionCerveaux
    from assistant.modeles import Choix

    monkeypatch.setattr(modeles, "FICHIER_CHOIX", tmp_path / "cerveau.json")
    monkeypatch.setattr(modeles, "a_telecharger", lambda m: [])
    ecrits = []
    monkeypatch.setattr(modeles, "ecrire_choix", lambda c, f=None: ecrits.append(c))
    g = GestionCerveaux(Reglages(), Choix(), fabrique=FauxServeur)
    g.demarrer()
    assert g.ecoute.modele.name == "gemma-4-E2B-it-Q4_K_M.gguf" and g.ecoute.mmproj is not None
    assert g.redaction is None and g.redacteur is g.cerveau
    # Passage à E4B + 12B sur la carte graphique.
    g._appliquer(Choix("e4b", "12b", False))
    assert g.etat["etape"] == "pret" and g.description() == "Gemma 4 E4B + Gemma 4 12B"
    assert g.redaction.mmproj is None and g.redaction.nom == "rédaction" and g.redacteur is not g.cerveau
    assert ecrits[-1] == Choix("e4b", "12b", False)
    # Un choix qui ne démarre pas : l'ancien est remis en route.
    FauxServeur.pannes = {"gemma-4-31B-it-Q4_K_M.gguf"}
    g._appliquer(Choix("e4b", "31b", False))
    FauxServeur.pannes = set()
    assert g.etat["etape"] == "erreur" and "pas assez de mémoire" in g.etat["message"]
    assert "L'ancien cerveau a été remis en route" in g.etat["message"]
    assert g.choix == Choix("e4b", "12b", False) and g.ecoute.demarre and g.redaction.demarre
    assert g.pret.is_set()
    # Au démarrage, un rédacteur qui ne démarre pas n'empêche pas l'assistant de marcher.
    FauxServeur.pannes = {"gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"}
    g2 = GestionCerveaux(Reglages(), Choix("e2b", "26b"), fabrique=FauxServeur)
    g2.demarrer()
    FauxServeur.pannes = set()
    assert g2.redaction is None and g2.choix.redacteur is None and g2.ecoute.demarre


def test_ecrit_au_redacteur_voix_a_l_ecoute(tmp_path, monkeypatch):
    """Avec deux cerveaux : les messages écrits vont au rédacteur, la voix et les
    retranscriptions au cerveau d'écoute."""
    from assistant.cerveaux import GestionCerveaux
    from assistant.modeles import Choix

    journal_ecoute, journal_redaction = [], []
    port_e, port_r, port_app = port_libre(), port_libre(), port_libre()
    faux_e = lancer_serveur(faux_llamacpp(journal_ecoute), port_e)
    faux_r = lancer_serveur(faux_llamacpp(journal_redaction), port_r)
    FauxServeur.adresses = {"écoute": f"http://127.0.0.1:{port_e}", "rédaction": f"http://127.0.0.1:{port_r}"}
    g = GestionCerveaux(Reglages(), Choix("e2b", "12b"), fabrique=FauxServeur)
    g.demarrer()
    assistants = tmp_path / "assistants"
    assistants.mkdir()
    (assistants / "a.md").write_text("---\nnom: A\n---\nTu es un assistant.", encoding="utf-8")
    r = Reglages(port=port_app, pause_s=0.8, dossier_assistants=assistants,
                 dossier_conversations=tmp_path / "conv", dossier_utilisateurs=tmp_path / "personnes")
    etat = Application(r, g, moteurs.MoteurVoixSimulation(), detecteur_energie)
    serveur = lancer_serveur(creer_app(etat), port_app)

    async def actions(ws, recus, attendre):
        await ws.send(json.dumps({"type": "texte", "contenu": "Bonjour"}))
        await attendre(lambda m: m.get("type") == "fin")
        await ws.send(json.dumps({"type": "voix", "active": True}))
        signal = np.concatenate([ton(1.5), np.zeros(24000, np.float32)])
        for i in range(0, len(signal), 1600):
            await ws.send(moteurs.float_vers_pcm16(signal[i:i + 1600]))
            await asyncio.sleep(0.1)
        await attendre(lambda m: m.get("type") == "utilisateur" and m.get("texte", "").startswith("Je voudrais"))

    try:
        asyncio.run(discuter(port_app, "conv-deux", "a", actions))
    finally:
        serveur.should_exit = faux_e.should_exit = faux_r.should_exit = True
        FauxServeur.adresses = {}
    ecrit = [c for c in journal_redaction if c.get("stream") and c.get("max_tokens") != 1]
    voix = [c for c in journal_ecoute if c.get("stream") and c.get("max_tokens") != 1]
    assert ecrit and ecrit[0]["messages"][-1]["content"] == "Bonjour"
    assert isinstance(voix[0]["messages"][-1]["content"], list)  # la phrase dite, en audio
    assert voix[-1]["messages"][-1]["role"] == "tool"  # la suite de la réponse vocale, après l'outil
    assert any("messages" in c and not c.get("stream") for c in journal_ecoute)  # retranscription
    assert not any(isinstance(c.get("messages", [{}])[-1].get("content"), list) for c in journal_redaction)


def test_profils_des_personnes(environnement):
    import urllib.request

    from assistant.utilisateurs import Utilisateurs

    env = environnement
    base = f"http://127.0.0.1:{env['port']}"

    def appel(methode, chemin, donnees=None):
        req = urllib.request.Request(base + chemin, method=methode,
                                     data=json.dumps(donnees).encode() if donnees is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    code, err = appel("POST", "/api/utilisateurs", {"prenom": " "})
    assert code == 400 and "prénom" in err["erreur"]
    _, sophie = appel("POST", "/api/utilisateurs", {
        "prenom": "Sophie", "role": "Comptable", "besoins": "Je suis dyslexique : phrases courtes.",
        "preferences": "Vouvoiement, réponses avec des listes."})
    _, marc = appel("POST", "/api/utilisateurs", {"prenom": "Marc"})
    assert sophie["id"].startswith("sophie-") and (env["tmp"] / "conversations").exists()

    async def actions(ws, recus, attendre):
        await ws.send(json.dumps({"type": "texte", "contenu": "Bonjour"}))
        await attendre(lambda m: m.get("type") == "fin")

    async def discuter_pour(qui, conv_id):
        import websockets

        async with websockets.connect(f"ws://127.0.0.1:{env['port']}/ws/{conv_id}?assistant=assistant-du-quotidien"
                                      f"&utilisateur={qui}") as ws:
            while json.loads(await ws.recv()).get("etat") != "pret":
                pass
            await ws.send(json.dumps({"type": "texte", "contenu": "Bonjour"}))
            while json.loads(await ws.recv()).get("type") != "fin":
                pass

    asyncio.run(discuter_pour(sophie["id"], "conv-sophie"))
    asyncio.run(discuter_pour(marc["id"], "conv-marc"))
    systeme = [c for c in env["journal"] if c.get("stream") and c.get("max_tokens") != 1][0]["messages"][0]["content"]
    assert "# La personne avec qui tu parles" in systeme and "Prénom : Sophie" in systeme
    assert "Mes besoins particuliers : Je suis dyslexique" in systeme and "Vouvoiement" in systeme
    # Chacun retrouve ses conversations (et pas celles des autres).
    _, liste_sophie = appel("GET", f"/api/conversations?utilisateur={sophie['id']}")
    _, liste_marc = appel("GET", f"/api/conversations?utilisateur={marc['id']}")
    assert [c["id"] for c in liste_sophie] == ["conv-sophie"] and [c["id"] for c in liste_marc] == ["conv-marc"]
    # Modifier puis supprimer un profil.
    _, sophie = appel("PUT", f"/api/utilisateurs/{sophie['id']}", {**sophie, "role": "Cheffe comptable"})
    assert Utilisateurs(env["tmp"] / "personnes").lire(sophie["id"]).role == "Cheffe comptable"
    code, _ = appel("DELETE", f"/api/utilisateurs/{marc['id']}")
    assert code == 200 and [u["prenom"] for u in appel("GET", "/api/utilisateurs")[1]] == ["Sophie"]
