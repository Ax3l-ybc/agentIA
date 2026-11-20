"""llama.cpp pour la carte graphique (Windows).

La version installée par winget tourne sur le processeur. Pour une carte graphique, on
télécharge la version de llama.cpp faite pour elle, depuis les publications officielles :
- NVIDIA : la version CUDA (avec les bibliothèques CUDA, sans rien installer d'autre) ;
- AMD Radeon RX, Intel Arc : la version Vulkan.
Elle est rangée dans %USERPROFILE%\\MonAssistant\\llama.cpp-cuda (ou -vulkan).
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import shutil
import urllib.request
import zipfile

from .reglages import DOSSIER_DONNEES

PUBLICATION = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def dossier(genre: str) -> pathlib.Path:
    return DOSSIER_DONNEES / f"llama.cpp-{'cuda' if genre == 'nvidia' else 'vulkan'}"


def serveur_installe(genre: str | None = None) -> pathlib.Path | None:
    """llama-server.exe pour la carte graphique, s'il a été installé."""
    for g in ([genre] if genre else ["nvidia", "vulkan"]):
        for nom in ("llama-server.exe", "llama-server"):
            trouves = sorted(dossier(g).rglob(nom)) if dossier(g).is_dir() else []
            if trouves:
                return trouves[0]
    return None


def choisir_fichiers(noms: list[str], genre: str) -> list[str]:
    """Les fichiers de la publication à télécharger pour cette carte."""
    if genre == "nvidia":
        # CUDA 12 marche avec presque tous les pilotes récents ; CUDA 13 en demande de très récents.
        versions = {}
        for n in noms:
            m = re.fullmatch(r"llama-.*-bin-win-cuda-(\d+)\.(\d+)-x64\.zip", n)
            if m:
                versions[(int(m.group(1)), int(m.group(2)))] = n
        if not versions:
            raise RuntimeError("pas de version CUDA pour Windows dans la dernière publication")
        preferees = [v for v in versions if v[0] == 12] or list(versions)
        version = max(preferees)
        choisis = [versions[version]]
        cudart = f"cudart-llama-bin-win-cuda-{version[0]}.{version[1]}-x64.zip"
        if cudart in noms:
            choisis.append(cudart)
        return choisis
    vulkan = [n for n in noms if re.fullmatch(r"llama-.*-bin-win-vulkan-x64\.zip", n)]
    if not vulkan:
        raise RuntimeError("pas de version Vulkan pour Windows dans la dernière publication")
    return vulkan[:1]


def installer(genre: str, afficher=print) -> pathlib.Path:
    """Télécharge et décompresse llama.cpp pour la carte. Renvoie le chemin de llama-server."""
    with urllib.request.urlopen(urllib.request.Request(PUBLICATION, headers={"User-Agent": "MonAssistant"}),
                                timeout=30) as r:
        publication = json.load(r)
    adresses = {a["name"]: a["browser_download_url"] for a in publication.get("assets", [])}
    cible = dossier(genre)
    if cible.exists():
        shutil.rmtree(cible)
    cible.mkdir(parents=True)
    for nom in choisir_fichiers(list(adresses), genre):
        afficher(f"  {nom}…")
        with urllib.request.urlopen(urllib.request.Request(adresses[nom], headers={"User-Agent": "MonAssistant"}),
                                    timeout=120) as r:
            donnees = r.read()
        with zipfile.ZipFile(io.BytesIO(donnees)) as z:
            z.extractall(cible)
    (cible / "version.txt").write_text(publication.get("tag_name", ""), encoding="utf-8")
    serveur = serveur_installe(genre)
    if serveur is None:
        raise RuntimeError("llama-server.exe introuvable dans le fichier téléchargé")
    # Les bibliothèques CUDA (cudart) doivent être à côté de llama-server.exe.
    for dll in cible.glob("*.dll"):
        if dll.parent != serveur.parent and not (serveur.parent / dll.name).exists():
            shutil.copy2(dll, serveur.parent / dll.name)
    return serveur
