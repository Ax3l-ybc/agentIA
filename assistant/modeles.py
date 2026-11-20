"""Les cerveaux possibles (modèles Gemma 4 pour llama.cpp), leur téléchargement et le choix
conseillé selon le PC.

Deux rôles :
- l'**écoute** : le modèle qui converse, et qui écoute directement la voix (E2B ou E4B) ;
- la **rédaction** (facultatif) : un modèle plus grand, qui rédige les réponses écrites, les
  comptes rendus et le travail sur les documents. Il n'écoute pas la voix : la voix et la
  transcription des réunions restent au modèle d'écoute.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import time
import urllib.request

from .reglages import DOSSIER_DONNEES, DOSSIERS_MODELES

log = logging.getLogger("assistant.modeles")


@dataclasses.dataclass(frozen=True)
class Modele:
    id: str
    nom: str
    depot: str  # dépôt Hugging Face
    fichier: str
    mmproj: str | None  # module audio (modèles d'écoute seulement)
    ecoute: bool
    taille_go: float  # téléchargement total
    ram_go: float  # mémoire vive conseillée pour le faire tourner sur le processeur (PC entier)
    vram_go: float  # mémoire de carte graphique pour le faire tourner entièrement dessus
    description: str

    def chemin(self) -> pathlib.Path:
        return trouver(self.fichier)

    def chemin_mmproj(self) -> pathlib.Path | None:
        if not self.mmproj:
            return None
        # Les modules audio s'appellent tous « mmproj-F16.gguf » : celui d'E2B est rangé à côté
        # du modèle (comme avant), les autres dans un sous-dossier à leur nom.
        nom = self.mmproj if self.id == "e2b" else f"{self.id}/{self.mmproj}"
        return trouver(nom)

    def installe(self) -> bool:
        fichiers = [self.chemin()] + ([self.chemin_mmproj()] if self.mmproj else [])
        return all(f.is_file() and f.stat().st_size > 10_000_000 for f in fichiers)

    def en_dict(self) -> dict:
        return {**dataclasses.asdict(self), "installe": self.installe()}


CATALOGUE = [
    Modele("e2b", "Gemma 4 E2B", "unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf",
           "mmproj-F16.gguf", True, 4.1, 8, 5,
           "Rapide, écoute la voix. Le bon choix sur un PC sans carte graphique."),
    Modele("e4b", "Gemma 4 E4B", "unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf",
           "mmproj-F16.gguf", True, 6.0, 12, 7,
           "Plus fin, écoute la voix. Sur un simple processeur, environ deux fois plus lent qu'E2B."),
    Modele("12b", "Gemma 4 12B", "unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf",
           None, False, 7.1, 20, 9,
           "Rédaction nettement plus fine. N'écoute pas la voix : il rédige, E2B ou E4B écoute."),
    Modele("26b", "Gemma 4 26B-A4B", "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
           None, False, 17.0, 32, 20,
           "Très bonne rédaction, et rapide pour sa taille (seule une petite partie travaille à "
           "chaque mot). Demande beaucoup de mémoire."),
    Modele("31b", "Gemma 4 31B", "unsloth/gemma-4-31B-it-GGUF", "gemma-4-31B-it-Q4_K_M.gguf",
           None, False, 18.3, 48, 24,
           "Le plus fin. Pour un PC avec une grosse carte graphique (24 Go)."),
]
PAR_ID = {m.id: m for m in CATALOGUE}


def trouver(nom: str) -> pathlib.Path:
    """Le fichier dans le premier dossier de modèles qui le contient (sinon le premier dossier)."""
    for dossier in DOSSIERS_MODELES:
        if (dossier / nom).is_file():
            return dossier / nom
    return DOSSIERS_MODELES[0] / nom


# ---------------------------------------------------------------------------
# Le choix enregistré (fait dans la page, panneau « Cerveau »)
# ---------------------------------------------------------------------------

FICHIER_CHOIX = DOSSIER_DONNEES / "cerveau.json"


@dataclasses.dataclass
class Choix:
    ecoute: str = "e2b"
    redacteur: str | None = None  # None : le modèle d'écoute rédige aussi
    carte_graphique: bool = False

    def valider(self) -> None:
        if self.ecoute not in PAR_ID or not PAR_ID[self.ecoute].ecoute:
            raise ValueError("Le modèle d'écoute doit être Gemma 4 E2B ou E4B.")
        if self.redacteur is not None and (self.redacteur not in PAR_ID or PAR_ID[self.redacteur].ecoute):
            raise ValueError("Modèle de rédaction inconnu.")


def lire_choix(fichier: pathlib.Path = FICHIER_CHOIX) -> Choix:
    try:
        d = json.loads(fichier.read_text(encoding="utf-8"))
        choix = Choix(ecoute=d.get("ecoute") or "e2b", redacteur=d.get("redacteur") or None,
                      carte_graphique=bool(d.get("carte_graphique")))
        choix.valider()
        return choix
    except (OSError, ValueError):
        return Choix()


def ecrire_choix(choix: Choix, fichier: pathlib.Path = FICHIER_CHOIX) -> None:
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(json.dumps(dataclasses.asdict(choix), indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Conseils selon le PC
# ---------------------------------------------------------------------------

def adequation(m: Modele, ram_go: float, vram_go: float, autre: Modele | None = None) -> str:
    """« carte » (tient sur la carte graphique), « processeur » (tient en mémoire vive) ou
    « trop_gros ». `autre` : l'autre modèle chargé en même temps (il prend de la place)."""
    vram_libre = vram_go - (autre.vram_go if autre else 0)
    ram_libre = ram_go - (autre.taille_go if autre else 0)
    if vram_go and m.vram_go <= vram_libre:
        return "carte"
    if m.ram_go <= ram_libre + 2:  # quelques Go de marge suffisent si l'autre modèle est petit
        return "processeur"
    return "trop_gros"


def conseil(ram_go: float, vram_go: float) -> Choix:
    """Le choix conseillé pour ce PC (affiché dans la page, jamais imposé)."""
    e2b, e4b = PAR_ID["e2b"], PAR_ID["e4b"]
    if vram_go >= e4b.vram_go or ram_go >= 32:
        ecoute = e4b
    else:
        ecoute = e2b
    redacteur = None
    for m in (PAR_ID["31b"], PAR_ID["26b"], PAR_ID["12b"]):
        place = adequation(m, ram_go, vram_go, ecoute)
        # Sur un simple processeur, seul 26B-A4B reste assez rapide (peu de calcul par mot).
        if place == "carte" or (place == "processeur" and m.id == "26b" and ram_go >= 40):
            redacteur = m.id
            break
    return Choix(ecoute=ecoute.id, redacteur=redacteur, carte_graphique=vram_go >= e2b.vram_go)


# ---------------------------------------------------------------------------
# Téléchargement (avec reprise et progression)
# ---------------------------------------------------------------------------

def url(depot: str, fichier: str) -> str:
    return f"https://huggingface.co/{depot}/resolve/main/{fichier}"


def telecharger_fichier(adresse: str, cible: pathlib.Path, progression=None, arret=None) -> None:
    """Télécharge `adresse` dans `cible`. Un téléchargement interrompu reprend où il s'était
    arrêté. `progression(reçus, total)` est appelée régulièrement."""
    cible.parent.mkdir(parents=True, exist_ok=True)
    partiel = cible.with_name(cible.name + ".partiel")
    deja = partiel.stat().st_size if partiel.exists() else 0
    entetes = {"User-Agent": "MonAssistant"}
    jeton = os.environ.get("HF_TOKEN")
    if jeton and "huggingface.co" in adresse:
        entetes["Authorization"] = f"Bearer {jeton}"
    if deja:
        entetes["Range"] = f"bytes={deja}-"
    with urllib.request.urlopen(urllib.request.Request(adresse, headers=entetes), timeout=60) as r:
        if deja and r.status != 206:  # le serveur ne reprend pas : on recommence
            deja = 0
        total = deja + int(r.headers.get("Content-Length") or 0)
        recus, dernier = deja, 0.0
        with open(partiel, "ab" if deja else "wb") as f:
            while True:
                if arret is not None and arret.is_set():
                    raise InterruptedError("téléchargement interrompu")
                bloc = r.read(1 << 20)
                if not bloc:
                    break
                f.write(bloc)
                recus += len(bloc)
                if progression and time.monotonic() - dernier > 0.5:
                    progression(recus, total)
                    dernier = time.monotonic()
    if total and recus < total:
        raise OSError(f"téléchargement incomplet ({recus} octets sur {total})")
    partiel.replace(cible)
    if progression:
        progression(recus, total)


def a_telecharger(m: Modele) -> list[tuple[str, pathlib.Path]]:
    """Les fichiers manquants d'un modèle : [(adresse, destination)]."""
    manquants = []
    if not (m.chemin().is_file() and m.chemin().stat().st_size > 10_000_000):
        manquants.append((url(m.depot, m.fichier), DOSSIERS_MODELES[0] / m.fichier))
    if m.mmproj:
        mm = m.chemin_mmproj()
        if not (mm.is_file() and mm.stat().st_size > 10_000_000):
            nom = m.mmproj if m.id == "e2b" else f"{m.id}/{m.mmproj}"
            manquants.append((url(m.depot, m.mmproj), DOSSIERS_MODELES[0] / nom))
    return manquants
