"""Les assistants : un fichier par assistant dans le dossier « assistants ».

Chaque fichier se lit et se modifie avec le Bloc-notes (ou depuis la page web) :

    ---
    nom: Assistant du quotidien
    description: Un assistant polyvalent pour tous les jours.
    voix: Hélène
    dossier_documents: ''
    ---
    Tu es mon assistant personnel. (… les consignes, en texte libre …)

- `nom` et `description` s'affichent dans la page ;
- `voix` : une voix Phonon française (Hélène, Arnaud, Lucien… ; vide = voix par défaut) ;
- `dossier_documents` : facultatif. Un dossier de fichiers .md ou .txt que l'assistant peut
  consulter pour répondre. Il le lit sans jamais le modifier ;
- tout ce qui suit le second « --- » : les consignes de l'assistant (son rôle, son ton,
  ce qu'il doit faire ou éviter). Elles sont relues à chaque nouvelle conversation.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pathlib
import re
import shutil
import unicodedata

from .reglages import ErreurReglages, lire_yaml


@dataclasses.dataclass
class Profil:
    id: str  # nom du fichier, sans « .md »
    nom: str
    consignes: str
    description: str = ""
    voix: str = ""
    dossier_documents: str = ""

    def dossier(self) -> pathlib.Path | None:
        if not self.dossier_documents.strip():
            return None
        return pathlib.Path(os.path.expandvars(os.path.expanduser(self.dossier_documents.strip())))

    def en_dict(self) -> dict:
        return dataclasses.asdict(self)


def identifiant(nom: str) -> str:
    """Nom de fichier simple à partir du nom affiché : « Coach d'écriture » -> coach-d-ecriture."""
    texte = unicodedata.normalize("NFKD", nom)
    texte = "".join(c for c in texte if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "-", texte).strip("-")[:50] or "assistant"


def lire_fichier(chemin: pathlib.Path) -> Profil:
    texte = chemin.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    entete, consignes = {}, texte
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", texte, re.S)
    if m:
        entete = lire_yaml(m.group(1), chemin.name)
        consignes = m.group(2)
    return Profil(
        id=chemin.stem,
        nom=str(entete.get("nom") or chemin.stem).strip(),
        consignes=consignes.strip(),
        description=str(entete.get("description") or "").strip(),
        voix=str(entete.get("voix") or "").strip(),
        dossier_documents=str(entete.get("dossier_documents") or "").strip(),
    )


def _yaml_simple(valeur: str) -> str:
    return "'" + valeur.replace("'", "''").replace("\n", " ") + "'"


def texte_fichier(p: Profil) -> str:
    return (
        "---\n"
        f"nom: {_yaml_simple(p.nom)}\n"
        f"description: {_yaml_simple(p.description)}\n"
        f"voix: {_yaml_simple(p.voix)}\n"
        f"dossier_documents: {_yaml_simple(p.dossier_documents)}\n"
        "---\n"
        f"{p.consignes.strip()}\n"
    )


class Profils:
    def __init__(self, dossier: pathlib.Path):
        self.dossier = dossier
        self.dossier.mkdir(parents=True, exist_ok=True)

    def lister(self) -> list[Profil]:
        profils = []
        for chemin in sorted(self.dossier.glob("*.md")):
            try:
                profils.append(lire_fichier(chemin))
            except (OSError, ErreurReglages):
                continue  # fichier illisible : on l'ignore (il est signalé dans la page)
        return profils

    def lire(self, ident: str) -> Profil | None:
        if not re.fullmatch(r"[\w-]+", ident or ""):
            return None
        chemin = self.dossier / f"{ident}.md"
        return lire_fichier(chemin) if chemin.is_file() else None

    def par_defaut(self) -> Profil:
        profils = self.lister()
        if not profils:
            raise ErreurReglages(
                f"Aucun assistant dans {self.dossier} : remettez au moins un fichier .md "
                "(par exemple celui fourni avec le programme)."
            )
        return next((p for p in profils if p.id == "assistant-du-quotidien"), profils[0])

    def enregistrer(self, p: Profil) -> Profil:
        """Enregistre un assistant (nouveau si `p.id` est vide). Renvoie l'assistant enregistré."""
        if not p.nom.strip():
            raise ValueError("Donnez un nom à l'assistant.")
        if not p.consignes.strip():
            raise ValueError("Écrivez quelques consignes pour l'assistant.")
        if p.dossier() is not None and not p.dossier().is_dir():
            raise ValueError(f"Le dossier de documents n'existe pas : {p.dossier_documents}")
        if not p.id:
            base = identifiant(p.nom)
            ident, n = base, 2
            while (self.dossier / f"{ident}.md").exists():
                ident, n = f"{base}-{n}", n + 1
            p.id = ident
        if not re.fullmatch(r"[\w-]+", p.id):
            raise ValueError("Identifiant d'assistant invalide.")
        chemin = self.dossier / f"{p.id}.md"
        chemin.write_text(texte_fichier(p), encoding="utf-8")
        return lire_fichier(chemin)

    def supprimer(self, ident: str) -> None:
        """Range le fichier dans « assistants/corbeille » (rien n'est vraiment effacé)."""
        chemin = self.dossier / f"{ident}.md"
        if not re.fullmatch(r"[\w-]+", ident or "") or not chemin.is_file():
            raise ValueError("Assistant introuvable.")
        if len(self.lister()) <= 1:
            raise ValueError("Gardez au moins un assistant.")
        corbeille = self.dossier / "corbeille"
        corbeille.mkdir(exist_ok=True)
        horodatage = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M%S")
        shutil.move(str(chemin), str(corbeille / f"{ident}_{horodatage}.md"))
