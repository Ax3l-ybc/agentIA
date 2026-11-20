"""Les documents d'un assistant : un dossier de fichiers .md ou .txt, en LECTURE SEULE.

Les fichiers sont lus directement sur le disque, à chaque demande, sans copie. Aucune
fonction de ce module n'ouvre un fichier en écriture.
"""

from __future__ import annotations

import pathlib
import re
import unicodedata

EXTENSIONS = (".md", ".txt")
LONGUEUR_MAX = 12000


def simplifier(texte: str) -> str:
    """Minuscules, sans accents : pour comparer des noms ou des mots."""
    texte = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in texte if not unicodedata.combining(c)).lower()


def est_dans(enfant: pathlib.Path, parent: pathlib.Path) -> bool:
    enfant, parent = enfant.resolve(), parent.resolve()
    return enfant == parent or parent in enfant.parents


class Documents:
    def __init__(self, dossier: pathlib.Path):
        self.dossier = dossier.resolve()

    def lister(self) -> list[tuple[str, pathlib.Path]]:
        """(nom relatif sans extension, chemin) pour chaque document."""
        docs = []
        for chemin in sorted(self.dossier.rglob("*")):
            if chemin.suffix.lower() not in EXTENSIONS or not chemin.is_file():
                continue
            rel = chemin.relative_to(self.dossier)
            if any(part.startswith(".") for part in rel.parts):  # .obsidian, .trash…
                continue
            if not est_dans(chemin, self.dossier):  # lien qui sort du dossier
                continue
            docs.append((rel.with_suffix("").as_posix(), chemin))
        return docs

    def _texte(self, chemin: pathlib.Path) -> str:
        return chemin.read_text(encoding="utf-8-sig", errors="replace")

    def trouver(self, nom: str) -> tuple[str, pathlib.Path] | None:
        cherche = simplifier(nom.strip().removesuffix(".md").removesuffix(".txt"))
        docs = self.lister()
        for rel, chemin in docs:  # nom exact, puis nom du fichier, puis morceau du nom
            if simplifier(rel) == cherche:
                return rel, chemin
        for rel, chemin in docs:
            if simplifier(chemin.stem) == cherche:
                return rel, chemin
        for rel, chemin in docs:
            if cherche and cherche in simplifier(rel):
                return rel, chemin
        return None

    def lire(self, nom: str, longueur_max: int = LONGUEUR_MAX) -> dict:
        trouve = self.trouver(nom)
        if trouve is None:
            return {"erreur": f"Document introuvable : {nom}",
                    "documents": [rel for rel, _ in self.lister()][:50]}
        rel, chemin = trouve
        texte = self._texte(chemin)
        resultat = {"document": rel, "contenu": texte[:longueur_max]}
        if len(texte) > longueur_max:
            resultat["note"] = f"Document tronqué ({len(texte)} caractères au total)."
        return resultat

    def chercher(self, requete: str, max_resultats: int = 8) -> dict:
        mots = [m for m in re.findall(r"\w+", simplifier(requete)) if len(m) > 2]
        if not mots:
            return {"resultats": []}
        resultats = []
        for rel, chemin in self.lister():
            for ligne in self._texte(chemin).splitlines():
                simple = simplifier(ligne)
                score = sum(simple.count(m) for m in mots)
                if score:
                    resultats.append((score, rel, ligne.strip()[:300]))
        resultats.sort(key=lambda x: -x[0])
        return {"resultats": [{"document": rel, "extrait": ligne}
                              for _, rel, ligne in resultats[:max_resultats]]}

    def taille_totale(self) -> int:
        return sum(chemin.stat().st_size for _, chemin in self.lister())

    def tout_le_texte(self) -> str:
        return "\n\n".join(f"=== DOCUMENT : {rel} ===\n{self._texte(chemin)}" for rel, chemin in self.lister())
