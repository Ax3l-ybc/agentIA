"""Accès au coffre Obsidian.

Règle d'or : le dossier des notes est en LECTURE SEULE.
- Les notes sont lues directement sur le disque, à chaque demande, sans copie.
- Aucune fonction de ce module n'ouvre un fichier du dossier des notes en écriture.
- La seule écriture possible est la création d'une NOUVELLE fiche dans le dossier
  des recaps, qui est vérifié comme étant en dehors du dossier des notes.
  Un fichier existant n'est jamais écrasé.
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import re
import unicodedata

from .reglages import est_dans

LONGUEUR_MAX_NOTE = 12000


def simplifier(texte: str) -> str:
    """Minuscules, sans accents : pour comparer des noms ou des mots."""
    texte = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in texte if not unicodedata.combining(c)).lower()


@dataclasses.dataclass
class Note:
    chemin: pathlib.Path
    relatif: str  # chemin relatif au dossier des notes, sans « .md »

    @property
    def titre(self) -> str:
        return self.chemin.stem


class Coffre:
    def __init__(self, dossier_notes: pathlib.Path, dossier_recaps: pathlib.Path):
        self.notes_dir = dossier_notes.resolve()
        self.recaps_dir = dossier_recaps.resolve()
        if est_dans(self.recaps_dir, self.notes_dir):
            raise ValueError("Le dossier des recaps ne peut pas être dans le dossier des notes.")

    # ------------------------------------------------------------------ lecture

    def lister(self) -> list[Note]:
        notes = []
        for chemin in sorted(self.notes_dir.rglob("*.md")):
            rel = chemin.relative_to(self.notes_dir)
            # On ignore les dossiers cachés (.obsidian, .trash, ...).
            if any(part.startswith(".") for part in rel.parts):
                continue
            if not est_dans(chemin, self.notes_dir):  # lien symbolique sortant
                continue
            notes.append(Note(chemin=chemin, relatif=rel.with_suffix("").as_posix()))
        return notes

    def lire_texte(self, note: Note) -> str:
        # Ouverture en lecture seule ("r").
        with open(note.chemin, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def trouver(self, nom: str) -> Note | None:
        """Retrouve une note à partir d'un nom approximatif (dicté à l'oral)."""
        cible = simplifier(nom.strip().removesuffix(".md").strip("[] "))
        if not cible:
            return None
        notes = self.lister()
        for critere in (
            lambda n: simplifier(n.relatif) == cible,
            lambda n: simplifier(n.titre) == cible,
            lambda n: cible in simplifier(n.titre),
            lambda n: cible in simplifier(n.relatif),
        ):
            trouvees = [n for n in notes if critere(n)]
            if trouvees:
                return min(trouvees, key=lambda n: len(n.relatif))
        # Dernier recours : la note qui partage le plus de mots avec la demande.
        mots = set(re.findall(r"\w{3,}", cible))
        if not mots:
            return None
        meilleure, score_max = None, 0
        for n in notes:
            score = len(mots & set(re.findall(r"\w{3,}", simplifier(n.relatif))))
            if score > score_max:
                meilleure, score_max = n, score
        return meilleure

    def lire(self, nom: str, longueur_max: int = LONGUEUR_MAX_NOTE) -> dict:
        note = self.trouver(nom)
        if note is None:
            return {
                "erreur": f"Aucune note ne correspond à « {nom} ».",
                "notes_disponibles": [n.relatif for n in self.lister()][:60],
            }
        texte = self.lire_texte(note)
        resultat = {"note": note.relatif, "contenu": texte[:longueur_max]}
        if len(texte) > longueur_max:
            resultat["remarque"] = (
                f"Note tronquée : seuls les {longueur_max} premiers caractères "
                f"sur {len(texte)} sont fournis."
            )
        return resultat

    def index(self) -> list[dict]:
        """Liste des notes avec leurs intertitres, pour donner une vue d'ensemble."""
        resultat = []
        for n in self.lister():
            texte = self.lire_texte(n)
            titres = re.findall(r"^#{1,3}\s+(.+)$", texte, flags=re.M)[:12]
            resultat.append(
                {"note": n.relatif, "mots": len(texte.split()), "intertitres": titres}
            )
        return resultat

    def chercher(self, requete: str, max_resultats: int = 8) -> dict:
        mots = [m for m in re.findall(r"\w+", simplifier(requete)) if len(m) >= 3]
        if not mots:
            return {"erreur": "Requête trop courte."}
        trouves = []
        for n in self.lister():
            lignes = self.lire_texte(n).splitlines()
            for i, ligne in enumerate(lignes):
                ligne_s = simplifier(ligne)
                score = sum(1 for m in mots if m in ligne_s)
                if score:
                    extrait = " ".join(l.strip() for l in lignes[max(0, i - 1) : i + 2])
                    trouves.append((score, n.relatif, extrait[:400]))
        trouves.sort(key=lambda t: -t[0])
        return {
            "requete": requete,
            "resultats": [{"note": r, "extrait": e} for _, r, e in trouves[:max_resultats]],
        }

    def tout_le_texte(self) -> str:
        morceaux = []
        for n in self.lister():
            morceaux.append(f"=== NOTE : {n.relatif} ===\n{self.lire_texte(n).strip()}\n")
        return "\n".join(morceaux)

    def taille_totale(self) -> int:
        return sum(len(self.lire_texte(n)) for n in self.lister())

    # ------------------------------------------------------- fiches de séance

    def lister_fiches(self) -> list[pathlib.Path]:
        """Fiches des séances précédentes, de la plus récente à la plus ancienne."""
        if not self.recaps_dir.is_dir():
            return []
        fiches = [p for p in self.recaps_dir.glob("*.md") if p.is_file()]
        # Les noms commencent par la date (AAAA-MM-JJ_...) ; à défaut, date de modification.
        return sorted(fiches, key=lambda p: (p.name[:10], p.stat().st_mtime), reverse=True)

    def fiches_recentes(self, nombre: int = 3, longueur_max: int = 5000) -> list[dict]:
        resultat = []
        for p in self.lister_fiches()[:nombre]:
            texte = p.read_text(encoding="utf-8", errors="replace")
            resultat.append({"nom": p.stem, "contenu": texte[:longueur_max]})
        return resultat

    def lire_fiche(self, nom: str) -> dict:
        fiches = self.lister_fiches()
        cible = simplifier(nom.strip().removesuffix(".md"))
        for p in fiches:
            if cible and (cible == simplifier(p.stem) or cible in simplifier(p.stem)):
                return {"fiche": p.stem, "contenu": p.read_text(encoding="utf-8", errors="replace")[:LONGUEUR_MAX_NOTE]}
        return {"erreur": f"Aucune fiche ne correspond à « {nom} ».", "fiches": [p.stem for p in fiches][:40]}

    # ----------------------------------------------------------------- écriture

    def ecrire_recap(
        self, contenu: str, quand: datetime.datetime | None = None, nom_fichier: str | None = None
    ) -> pathlib.Path:
        """Crée une NOUVELLE fiche dans le dossier des recaps. N'écrase jamais rien."""
        quand = quand or datetime.datetime.now()
        self.recaps_dir.mkdir(parents=True, exist_ok=True)
        base = pathlib.Path(nom_fichier).name.removesuffix(".md") if nom_fichier else f"Recap {quand:%Y-%m-%d %Hh%M}"
        base = base.strip(" .") or f"Recap {quand:%Y-%m-%d %Hh%M}"
        for i in range(1, 1000):
            nom = base if i == 1 else f"{base} ({i})"
            cible = (self.recaps_dir / f"{nom}.md").resolve()
            # Double sécurité : jamais dans le dossier des notes.
            if est_dans(cible, self.notes_dir) or not est_dans(cible, self.recaps_dir):
                raise PermissionError("Écriture refusée : cible hors du dossier des recaps.")
            try:
                with open(cible, "x", encoding="utf-8") as f:  # "x" = création seulement
                    f.write(contenu)
                return cible
            except FileExistsError:
                continue
        raise RuntimeError("Impossible de trouver un nom de fichier libre pour la fiche.")
