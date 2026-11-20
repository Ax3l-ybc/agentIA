"""Les profils des personnes qui utilisent l'assistant sur ce PC.

Chacun décrit qui il est, ses besoins particuliers et ses préférences : tous les assistants en
tiennent compte. Chaque personne retrouve ses propres conversations.

Les profils sont rangés dans %USERPROFILE%\\MonAssistant\\utilisateurs (un fichier par
personne). Ils ne sont pas protégés par un mot de passe : toute personne qui utilise ce PC
peut les voir.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import secrets

from .profils import identifiant

CHAMPS = {
    "role": "Métier ou rôle",
    "presentation": "Qui je suis",
    "besoins": "Mes besoins particuliers",
    "preferences": "Mes préférences",
    "autres": "À savoir aussi",
}


@dataclasses.dataclass
class Utilisateur:
    id: str
    prenom: str
    role: str = ""
    presentation: str = ""
    besoins: str = ""
    preferences: str = ""
    autres: str = ""

    def en_dict(self) -> dict:
        return dataclasses.asdict(self)


def bloc_consignes(u: Utilisateur | None) -> str:
    """Ce que les assistants savent de la personne (ajouté à leurs consignes)."""
    if u is None:
        return ""
    lignes = [f"Prénom : {u.prenom}"]
    for cle, titre in CHAMPS.items():
        valeur = getattr(u, cle).strip()
        if valeur:
            lignes.append(f"{titre} : {valeur}")
    return ("# La personne avec qui tu parles (elle l'a écrit elle-même)\n" + "\n".join(lignes) + "\n\n"
            "RÈGLE PRIORITAIRE : respecte ses besoins particuliers et ses préférences dans chaque "
            "réponse, même s'ils contredisent le ton indiqué plus haut (par exemple, si elle préfère "
            "le vouvoiement, vouvoie-la toujours, dès le premier mot, même si les consignes disent de "
            "tutoyer). Appelle-la par son prénom quand c'est naturel.")


class Utilisateurs:
    def __init__(self, dossier: pathlib.Path):
        self.dossier = dossier
        self.dossier.mkdir(parents=True, exist_ok=True)

    def _chemin(self, ident: str) -> pathlib.Path:
        if not re.fullmatch(r"[\w-]+", ident or ""):
            raise ValueError("Identifiant de personne invalide.")
        return self.dossier / f"{ident}.json"

    def lister(self) -> list[Utilisateur]:
        personnes = []
        for chemin in sorted(self.dossier.glob("*.json")):
            try:
                personnes.append(self._lire_fichier(chemin))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(personnes, key=lambda u: u.prenom.lower())

    def _lire_fichier(self, chemin: pathlib.Path) -> Utilisateur:
        d = json.loads(chemin.read_text(encoding="utf-8"))
        champs = {f.name for f in dataclasses.fields(Utilisateur)}
        return Utilisateur(**{k: str(v) for k, v in d.items() if k in champs} | {"id": chemin.stem})

    def lire(self, ident: str) -> Utilisateur | None:
        try:
            chemin = self._chemin(ident)
        except ValueError:
            return None
        return self._lire_fichier(chemin) if chemin.is_file() else None

    def enregistrer(self, u: Utilisateur) -> Utilisateur:
        """Enregistre un profil (nouveau si `u.id` est vide)."""
        u.prenom = u.prenom.strip()
        if not u.prenom:
            raise ValueError("Indiquez au moins votre prénom.")
        for cle in CHAMPS:
            setattr(u, cle, getattr(u, cle).strip()[:4000])
        if not u.id:
            u.id = f"{identifiant(u.prenom)}-{secrets.token_hex(2)}"
        self._chemin(u.id).write_text(json.dumps(u.en_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return u

    def supprimer(self, ident: str) -> None:
        chemin = self._chemin(ident)
        if not chemin.is_file():
            raise ValueError("Profil introuvable.")
        chemin.unlink()
