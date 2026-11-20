"""Les conversations, gardées sur l'ordinateur (un fichier .json par conversation).

Elles sont rangées dans %USERPROFILE%\\MonAssistant\\conversations : on peut les reprendre
plus tard, les renommer ou les supprimer depuis la page web.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import re
import secrets


def maintenant() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclasses.dataclass
class Conversation:
    id: str
    assistant: str  # identifiant de l'assistant (son fichier)
    utilisateur: str = ""  # la personne à qui appartient la conversation ("" : à tout le monde)
    titre: str = "Nouvelle conversation"
    cree: str = dataclasses.field(default_factory=maintenant)
    modifie: str = dataclasses.field(default_factory=maintenant)
    # {"role": "utilisateur" | "assistant", "texte": …, "mode": "texte" | "voix", "date": …}
    messages: list[dict] = dataclasses.field(default_factory=list)

    def ajouter(self, role: str, texte: str, mode: str = "texte") -> dict:
        message = {"role": role, "texte": texte, "mode": mode, "date": maintenant()}
        self.messages.append(message)
        self.modifie = message["date"]
        if role == "utilisateur" and self.titre == "Nouvelle conversation" and texte.strip():
            self.titre = titre_depuis(texte)
        return message


def titre_depuis(texte: str, longueur: int = 48) -> str:
    texte = re.sub(r"\s+", " ", texte).strip()
    if len(texte) <= longueur:
        return texte
    coupe = texte[:longueur].rsplit(" ", 1)[0]
    return coupe + "…"


class Conversations:
    def __init__(self, dossier: pathlib.Path):
        self.dossier = dossier
        self.dossier.mkdir(parents=True, exist_ok=True)

    def _chemin(self, ident: str) -> pathlib.Path:
        if not re.fullmatch(r"[\w-]+", ident):
            raise ValueError("Identifiant de conversation invalide.")
        return self.dossier / f"{ident}.json"

    def creer(self, assistant: str, utilisateur: str = "") -> Conversation:
        ident = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        return Conversation(id=ident, assistant=assistant, utilisateur=utilisateur)

    def charger(self, ident: str) -> Conversation | None:
        chemin = self._chemin(ident)
        if not chemin.is_file():
            return None
        donnees = json.loads(chemin.read_text(encoding="utf-8"))
        champs = {f.name for f in dataclasses.fields(Conversation)}
        return Conversation(**{k: v for k, v in donnees.items() if k in champs})

    def enregistrer(self, conv: Conversation) -> None:
        if not conv.messages:
            return  # une conversation vide n'est pas gardée
        chemin = self._chemin(conv.id)
        temporaire = chemin.with_suffix(".tmp")
        temporaire.write_text(json.dumps(dataclasses.asdict(conv), ensure_ascii=False, indent=1),
                              encoding="utf-8")
        temporaire.replace(chemin)

    def lister(self, utilisateur: str | None = None) -> list[dict]:
        """Les conversations (d'une personne, avec celles qui n'appartiennent à personne)."""
        resume = []
        for chemin in self.dossier.glob("*.json"):
            try:
                d = json.loads(chemin.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if utilisateur is not None and d.get("utilisateur") not in ("", None, utilisateur):
                continue
            resume.append({k: d.get(k) for k in ("id", "titre", "assistant", "modifie", "utilisateur")})
        return sorted(resume, key=lambda c: c.get("modifie") or "", reverse=True)

    def renommer(self, ident: str, titre: str) -> None:
        conv = self.charger(ident)
        if conv is None:
            raise ValueError("Conversation introuvable.")
        conv.titre = titre.strip()[:100] or conv.titre
        self.enregistrer(conv)

    def supprimer(self, ident: str) -> None:
        self._chemin(ident).unlink(missing_ok=True)
