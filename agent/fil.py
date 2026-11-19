"""Aides pour une conversation orale fluide avec un petit modèle (Gemma 4 E2B).

Un petit modèle oublie vite ce qu'il a déjà demandé et prend des tics de langage. Le
programme tient donc le fil à sa place :
- il rappelle au modèle ses dernières questions, pour qu'il ne les repose pas ;
- s'il en repose une quand même, il lui demande au tour suivant de passer à autre chose ;
- il propose à chaque tour deux façons de commencer la réponse (une neutre, une plus
  appréciative) parmi celles qui n'ont pas servi récemment, et retire un « Ah, » qui
  vient d'être dit.
Ces rappels accompagnent seulement le message en cours : ils ne sont pas gardés dans
l'historique, et ne gênent donc pas la mémoire de llama.cpp.
"""

from __future__ import annotations

import random
import re

REACTIONS_NEUTRES = ["D'accord", "Je vois", "Entendu", "Je note", "Bien", "Très bien", "Soit",
                     "Je comprends", "Oui", "Voyons", "Admettons", "Hmm"]
REACTIONS_APPRECIATIVES = ["Intéressant", "Bien vu", "Bon point", "C'est clair", "Excellent",
                           "Bonne remarque", "Pertinent", "Ça se tient", "Juste", "Parfait",
                           "Bonne idée", "C'est précis"]

PASSER_A_AUTRE_CHOSE = ("(Tu as déjà posé cette question. Accepte sa réponse telle quelle, même "
                        "partielle, et passe à un autre point sans revenir dessus.)")

_TIC = re.compile(r"^\s*(ah|oh|hmm|bien|intéressant|parfait|excellent)\s*[,!.…]+\s*", re.I)
_MOTS_VIDES = set("""alors aussi avec avez avoir avais cette comme dans donc elle elles est était
être fait faire leur mais même moins nous peux peut plus pour quel quelle quelles quels quoi sans
sont sous suis tout toute tous très vous votre veux""".split())


def mot_d_ouverture(texte: str) -> str:
    m = re.match(r"\W*([\w'’]+)", texte)
    return m.group(1).lower() if m else ""


def proposer_reactions(recentes: list[str], hasard: random.Random = random) -> tuple[str, str]:
    """Une réaction neutre et une appréciative dont le premier mot n'a pas servi récemment."""
    def choisir(panel):
        libres = [x for x in panel if mot_d_ouverture(x) not in recentes]
        return hasard.choice(libres or panel)
    return choisir(REACTIONS_NEUTRES), choisir(REACTIONS_APPRECIATIVES)


def retirer_tic(texte: str, recentes: list[str]) -> str:
    """Retire un « Ah, » (ou « Intéressant ! »…) en début de réponse s'il vient d'être dit."""
    m = _TIC.match(texte)
    if not m or m.group(1).lower() not in recentes:
        return texte
    reste = texte[m.end():]
    return reste[:1].upper() + reste[1:] if reste else texte


def mots_cles(question: str) -> set[str]:
    return {m for m in re.findall(r"\w+", question.lower()) if len(m) > 3 and m not in _MOTS_VIDES}


def derniere_question(texte: str) -> str:
    """La dernière phrase interrogative d'une réponse (ou "")."""
    phrases = re.findall(r"[^.!?…]*\?", texte)
    return phrases[-1].strip() if phrases else ""


def deja_posee(question: str, precedentes: list[str], seuil: float = 0.5) -> bool:
    """Vrai si `question` ressemble à une question déjà posée (mêmes mots importants)."""
    nouveaux = mots_cles(question)
    if len(nouveaux) < 2:
        return False
    for ancienne in precedentes:
        anciens = mots_cles(ancienne)
        if anciens and len(nouveaux & anciens) / len(nouveaux | anciens) >= seuil:
            return True
    return False


class Fil:
    """Ce que le programme retient de la séance pour guider le modèle."""

    def __init__(self):
        self.questions: list[str] = []
        self.ouvertures: list[str] = []
        self.repetition = False

    def indication(self) -> str:
        """Texte ajouté devant la consigne du message vocal en cours."""
        texte = ""
        if self.repetition:
            texte += PASSER_A_AUTRE_CHOSE + " "
        if self.questions:
            dernieres = " / ".join(f"« {q} »" for q in self.questions[-2:])
            texte += f"(Tes questions précédentes : {dernieres}. Ta prochaine question doit être nouvelle.) "
        neutre, appreciative = proposer_reactions(self.ouvertures[-5:])
        texte += (f"(Commence par « {neutre}, » ou « {appreciative}, » selon ce qu'il vient de dire, "
                  "puis réagis en une phrase et pose-lui une question précise.) ")
        return texte

    def nettoyer_debut(self, texte: str) -> str:
        return retirer_tic(texte, self.ouvertures[-3:])

    def noter(self, reponse: str, vocal: bool) -> bool:
        """Retient la réponse du modèle. Renvoie vrai s'il vient de reposer une question."""
        if vocal:
            self.repetition = False  # le rappel « passe à autre chose » a été transmis
        if reponse:
            self.ouvertures.append(mot_d_ouverture(reponse))
        question = derniere_question(reponse)
        if not question:
            return False
        repetee = deja_posee(question, self.questions)
        self.repetition = self.repetition or repetee
        self.questions.append(question)
        return repetee
