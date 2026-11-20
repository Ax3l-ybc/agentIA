"""Préparation du texte lu à voix haute : découpe en phrases et nettoyage."""

from __future__ import annotations

import re

_FIN_PHRASE = re.compile(r"([.!?…;:])(\s+)")
_A_NETTOYER = re.compile(r"[*_#`>|~]|\[\[|\]\]|https?://\S+")
_EMOJIS = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


def nettoyer_pour_voix(texte: str) -> str:
    """Retire ce qui ne se prononce pas (Markdown, émojis)."""
    texte = _EMOJIS.sub("", _A_NETTOYER.sub(" ", texte))
    return re.sub(r"\s+", " ", texte).strip()


def couper_phrases(tampon: str, premiere: bool) -> tuple[list[str], str]:
    """Sépare les phrases complètes du reste du texte en attente (un retour à la ligne termine
    aussi une phrase : titres et listes d'une réponse écrite)."""
    phrases = []
    while "\n" in tampon:
        ligne, tampon = tampon.split("\n", 1)
        if ligne.strip():
            sous, reste = couper_phrases(ligne + " ", premiere and not phrases)
            phrases += sous + ([reste] if reste.strip() else [])
    while True:
        m = _FIN_PHRASE.search(tampon)
        if not m:
            break
        phrases.append(tampon[: m.end(1)])
        tampon = tampon[m.end() :]
    # Phrase très longue sans ponctuation forte : on coupe à une virgule.
    seuil = 60 if premiere and not phrases else 220
    if len(tampon) > seuil and "," in tampon[20:]:
        i = tampon.rindex(",")
        phrases.append(tampon[: i + 1])
        tampon = tampon[i + 1 :].lstrip()
    return phrases, tampon


# Longueur maximale (en caractères) des premiers morceaux dits à voix haute. La voix fabrique
# chaque morceau d'un coup, en un temps proportionnel à sa longueur : un premier morceau court
# fait entendre la voix plus tôt, puis les morceaux grandissent pour que le suivant soit prêt
# avant la fin du précédent (pas de blanc entre deux morceaux).
LIMITES = (32, 45, 90)
LIMITE_ENSUITE = 160


def limite(n: int) -> int:
    return LIMITES[n] if n < len(LIMITES) else LIMITE_ENSUITE


def scinder(phrase: str, n: int) -> list[str]:
    """Coupe une phrase trop longue pour son rang `n` : à une virgule, ou pour le tout premier
    morceau, entre deux mots."""
    morceaux = []
    while len(phrase) > limite(n):
        i = phrase.rfind(",", 6, limite(n) + 1)
        if i <= 0 and n == 0:
            i = phrase.rfind(" ", 12, limite(n) + 1)
        if i <= 0:
            break
        morceaux.append(phrase[: i + 1].strip())
        phrase = phrase[i + 1 :].lstrip()
        n += 1
    if phrase.strip():
        morceaux.append(phrase)
    return morceaux


def debut_a_dire(tampon: str) -> tuple[str | None, str]:
    """Avant la première phrase complète : le premier morceau peut partir dès la première
    virgule, ou après une trentaine de caractères sans ponctuation (entre deux mots)."""
    i = tampon.find(",", 6)
    if 0 < i <= limite(0):
        return tampon[: i + 1], tampon[i + 1 :].lstrip()
    if len(tampon) > limite(0):
        j = tampon.rfind(" ", 12, limite(0) + 1)
        if j > 0:
            return tampon[:j], tampon[j + 1 :]
    return None, tampon
