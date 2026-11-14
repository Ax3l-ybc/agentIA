"""Préparation du texte lu à voix haute : découpe en phrases et nettoyage."""

from __future__ import annotations

import re

_FIN_PHRASE = re.compile(r"([.!?…;:])(\s+)")
_A_NETTOYER = re.compile(r"[*_#`>|~]|\[\[|\]\]")
_EMOJIS = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


def nettoyer_pour_voix(texte: str) -> str:
    """Retire ce qui ne se prononce pas (Markdown, émojis)."""
    texte = _EMOJIS.sub("", _A_NETTOYER.sub(" ", texte))
    return re.sub(r"\s+", " ", texte).strip()


def couper_phrases(tampon: str, premiere: bool) -> tuple[list[str], str]:
    """Sépare les phrases complètes du reste du texte en attente."""
    phrases = []
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
