"""Lecture des fichiers joints à une conversation : PDF, Word, texte (et export Word).

Le fichier n'est pas gardé : on n'en garde que le texte, dans la conversation, sur le PC.
"""

from __future__ import annotations

import io
import re

EXTENSIONS_TEXTE = (".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".log", ".rtf")
EXTENSIONS = (".pdf", ".docx") + EXTENSIONS_TEXTE
TAILLE_MAX = 30 * 1024 * 1024  # 30 Mo
# Au-delà, le texte est tronqué : la mémoire de travail du modèle a ses limites
# (16 384 jetons, soit environ 40 000 caractères en tout, consignes comprises).
CARACTERES_MAX = 24000


class FichierIllisible(Exception):
    pass


def extraire_texte(nom: str, donnees: bytes) -> str:
    """Le texte d'un fichier PDF, Word (.docx) ou texte."""
    ext = ("." + nom.rsplit(".", 1)[-1].lower()) if "." in nom else ""
    if len(donnees) > TAILLE_MAX:
        raise FichierIllisible(f"« {nom} » est trop gros (30 Mo au plus).")
    if ext == ".pdf":
        texte = _pdf(nom, donnees)
    elif ext == ".docx":
        texte = _docx(nom, donnees)
    elif ext in EXTENSIONS_TEXTE:
        texte = _texte(donnees)
    elif ext == ".doc":
        raise FichierIllisible(f"« {nom} » est au format Word ancien (.doc) : ouvrez-le dans Word et "
                               "enregistrez-le en .docx, ou en PDF.")
    else:
        raise FichierIllisible(f"« {nom} » : format non pris en charge (PDF, Word .docx ou texte).")
    texte = re.sub(r"[ \t]+\n", "\n", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte).strip()
    if not texte:
        raise FichierIllisible(f"« {nom} » ne contient pas de texte lisible (document scanné ?).")
    return texte


def _texte(donnees: bytes) -> str:
    for codage in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return donnees.decode(codage)
        except UnicodeDecodeError:
            continue
    return donnees.decode("utf-8", "replace")


def _pdf(nom: str, donnees: bytes) -> str:
    from pypdf import PdfReader

    try:
        lecteur = PdfReader(io.BytesIO(donnees))
        if lecteur.is_encrypted:
            lecteur.decrypt("")
        pages = []
        for i, page in enumerate(lecteur.pages, 1):
            texte = (page.extract_text() or "").strip()
            if texte:
                pages.append(f"[Page {i}]\n{texte}")
        return "\n\n".join(pages)
    except Exception as exc:  # noqa: BLE001
        raise FichierIllisible(f"« {nom} » n'a pas pu être lu (PDF protégé ou abîmé ?).") from exc


def _docx(nom: str, donnees: bytes) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(donnees))
    except Exception as exc:  # noqa: BLE001
        raise FichierIllisible(f"« {nom} » n'a pas pu être lu (fichier Word abîmé ?).") from exc
    lignes = []
    for bloc in document.element.body.iterchildren():
        balise = bloc.tag.rsplit("}", 1)[-1]
        if balise == "p":
            p = docx.text.paragraph.Paragraph(bloc, document)
            texte = p.text.strip()
            style = (p.style.name if p.style is not None else "") or ""
            if texte and style.lower().startswith(("heading", "titre")):
                niveau = re.findall(r"\d", style)
                texte = "#" * int(niveau[0] if niveau else 1) + " " + texte
            elif texte and "list" in style.lower():
                texte = "- " + texte
            lignes.append(texte)
        elif balise == "tbl":
            tableau = docx.table.Table(bloc, document)
            for rangee in tableau.rows:
                cellules = [c.text.strip().replace("\n", " ") for c in rangee.cells]
                lignes.append(" | ".join(cellules))
            lignes.append("")
    return "\n".join(lignes)


def tronquer(texte: str, maximum: int = CARACTERES_MAX) -> tuple[str, bool]:
    if len(texte) <= maximum:
        return texte, False
    return texte[:maximum].rsplit("\n", 1)[0], True


def bloc_pour_le_modele(fichiers: list[dict]) -> str:
    """Le texte des fichiers joints, tel qu'il est donné au modèle avec le message."""
    morceaux = []
    for f in fichiers:
        entete = f"=== FICHIER JOINT : {f['nom']} ==="
        if f.get("tronque"):
            entete += " (seul le début est fourni : le document est long)"
        morceaux.append(f"{entete}\n{f['texte']}\n=== FIN DE {f['nom']} ===")
    return "\n\n".join(morceaux)


# ---------------------------------------------------------------------------
# Export Word
# ---------------------------------------------------------------------------

def _en_ligne(paragraphe, texte: str) -> None:
    """Ajoute du texte avec **gras**, *italique* et `code` à un paragraphe Word."""
    for morceau in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|(?<![*\w])\*[^*\s][^*]*\*)", texte):
        if not morceau:
            continue
        if morceau.startswith("**") and morceau.endswith("**") and len(morceau) > 4:
            paragraphe.add_run(morceau[2:-2]).bold = True
        elif morceau.startswith("`") and morceau.endswith("`") and len(morceau) > 2:
            run = paragraphe.add_run(morceau[1:-1])
            run.font.name = "Consolas"
        elif morceau.startswith("*") and morceau.endswith("*") and len(morceau) > 2:
            paragraphe.add_run(morceau[1:-1]).italic = True
        else:
            paragraphe.add_run(morceau)


def ajouter_markdown(document, texte: str) -> None:
    """Ajoute un texte Markdown (titres, listes, gras, code) à un document Word."""
    lignes = texte.replace("\r", "").split("\n")
    i = 0
    while i < len(lignes):
        ligne = lignes[i]
        if ligne.strip().startswith("```"):
            code = []
            i += 1
            while i < len(lignes) and not lignes[i].strip().startswith("```"):
                code.append(lignes[i])
                i += 1
            p = document.add_paragraph()
            run = p.add_run("\n".join(code))
            run.font.name = "Consolas"
        elif m := re.match(r"^(#{1,6})\s+(.*)", ligne):
            document.add_heading(m.group(2).strip().strip("*"), level=min(len(m.group(1)), 3))
        elif m := re.match(r"^\s*[-*•]\s+(.*)", ligne):
            _en_ligne(document.add_paragraph(style="List Bullet"), m.group(1))
        elif m := re.match(r"^\s*\d+[.)]\s+(.*)", ligne):
            _en_ligne(document.add_paragraph(style="List Number"), m.group(1))
        elif m := re.match(r"^>\s?(.*)", ligne):
            _en_ligne(document.add_paragraph(style="Quote"), m.group(1))
        elif ligne.strip() and not re.match(r"^\s*([-*_])\s*\1\s*\1[\s\-*_]*$", ligne):
            _en_ligne(document.add_paragraph(), ligne.strip())
        i += 1


def document_word(titre: str, sections: list[tuple[str, str]]) -> bytes:
    """Un document Word : un titre, puis des sections (sous-titre éventuel, texte Markdown)."""
    import docx

    document = docx.Document()
    document.core_properties.title = titre
    document.add_heading(titre, level=0)
    for sous_titre, texte in sections:
        if sous_titre:
            document.add_heading(sous_titre, level=2)
        ajouter_markdown(document, texte)
    sortie = io.BytesIO()
    document.save(sortie)
    return sortie.getvalue()


def nom_de_fichier(titre: str) -> str:
    """Un nom de fichier simple, sans accents ni caractères interdits par Windows (tous les
    navigateurs l'acceptent tel quel)."""
    import unicodedata

    propre = unicodedata.normalize("NFKD", titre.replace("…", " "))
    propre = "".join(c for c in propre if not unicodedata.combining(c)).encode("ascii", "ignore").decode()
    propre = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", propre)
    propre = re.sub(r"\s+", " ", propre).strip().rstrip(".")[:80].strip()
    return (propre or "Document") + ".docx"
