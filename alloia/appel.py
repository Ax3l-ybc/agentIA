"""Un appel AlloIA (version « omni ») : le modèle écoute directement la voix du maker.

Déroulement d'un tour de parole :
1. le micro est découpé en phrases (détecteur Silero, pause de fin de phrase) ;
2. l'audio de la phrase est envoyé tel quel au modèle (Gemma 4), avec l'historique ;
3. la réponse arrive en flux et chaque phrase est dite par la voix dès qu'elle est complète ;
4. en arrière-plan, la phrase du maker est retranscrite (historique et documentation).
Le micro est coupé pendant que l'agent parle (anti-écho).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import random
import re
import threading
import time

import numpy as np

from agent.serveur_voix import couper_phrases, nettoyer_pour_voix
from agent.moteurs import Reechantillonneur

from . import ollama_audio
from .audio import Combine
from .phrases import DecoupeurPhrases
from .reglages import Reglages

log = logging.getLogger("alloia.appel")

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "raccrocher",
            "description": "Termine l'appel. À utiliser après avoir remercié le maker, quand il a "
            "fini de raconter son projet ou qu'il dit au revoir.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

ADAPTATION_ORALE = """
# Conversation téléphonique (règles de forme, prioritaires)
- Tout ce que tu écris est lu à voix haute par une voix de synthèse : phrases courtes et
  naturelles, pas de listes, pas de titres, pas d'émojis, pas de symboles, pas de Markdown.
- Les messages du maker t'arrivent en audio. Réponds à ce qu'il dit : ne répète pas et ne
  transcris pas ses paroles.
- Les messages entre parenthèses ne viennent pas du maker : ce sont des indications sur l'appel.
- Termine toujours ta réponse par une question ou une invitation à continuer : au téléphone,
  un silence met mal à l'aise. Seule exception : quand tu dis au revoir.
- Si le maker dit au revoir ou qu'il a fini : ne pose plus aucune question. Remercie-le en une
  phrase, dis que la documentation va être rédigée, et termine par « Au revoir ! ».
- Commence chaque réponse par une réaction très courte de quelques mots (comme on le fait au
  téléphone), suivie d'une virgule ou d'un point, puis enchaîne. Varie ces réactions : ne
  commence jamais deux réponses de suite de la même façon.
"""

CONSIGNE_VOCALE = "(Message vocal du maker. Réponds-lui directement, sans répéter ses paroles.)"
DECROCHE = "(Le maker vient de décrocher le combiné. Commence l'appel.)"
RELANCE = ("(Le maker ne dit plus rien depuis un moment. Relance-le gentiment, en une phrase, "
           "sans reposer ta dernière question : propose plutôt de parler d'autre chose.)")
# Ajouté au message vocal suivant quand AlloIA vient de reposer une question déjà posée.
PASSER_A_AUTRE_CHOSE = ("(Tu as déjà posé cette question. Accepte sa réponse telle quelle, même "
                        "vague, et passe à un autre sujet sans revenir dessus.)")

_MOTS_VIDES = set("""alors aussi avec avez avoir avais cette comme dans donc elle elles est était
être fait faire leur mais même moins nous peux peut plus pour quel quelle quelles quels quoi sans
sont sous suis tout toute tous très vous votre veux""".split())
EXCUSE = "Désolé, j'ai eu un petit souci. Peux-tu répéter ce que tu viens de dire ?"

# Les pistes de la documentation, dans l'ordre où AlloIA les propose, avec les mots qui
# signalent qu'une question porte dessus. Le programme tient le fil : un sujet sur lequel
# AlloIA a déjà posé deux questions est clos, et on lui propose la piste suivante.
PISTES = [
    ("à quoi sert le projet et d'où vient l'idée",
     ("but", "sert", "servir", "idée", "usage", "utilité", "inspir", "pourquoi")),
    ("les matériaux", ("bois", "matériau", "matière", "contreplaqué", "mdf", "épaisseur", "pla",
                       "filament", "plexi", "acrylique", "tissu", "métal", "chute", "dimension")),
    ("les machines et leurs réglages", ("laser", "imprimante", "impression", "fraiseuse", "cnc",
                                        "machine", "réglage", "puissance", "vitesse", "brodeuse")),
    ("les logiciels et les fichiers", ("logiciel", "fichier", "inkscape", "fusion", "dessin",
                                       "dessiné", "modélis", "programm", "code")),
    ("les étapes de fabrication", ("étape", "assembl", "montage", "monté", "finition", "colle",
                                   "procédé", "fabrication")),
    ("les difficultés et comment il s'en est sorti", ("difficult", "problème", "raté", "galère",
                                                       "obstacle", "compliqué", "délicat", "erreur")),
    ("le temps passé et le coût", ("temps", "heure", "durée", "coût", "prix", "combien", "budget")),
    ("ses conseils pour quelqu'un qui voudrait le refaire", ("conseil", "refaire", "améliorer",
                                                             "changer", "recommand")),
]


# Panel de réactions pour commencer une réponse : à chaque tour, le programme en propose une
# neutre et une enthousiaste, parmi celles qu'AlloIA n'a pas utilisées récemment.
REACTIONS_NEUTRES = ["D'accord", "Je vois", "Ok", "Je note", "Entendu", "Je comprends", "Ah oui",
                     "Très bien", "Ça marche", "Oui", "Ah d'accord", "Hmm"]
REACTIONS_ENTHOUSIASTES = ["Super", "Génial", "Chouette", "Pas mal", "Bien vu", "Top", "Excellent",
                           "Trop bien", "Intéressant", "Sympa", "Waouh", "Bravo"]
# Petits mots qu'on retire s'ils reviennent en début de réponse juste après avoir été dits.
_TIC = re.compile(r"^\s*(ah|oh|waouh|wow|hmm|super|génial|intéressant)\s*[,!.…]+\s*", re.I)


def mot_d_ouverture(texte: str) -> str:
    m = re.match(r"\W*([\w'’]+)", texte)
    return m.group(1).lower() if m else ""


def proposer_reactions(recentes: list[str], hasard: random.Random = random) -> tuple[str, str]:
    """Une réaction neutre et une enthousiaste dont le premier mot n'a pas servi récemment."""
    def choisir(panel):
        libres = [x for x in panel if mot_d_ouverture(x) not in recentes]
        return hasard.choice(libres or panel)
    return choisir(REACTIONS_NEUTRES), choisir(REACTIONS_ENTHOUSIASTES)


def retirer_tic(texte: str, recentes: list[str]) -> str:
    """Retire un « Ah, » (ou « Super ! »…) en début de réponse s'il vient d'être dit."""
    m = _TIC.match(texte)
    if not m or m.group(1).lower() not in recentes:
        return texte
    reste = texte[m.end():]
    return reste[:1].upper() + reste[1:] if reste else texte


def pistes_de(question: str) -> list[int]:
    q = question.lower()
    return [i for i, (_, mots) in enumerate(PISTES) if any(m in q for m in mots)]


def mots_cles(question: str) -> set[str]:
    return {m for m in re.findall(r"\w+", question.lower()) if len(m) > 3 and m not in _MOTS_VIDES}


def derniere_question(texte: str) -> str:
    """La dernière phrase interrogative d'une réponse d'AlloIA (ou "")."""
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


_AU_REVOIR = re.compile(r"\b(au revoir|bonne journée|bonne soirée|à plus tard|à bientôt|ciao)\b", re.I)
_EMOJIS = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


@dataclasses.dataclass
class Echange:
    role: str  # "maker" ou "alloia"
    texte: str


@dataclasses.dataclass
class Conversation:
    debut: datetime.datetime
    echanges: list[Echange] = dataclasses.field(default_factory=list)
    raison_fin: str = ""

    def ajouter(self, role: str, texte: str) -> Echange:
        e = Echange(role, texte.strip())
        self.echanges.append(e)
        return e

    def maker_a_parle(self) -> bool:
        return any(e.role == "maker" and e.texte for e in self.echanges)

    def transcription(self) -> str:
        noms = {"maker": "Maker", "alloia": "AlloIA"}
        return "\n".join(f"{noms[e.role]} : {e.texte}" for e in self.echanges if e.texte)


# ---------------------------------------------------------------------------
# Lecture anticipée : pendant la veille, le cerveau lit déjà les consignes
# ---------------------------------------------------------------------------


def systeme_appel(consignes: str) -> str:
    return consignes.strip() + "\n\n" + ADAPTATION_ORALE.strip()


def prechauffer(cerveau, consignes: str) -> float:
    """Fait lire au cerveau, à l'avance, exactement le début du prochain appel.

    Au décroché, llama.cpp n'a plus que la salutation à écrire. Renvoie la durée (s).
    """
    debut = time.monotonic()
    messages = [{"role": "system", "content": systeme_appel(consignes)},
                {"role": "user", "content": DECROCHE}]
    for _ in cerveau.discuter_en_flux(messages, OUTILS, max_mots=1):
        pass
    return time.monotonic() - debut


# ---------------------------------------------------------------------------
# Veille : attendre que quelqu'un décroche et parle
# ---------------------------------------------------------------------------


async def attendre_decroche(
    combine: Combine, detecteur, duree_parole_s: float, evenement_clavier: asyncio.Event
) -> str:
    """Attend qu'on parle dans le combiné (ou qu'on appuie sur Entrée). Renvoie la cause."""
    reech = Reechantillonneur(24000, 16000)
    reste = np.zeros(0, dtype=np.float32)
    parole: list[float] = []
    if hasattr(detecteur, "reinitialiser"):
        detecteur.reinitialiser()
    combine.vider_micro()
    while True:
        if evenement_clavier.is_set():
            evenement_clavier.clear()
            return "clavier"
        try:
            trame = await asyncio.wait_for(combine.micro.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        reste = np.concatenate([reste, reech(trame)])
        maintenant = time.monotonic()
        while len(reste) >= 512:
            bloc, reste = reste[:512], reste[512:]
            if detecteur(bloc) > 0.5:
                parole.append(maintenant)
        parole = [t for t in parole if maintenant - t < 1.5]
        if len(parole) * 0.032 >= duree_parole_s:
            return "voix"


# ---------------------------------------------------------------------------
# L'appel
# ---------------------------------------------------------------------------


class Appel:
    def __init__(self, r: Reglages, consignes: str, voix, detecteur, cerveau, reparer=None):
        self.r = r
        self.systeme = systeme_appel(consignes)
        self.voix = voix
        self.cerveau = cerveau  # CerveauLlamaCpp ou CerveauOllama (alloia/cerveau.py)
        self.reparer = reparer  # redémarre le cerveau s'il s'est arrêté (llama.cpp)
        self.detecteur = detecteur
        self.historique: list[dict] = []  # messages texte (sans audio) des tours précédents
        self.transcriptions: set[asyncio.Task] = set()
        self.raccrocher = False
        self.questions: list[str] = []  # questions déjà posées par AlloIA
        self.questions_par_piste = [0] * len(PISTES)
        self.ouvertures: list[str] = []  # premier mot de chaque réponse d'AlloIA
        self.repetition = False  # AlloIA vient de reposer une question déjà posée
        self.raccroche: asyncio.Event | None = None  # combiné reposé sur le socle (bouton)

    # --------------------------------------------------------------- un tour

    async def tour(self, combine: Combine, conv: Conversation, audio: np.ndarray | None,
                   indication: str = "", fin_phrase: float | None = None,
                   feu_vert: asyncio.Event | None = None, arret: threading.Event | None = None) -> str:
        """Envoie une phrase du maker (ou une indication) et dit la réponse.

        Envoi anticipé : avec `feu_vert`, la réponse est préparée mais rien n'est dit ni gardé
        tant que le feu n'est pas vert (la personne a vraiment fini sa phrase). Si elle reprend
        la parole, la tâche est annulée et `arret` interrompt le cerveau.
        """
        await self._attendre_transcriptions(10)
        if audio is not None:
            wav = ollama_audio.wav_16k(audio)
            consigne = CONSIGNE_VOCALE
            # Rappel des dernières questions posées : un petit modèle oublie vite ce qu'il a déjà
            # demandé et a tendance à insister. (Ce rappel n'est pas gardé dans l'historique : il
            # ne gêne pas la mémoire de llama.cpp.)
            neutre, enthousiaste = proposer_reactions(self.ouvertures[-5:])
            consigne = (self.fil_conducteur() + f"(Pour commencer ta réponse, choisis « {neutre}, » ou "
                        f"« {enthousiaste}, » selon ce qu'il vient de dire.) " + consigne)
            if self.repetition:  # filet de sécurité : ne pas insister sur une question
                consigne = PASSER_A_AUTRE_CHOSE + " " + consigne
            actuel = self.cerveau.message_vocal(wav, consigne)
        else:
            actuel = {"role": "user", "content": indication}
        self._faire_de_la_place(len(audio) / 16000 if audio is not None else 0.0)
        messages = [{"role": "system", "content": self.systeme}, *self.historique, actuel]

        boucle = asyncio.get_running_loop()
        file: asyncio.Queue = asyncio.Queue()
        debut = time.monotonic()

        def produire():
            try:
                options = {"arret": arret} if arret is not None and hasattr(self.cerveau, "interrompre") else {}
                for m in self.cerveau.discuter_en_flux(messages, OUTILS, **options):
                    if arret is not None and arret.is_set():
                        break  # envoi anticipé abandonné : on libère le cerveau
                    boucle.call_soon_threadsafe(file.put_nowait, m)
            except Exception as exc:  # noqa: BLE001
                boucle.call_soon_threadsafe(file.put_nowait, exc)
            finally:
                boucle.call_soon_threadsafe(file.put_nowait, None)

        threading.Thread(target=produire, daemon=True).start()

        reponse, tampon, premiere = "", "", True
        premier_mot = premier_son = None
        while True:
            m = await file.get()
            if feu_vert is not None and not feu_vert.is_set():
                await feu_vert.wait()  # rien n'est dit avant que la phrase soit vraiment finie
            if isinstance(m, Exception):
                log.error("Le cerveau n'a pas répondu : %s", m)
                if self.reparer is not None and not reponse:
                    log.warning("Redémarrage du cerveau en mode plus économe…")
                    await self.voix.dire("Désolé, une petite seconde.", combine)
                    try:
                        await asyncio.to_thread(self.reparer)
                        reponse = EXCUSE
                        await self.voix.dire(EXCUSE, combine)
                    except Exception as exc:  # noqa: BLE001
                        log.error("Le cerveau n'a pas pu redémarrer : %s", exc)
                break
            phrases: list[str] = []
            if m is None:
                if tampon.strip():
                    phrases.append(tampon)
                tampon = ""
            elif "outil" in m:
                if m["outil"] == "raccrocher":
                    log.info("L'agent raccroche.")
                    self.raccrocher = True
            else:
                if premier_mot is None:
                    premier_mot = time.monotonic() - debut
                    log.info("  réflexion : premier mot après %.1f s", premier_mot)
                reponse += m["texte"]
                tampon += m["texte"]
                phrases, tampon = couper_phrases(tampon, premiere)
                if premiere and not phrases:
                    # Phonon parle dès le premier morceau : inutile d'attendre la fin de la
                    # première phrase, on commence à la première virgule.
                    i = tampon.find(",", 6)  # « Je vois, » ou « Intéressant, » mais pas « Ah, »
                    if i > 0:
                        phrases, tampon = [tampon[: i + 1]], tampon[i + 1 :].lstrip()
            for p in phrases:
                if self.raccroche is not None and self.raccroche.is_set():
                    continue  # combiné reposé : on finit la réponse en silence (pour la doc)
                if premiere:
                    p = retirer_tic(p, self.ouvertures[-3:])
                texte = _EMOJIS.sub("", nettoyer_pour_voix(p))
                if not re.search(r"\w", texte):
                    continue
                premiere = False
                delai = await self.voix.dire(texte, combine)
                if premier_son is None:
                    premier_son = time.monotonic()
                    total = premier_son - (fin_phrase if fin_phrase else debut)
                    log.info("  voix : premier son après %.1f s de synthèse (total : %.1f s)", delai, total)
            if m is None:
                break

        reponse = retirer_tic(_EMOJIS.sub("", reponse).strip(), self.ouvertures[-3:])
        if reponse:
            self.ouvertures.append(mot_d_ouverture(reponse))
        if audio is not None:
            self.repetition = False  # le rappel « passe à autre chose » a été transmis
        # Historique en texte : la phrase du maker sera remplacée par sa transcription.
        message_maker = {"role": "user", "content": indication or "(message vocal)"}
        self.historique.append(message_maker)
        self.historique.append({"role": "assistant", "content": reponse or "…"})
        if audio is not None:
            echange = conv.ajouter("maker", "")
            self._lancer_transcription(wav, message_maker, echange)
        conv.ajouter("alloia", reponse)
        if _AU_REVOIR.search(reponse):  # AlloIA a dit au revoir : on raccroche après sa phrase
            self.raccrocher = True
        question = derniere_question(reponse)
        if question:
            for i in pistes_de(question):
                self.questions_par_piste[i] += 1
            if deja_posee(question, self.questions):
                log.info("  question déjà posée : AlloIA passera à autre chose au tour suivant.")
                self.repetition = True
            self.questions.append(question)
        return reponse

    def _faire_de_la_place(self, secondes_audio: float) -> None:
        """Si la mémoire de travail de llama.cpp va déborder (appel très long), oublie les plus
        vieux échanges (en gardant la salutation). On en oublie beaucoup d'un coup : le cerveau
        doit alors relire l'historique, et c'est mieux que ça n'arrive qu'une fois."""
        def estimation() -> float:  # environ 3 caractères par jeton en français, 25 jetons/s d'audio
            texte = len(self.systeme) + sum(len(str(m.get("content"))) for m in self.historique)
            return texte / 3 + 400 + secondes_audio * 25 + 300  # + outils, audio, réponse

        limite = getattr(self.r, "llamacpp_contexte", 8192)
        if estimation() < 0.85 * limite:
            return
        oublies = 0
        while len(self.historique) > 4 and estimation() > 0.6 * limite:
            del self.historique[2:4]  # le plus vieil échange après la salutation
            oublies += 1
        log.info("  mémoire de travail presque pleine : %d anciens échanges oubliés", oublies)

    def fil_conducteur(self) -> str:
        """Indication donnée au cerveau avec chaque phrase du maker : les sujets clos (sur
        lesquels AlloIA a déjà insisté) et la prochaine piste à proposer."""
        if not self.questions:
            return ""
        clos = [PISTES[i][0] for i, n in enumerate(self.questions_par_piste) if n >= 2]
        libres = [PISTES[i][0] for i, n in enumerate(self.questions_par_piste) if n == 0]
        texte = f"(Ta dernière question : « {self.questions[-1]} » Ne la repose pas."
        if clos:
            texte += f" Sujets déjà traités, n'y reviens plus : {', '.join(clos)}."
        if libres:
            texte += f" Si sa réponse ne t'ouvre pas une suite naturelle, passe à : {libres[0]}."
        else:
            texte += " Tu as fait le tour : demande-lui s'il veut ajouter quelque chose."
        return texte + ") "

    def _lancer_transcription(self, wav: bytes, message: dict, echange: Echange) -> None:
        async def tache():
            t0 = time.monotonic()
            try:
                texte = await asyncio.to_thread(self.cerveau.transcrire, wav)
            except Exception as exc:  # noqa: BLE001
                log.warning("Transcription impossible : %s", exc)
                texte = ""
            message["content"] = texte or "(message vocal inaudible)"
            echange.texte = texte
            log.info("  maker (transcrit en %.1f s) : %s", time.monotonic() - t0, texte)

        t = asyncio.create_task(tache())
        self.transcriptions.add(t)
        t.add_done_callback(self.transcriptions.discard)

    async def _attendre_transcriptions(self, delai: float) -> None:
        if self.transcriptions:
            await asyncio.wait(set(self.transcriptions), timeout=delai)

    # ------------------------------------------------------------ l'appel

    async def derouler(self, combine: Combine, evenement_clavier: asyncio.Event,
                       raccroche: asyncio.Event | None = None) -> Conversation:
        """Déroule l'appel. `raccroche` : posé quand le combiné est reposé sur le socle."""
        r = self.r
        self.raccroche = raccroche
        conv = Conversation(debut=datetime.datetime.now())
        anticipation = r.anticipation_s if 0 < r.anticipation_s < r.pause_s else None
        decoupeur = DecoupeurPhrases(self.detecteur, pause_s=r.pause_s, anticipation_s=anticipation)
        phrases: asyncio.Queue = asyncio.Queue()
        fin = asyncio.Event()
        debut = time.monotonic()
        derniere_parole = [time.monotonic()]
        combine.vider_micro()

        async def ecouter():
            while not fin.is_set():
                try:
                    trame = await asyncio.wait_for(combine.micro.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not accueil_fini.is_set():
                    decoupeur.reinitialiser()  # « Allô ? » du décroché : on attend la salutation
                    continue
                if r.micro_coupe_quand_agent_parle and combine.agent_parle():
                    continue  # l'agent ne doit pas s'entendre lui-même
                for p in decoupeur.ajouter(trame):
                    derniere_parole[0] = time.monotonic()
                    # La personne a fini de parler `pause_s` avant que la phrase soit livrée.
                    phrases.put_nowait(("phrase", p, time.monotonic() - r.pause_s))

        def provisoire(audio, silence_s):
            phrases.put_nowait(("provisoire", audio, time.monotonic() - silence_s))

        decoupeur.sur_provisoire = provisoire
        decoupeur.sur_reprise = lambda: phrases.put_nowait(("reprise", None, 0.0))
        # Envoi anticipé en cours : (tâche, feu vert, arrêt, audio)
        anticipe: list = [None]

        def abandonner_anticipe():
            if anticipe[0] is not None:
                tache, _, arret, _ = anticipe[0]
                if hasattr(self.cerveau, "interrompre"):
                    self.cerveau.interrompre(arret)  # coupe net la réponse préparée pour rien
                arret.set()
                tache.cancel()
                anticipe[0] = None

        async def surveiller_socle():
            await raccroche.wait()
            log.info("Combiné reposé : fin de l'appel.")
            combine.vider_haut_parleur()  # la voix s'arrête net

        accueil_fini = asyncio.Event()
        ecoute = asyncio.create_task(ecouter())
        socle = asyncio.create_task(surveiller_socle()) if raccroche is not None else None
        try:
            await self.tour(combine, conv, None, DECROCHE)
            while combine.agent_parle():  # on écoute à nouveau une fois la salutation dite
                await asyncio.sleep(0.05)
            accueil_fini.set()
            relances = 0
            derniere_activite = time.monotonic()
            while True:
                try:
                    genre, audio, fin_phrase = await asyncio.wait_for(phrases.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    genre, audio = None, None
                if genre == "provisoire":
                    # La personne s'est peut-être arrêtée : on prépare déjà la réponse.
                    abandonner_anticipe()
                    feu, arret = asyncio.Event(), threading.Event()
                    tache = asyncio.create_task(
                        self.tour(combine, conv, audio, fin_phrase=fin_phrase, feu_vert=feu, arret=arret))
                    anticipe[0] = (tache, feu, arret, audio)
                    continue
                if genre == "reprise":  # finalement non : elle continue sa phrase
                    if anticipe[0] is not None:
                        log.info("  (réponse anticipée abandonnée : le maker continue)")
                    abandonner_anticipe()
                    continue
                if genre == "phrase" and anticipe[0] is not None:
                    tache, feu, _, audio_anticipe = anticipe[0]
                    anticipe[0] = None
                    if len(audio_anticipe) == len(audio) and phrases.empty():
                        feu.set()  # c'était bien la fin de la phrase : la réponse est déjà prête
                        relances = 0
                        await tache
                        derniere_activite = time.monotonic()
                        continue
                    tache.cancel()
                maintenant = time.monotonic()
                if raccroche is not None and raccroche.is_set():
                    conv.raison_fin = "combiné raccroché"
                    if anticipe[0] is not None:  # sa dernière phrase est gardée pour la doc
                        tache, feu, _, _ = anticipe[0]
                        anticipe[0] = None
                        feu.set()
                        await tache
                    break
                if combine.agent_parle():
                    derniere_activite = maintenant
                if audio is not None:
                    # Plusieurs phrases arrivées pendant la réflexion : on les regroupe.
                    while not phrases.empty():
                        g, suite, f = phrases.get_nowait()
                        if g == "phrase":
                            audio, fin_phrase = np.concatenate([audio, suite]), f
                    relances = 0
                    await self.tour(combine, conv, audio, fin_phrase=fin_phrase)
                    derniere_activite = time.monotonic()
                    continue
                if self.raccrocher and not combine.agent_parle():
                    conv.raison_fin = "l'agent a raccroché"
                    break
                if evenement_clavier.is_set():
                    evenement_clavier.clear()
                    conv.raison_fin = "touche Entrée"
                    break
                if maintenant - debut > r.duree_max_min * 60:
                    conv.raison_fin = "durée maximale atteinte"
                    break
                if decoupeur.en_cours or combine.agent_parle():
                    continue
                dernier_maker = next((e.texte for e in reversed(conv.echanges) if e.role == "maker"), "")
                # Filet de sécurité si le modèle n'appelle pas « raccrocher » : un « au revoir »
                # du maker termine l'appel peu après la dernière réponse d'AlloIA.
                if _AU_REVOIR.search(dernier_maker[-60:]) and maintenant - derniere_activite > 1.5:
                    conv.raison_fin = "au revoir"
                    break
                if maintenant - derniere_parole[0] > r.fin_apres_silence_s:
                    conv.raison_fin = "silence prolongé"
                    break
                if maintenant - derniere_activite > r.relance_apres_s and relances < 2:
                    relances += 1
                    await self.tour(combine, conv, None, RELANCE)
                    derniere_activite = time.monotonic()
        finally:
            abandonner_anticipe()
            fin.set()
            ecoute.cancel()
            if socle is not None:
                socle.cancel()
            await self._attendre_transcriptions(30)
            combine.vider_haut_parleur()
        log.info("Fin de l'appel (%s), %d échanges.", conv.raison_fin, len(conv.echanges))
        return conv
