# AlloIA : le téléphone de documentation du FabLab (version « omni »)

Un maker vient de finir son projet. Il décroche le combiné posé près de la Jetson, et AlloIA
lui pose des questions, avec un vrai intérêt, pour lui faire raconter **ce qu'il a fabriqué,
comment, avec quels matériaux et quelles machines, les difficultés rencontrées et ses
conseils**. Quand il raccroche, AlloIA rédige toute seule la **documentation du projet**
dans un fichier Markdown.

- **100 % local et gratuit** : tout tourne sur la carte Jetson, rien ne part sur internet.
- **Seul le texte est gardé** (documentation et transcription), jamais l'enregistrement audio.
  AlloIA le dit à chaque appel.

> **Branche `AlloIA-omni`** : ici, le cerveau **écoute directement la voix** (Gemma 4 dans
> Ollama), sans passer par une transcription Whisper. La branche `AlloIA` garde la version
> précédente (Whisper + Ministral 3 + Gradbot).

---

## Le matériel

- Une **Jetson Orin Nano** (8 Go) avec JetPack 6.
- Une **carte son USB** avec un **micro** (combiné, prise jack) et un **haut-parleur**.
- Une connexion internet pour l'installation et le premier lancement seulement.

## Comment ça marche

| Rôle | Logiciel | Où il tourne |
|---|---|---|
| Les oreilles **et** le cerveau | [Gemma 4 E2B](https://ollama.com/library/gemma4) de Google, dans [Ollama](https://ollama.com) : il écoute l'audio et répond | puce graphique de la Jetson |
| La bouche | [Phonon](https://huggingface.co/gradium/phonon-40m-fr) `phonon-40m-fr` de Gradium | processeur de la Jetson |
| Le détecteur de parole | Silero (repère le début et la fin de vos phrases) | processeur de la Jetson |

Pour chaque phrase du maker :

1. le détecteur repère la fin de la phrase (une pause d'environ 1,3 s) ;
2. **l'enregistrement de la phrase est envoyé tel quel à Gemma 4**, avec l'historique de l'appel ;
3. la réponse arrive mot à mot, et chaque phrase est dite par Phonon dès qu'elle est complète ;
4. pendant qu'AlloIA parle, Gemma 4 retranscrit la phrase du maker, pour l'historique et la doc.

Le micro est coupé pendant qu'AlloIA parle, pour qu'elle ne s'entende pas elle-même.

Déroulement d'un appel :

1. **En attente** : AlloIA écoute le micro. Dès qu'on parle (« Allô ? »), l'appel commence.
2. **L'appel** : « Allô ! Ici AlloIA… ». Une question à la fois.
3. **La fin** : quand tout est raconté ou que la personne dit au revoir, AlloIA remercie et
   raccroche. L'appel se termine aussi après une minute de silence, ou après 25 minutes.
4. **La documentation** est rédigée dans `~/AlloIA/docs/`, par exemple
   `2026-11-12_14h05_lampe-en-contreplaque.md`. Puis AlloIA se remet en attente.

---

## Passer de la branche AlloIA à AlloIA-omni (si AlloIA est déjà installée)

Dans un terminal sur la Jetson, une commande à la fois :

```bash
cd ~/AlloIA-code
```

```bash
cp alloia.yaml ~/ancien-alloia.yaml && git checkout -- alloia.yaml
```

```bash
git fetch origin && git checkout AlloIA-omni
```

```bash
./jetson/installer.sh
```

```bash
.venv/bin/python jetson/garder_reglages.py ~/ancien-alloia.yaml
```

La dernière commande recopie votre jeton Hugging Face, votre carte son et votre voix dans
`alloia.perso.yaml` (voir plus bas). Ensuite : `./jetson/lancer.sh`.

## Première installation

```bash
git clone -b AlloIA-omni https://github.com/Ax3l-ybc/agentIA.git ~/AlloIA-code
```

```bash
cd ~/AlloIA-code && ./jetson/installer.sh
```

Le script installe les outils nécessaires, la voix Phonon et le détecteur de parole, et
installe ou **met à jour Ollama** (il faut la version 0.40.13 ou plus récente). Il demande
votre mot de passe (celui de la session sur la Jetson). Il peut être relancé sans risque.

## Vos réglages personnels : `alloia.perso.yaml`

Mettez vos réglages personnels dans le fichier **`alloia.perso.yaml`** (à créer dans le
dossier du projet), plutôt que dans `alloia.yaml`. Ses valeurs sont prioritaires, et il
n'est **jamais modifié par les mises à jour** (`git pull`). Exemple :

```yaml
jeton_huggingface: 'hf_votre_jeton'
audio:
  entree: 'KT USB'
  sortie: 'KT USB'
voix:
  nom: 'Jules'
```

Pour l'ouvrir : `gedit alloia.perso.yaml`.

### La voix : jeton Hugging Face

La voix française Phonon est gratuite (usage non commercial) mais demande d'accepter ses
conditions : sur [huggingface.co/gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr),
acceptez les conditions, puis créez un jeton de type « Read » sur
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) et copiez-le dans
`alloia.perso.yaml`.

### Le son

Branchez la carte son USB, puis :

```bash
./jetson/lancer.sh --lister-audio
```

Mettez un morceau du nom de votre carte (par exemple `KT USB`) dans `alloia.perso.yaml`,
aux lignes `entree:` et `sortie:`. Puis testez :

```bash
./jetson/lancer.sh --test-audio
```

## Lancer AlloIA

```bash
cd ~/AlloIA-code && ./jetson/lancer.sh
```

Au premier lancement, AlloIA télécharge Gemma 4 E2B (environ 4,6 Go) et la voix. Quand le
terminal affiche **« En attente d'un appel »**, décrochez et dites « Allô ? ».
Pour les essais, la touche **Entrée** décroche et raccroche aussi. Pour arrêter : **Ctrl + C**
(attendez « Documentation enregistrée » si un appel vient de se terminer).

Pendant l'appel, le terminal affiche les temps de réponse :

```
  réflexion : premier mot après 0.9 s
  voix : premier son après 0.3 s de synthèse (total : 2.6 s)
  maker (transcrit en 1.2 s) : J'ai fabriqué une lampe…
```

---

## Personnaliser

- **`alloia_conduite.md`** : la personnalité d'AlloIA et ce qu'elle doit faire raconter.
  Relu à chaque appel : une modification s'applique tout de suite.
- **`alloia_modele_doc.md`** : le modèle de la documentation produite.
- **`alloia.yaml`** (ou mieux, `alloia.perso.yaml`) : les réglages.
  - `ollama > modele` : `gemma4:e2b` (conseillé) ou `lfm3-omni` (plus léger, 2 Go).
  - `ecoute > pause_fin_de_phrase_s` : plus court, AlloIA répond plus vite mais risque de
    couper la parole ; plus long, elle laisse le temps de réfléchir.
  - `voix > moteur` : `phonon` (conseillé) ou `ollama` (la voix de `lfm3-omni`, sans jeton
    Hugging Face, mais plus lente).

## Mémoire de la carte

La Jetson Orin Nano a 8 Go partagés entre le processeur et la puce graphique. AlloIA en
utilise environ 5 Go. Si la carte manque de mémoire (lenteurs, arrêt brutal), désactivez
le bureau graphique, qui occupe environ 1 Go :

```bash
sudo systemctl set-default multi-user.target
```

Puis redémarrez. Pour retrouver le bureau : `sudo systemctl set-default graphical.target`.

## En cas de problème

| Ce que vous voyez | Que faire |
|---|---|
| « Ollama … est trop ancien » | `curl -fsSL https://ollama.com/install.sh \| sh`, puis relancez. |
| « Ollama ne répond pas » | `sudo systemctl restart ollama`, puis relancez. |
| « Aucun micro ou haut-parleur utilisable » | Vérifiez la carte son avec `arecord -l` et `aplay -l`, puis son nom dans `alloia.perso.yaml`. |
| « Téléchargement de la voix refusé par Hugging Face » | Acceptez les conditions de Phonon et vérifiez le jeton dans `alloia.perso.yaml`. |
| AlloIA coupe la personne trop tôt | Augmentez `pause_fin_de_phrase_s` (par exemple `1.8`). |
| L'appel démarre tout seul (bruit de l'atelier) | Augmentez `parole_pour_decrocher_s` (par exemple `1.2`). |
| `git pull` refuse à cause d'`alloia.yaml` | Voir « Passer de la branche AlloIA à AlloIA-omni » : vos réglages vont dans `alloia.perso.yaml`. |

Envoyez les messages du terminal pour qu'on corrige ensemble.

## Pour les développeurs

```
alloia.yaml            réglages (alloia.perso.yaml : réglages personnels, non suivis par git)
alloia_conduite.md     personnalité et consignes de l'agent
alloia_modele_doc.md   modèle de la documentation
alloia/
  __main__.py          boucle : attente → appel → documentation
  appel.py             un appel : tours de parole, relances, fin d'appel, transcription
  ollama_audio.py      Ollama : conversation avec audio, transcription, synthèse vocale
  phrases.py           découpe du micro en phrases (Silero)
  voix.py              voix Phonon ou Ollama
  audio.py             le combiné : carte son USB (ou simulé pour les tests)
  doc.py               rédaction et enregistrement de la documentation
  reglages.py          lecture des réglages
agent/                 briques partagées avec le directeur de mémoire (Phonon, Ollama…)
jetson/                installation, lancement, reprise des réglages
tests/                 tests automatiques (combiné simulé, faux Ollama)
```

**Licences** : le code est libre. Phonon 40M est sous licence CC BY-NC 4.0 (usage non
commercial) ; vérifiez les licences des autres modèles sur leurs pages.
