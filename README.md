# AlloIA : le téléphone de documentation du FabLab (version « rapide »)

Un maker vient de finir son projet. Il décroche le combiné posé près de la Jetson, et AlloIA
lui pose des questions, avec un vrai intérêt, pour lui faire raconter **ce qu'il a fabriqué,
comment, avec quels matériaux et quelles machines, les difficultés rencontrées et ses
conseils**. Quand il raccroche, AlloIA rédige toute seule la **documentation du projet**
dans un fichier Markdown.

- **100 % local et gratuit** : tout tourne sur la carte Jetson, rien ne part sur internet.
- **Seul le texte est gardé** (documentation et transcription), jamais l'enregistrement audio.
  AlloIA le dit à chaque appel.

> **Branche `AlloIA-rapide`** : le cerveau **écoute directement la voix** (Gemma 4 E2B),
> et tourne dans **llama.cpp** compilé pour la puce graphique de la Jetson, dans une version
> compressée plus légère. C'est la version conçue pour répondre vite : la version Ollama
> (`AlloIA-omni`) saturait la mémoire de la carte (8 Go) et répondait en 40 à 50 s.
> La branche `AlloIA` garde la toute première version (Whisper + Ministral 3 + Gradbot).

---

## Le matériel

- Une **Jetson Orin Nano** (8 Go) avec JetPack 6.
- Une **carte son USB** avec un **micro** (combiné, prise jack) et un **haut-parleur**.
- Une connexion internet pour l'installation et le premier lancement seulement.

## Comment ça marche

| Rôle | Logiciel | Où il tourne |
|---|---|---|
| Les oreilles **et** le cerveau | [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) de Google (version compressée Q3_K_M, 2,4 Go, + module audio 0,9 Go), dans [llama.cpp](https://github.com/ggml-org/llama.cpp) : il écoute l'audio et répond | puce graphique de la Jetson |
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

## Passer à la version rapide (si AlloIA-omni est déjà installée)

Vos réglages personnels (`alloia.perso.yaml`) sont conservés. Dans un terminal sur la Jetson :

```bash
cd ~/AlloIA-code && git fetch origin && git checkout AlloIA-rapide
```

```bash
./jetson/installer.sh
```

L'installation compile llama.cpp (30 à 60 minutes, une seule fois) et télécharge le modèle
(3,3 Go). Elle **arrête Ollama**, qui n'est plus utilisé et garderait sinon un modèle en
mémoire. Ensuite : `./jetson/lancer.sh`.

Depuis la toute première version (branche `AlloIA`), mettez d'abord vos réglages de côté :
`cp alloia.yaml ~/ancien-alloia.yaml && git checkout -- alloia.yaml`, puis après l'installation
`.venv/bin/python jetson/garder_reglages.py ~/ancien-alloia.yaml`.

## Première installation

```bash
git clone -b AlloIA-rapide https://github.com/Ax3l-ybc/agentIA.git ~/AlloIA-code
```

```bash
cd ~/AlloIA-code && ./jetson/installer.sh
```

Le script installe les outils nécessaires, la voix Phonon et le détecteur de parole,
**compile llama.cpp** pour la puce graphique (30 à 60 minutes la première fois), télécharge
Gemma 4 E2B (3,3 Go dans `~/AlloIA/modeles`) et arrête Ollama s'il est installé. Il autorise
aussi AlloIA à vider le cache mémoire au lancement (voir « Mémoire de la carte »). Il demande
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

AlloIA démarre llama.cpp (le cerveau est prêt en quelques dizaines de secondes). Quand le
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
  - `cerveau > llamacpp > modele` : la version compressée de Gemma 4 E2B. Q3_K_M (défaut) est
    la plus rapide ; pour un peu plus de finesse, téléchargez la Q4_K_M (2,9 Go) dans
    `~/AlloIA/modeles` depuis [la page du modèle](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
    et indiquez son nom.
  - `cerveau > moteur` : `llamacpp` (conseillé) ou `ollama` (version précédente, plus lourde).
  - `ecoute > pause_fin_de_phrase_s` : plus court, AlloIA répond plus vite mais risque de
    couper la parole ; plus long, elle laisse le temps de réfléchir.
  - `voix > moteur` : `phonon` (conseillé) ou `ollama` (la voix de `lfm3-omni`, sans jeton
    Hugging Face, mais plus lente).

## Mémoire de la carte

La Jetson Orin Nano a 7,4 Go utilisables, partagés entre le processeur et la puce graphique.
**C'est la limite qui rendait AlloIA lente** : quand la mémoire déborde, la carte se sert du
disque (« swap ») et tout ralentit énormément. La version rapide utilise environ 4 Go pour le
cerveau, et vide le cache mémoire juste avant de le charger (la puce graphique de la Jetson
ne sait pas récupérer cette mémoire toute seule).

Pour vérifier pendant un appel : `free -h`. La ligne `Swap` doit rester proche de 0, et
`available` au-dessus de 1 Go. Sinon, désactivez le bureau graphique, qui occupe environ 1 Go :

```bash
sudo systemctl set-default multi-user.target
```

Puis redémarrez. Pour retrouver le bureau : `sudo systemctl set-default graphical.target`.

## En cas de problème

| Ce que vous voyez | Que faire |
|---|---|
| « llama.cpp s'est arrêté au démarrage » | Lisez la fin du journal affichée ; souvent la mémoire : fermez les autres programmes, ou passez sans bureau graphique. |
| « Fichier introuvable pour le modèle » | Relancez `./jetson/installer.sh` (le téléchargement reprend où il s'était arrêté). |
| « Ollama tourne encore » | `sudo systemctl stop ollama` (l'installation le désactive normalement). |
| Réponses lentes | `free -h` pendant un appel : si `Swap` est utilisé, voir « Mémoire de la carte ». |
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
  cerveau.py           le cerveau qui écoute : llama.cpp (lancé par AlloIA) ou Ollama
  ollama_audio.py      Ollama : conversation avec audio, transcription, synthèse vocale
  phrases.py           découpe du micro en phrases (Silero)
  voix.py              voix Phonon ou Ollama
  audio.py             le combiné : carte son USB (ou simulé pour les tests)
  doc.py               rédaction et enregistrement de la documentation
  reglages.py          lecture des réglages
agent/                 briques partagées avec le directeur de mémoire (Phonon, Ollama…)
jetson/                installation, lancement, reprise des réglages
tests/                 tests automatiques (combiné simulé, faux llama.cpp et faux Ollama)
```

**Licences** : le code est libre. Phonon 40M est sous licence CC BY-NC 4.0 (usage non
commercial) ; vérifiez les licences des autres modèles sur leurs pages.
