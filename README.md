# AlloIA : le téléphone de documentation du FabLab

Un maker vient de finir son projet. Il décroche le combiné posé près de la Jetson, et AlloIA
lui pose des questions, avec un vrai intérêt, pour lui faire raconter **ce qu'il a fabriqué, comment, avec quels matériaux et quelles machines,
les difficultés rencontrées et ses conseils**. Quand il raccroche, AlloIA rédige toute seule
la **documentation du projet** dans un fichier Markdown.

- **100 % local et gratuit** : tout tourne sur la carte Jetson, rien ne part sur internet.
- **Seul le texte est gardé** (documentation et transcription), jamais l'enregistrement audio.
  AlloIA le dit à chaque appel.

> Ce dépôt contient aussi l'agent « directeur de mémoire » pour Windows, dont AlloIA
> réutilise les briques : voir [README_directeur_memoire.md](README_directeur_memoire.md).

---

## Le matériel

- Une **Jetson Orin Nano** (8 Go) avec JetPack 6.
- Une **carte son USB** avec un **micro** (combiné, prise jack) et un **haut-parleur**.
- Une connexion internet pour l'installation et le premier lancement seulement.

## Comment ça marche

| Rôle | Logiciel | Où il tourne |
|---|---|---|
| Les oreilles | [Whisper](https://github.com/SYSTRAN/faster-whisper) `small` + détecteur de voix Silero | processeur de la Jetson |
| Le cerveau | [Ministral 3](https://ollama.com/library/ministral-3) (3B) de Mistral, dans [Ollama](https://ollama.com) | puce graphique de la Jetson |
| La bouche | [Phonon](https://huggingface.co/gradium/phonon-40m-fr) `phonon-40m-fr` de Gradium | processeur de la Jetson |
| Le chef d'orchestre | [Gradbot](https://github.com/gradium-ai/gradbot) : tours de parole, relances | processeur de la Jetson |

Déroulement d'un appel :

1. **En attente** : AlloIA écoute le micro. Dès qu'on parle (« Allô ? »), l'appel commence.
2. **L'appel** : « Allô ! Ici AlloIA… ». Une question à la fois. Le micro est coupé pendant
   qu'AlloIA parle, pour qu'elle ne s'entende pas elle-même.
3. **La fin** : quand tout est raconté ou que la personne dit au revoir, AlloIA remercie et
   raccroche. L'appel se termine aussi après une minute de silence, ou après 25 minutes.
4. **La documentation** est rédigée dans `~/AlloIA/docs/`, par exemple
   `2026-11-10_14h05_lampe-en-contreplaque.md`. Puis AlloIA se remet en attente.

---

## Installation (une seule fois)

Dans un terminal sur la Jetson :

```bash
git clone -b AlloIA https://github.com/Ax3l-ybc/agentIA.git ~/AlloIA-code
```

```bash
cd ~/AlloIA-code
```

```bash
./jetson/installer.sh
```

Le script installe les outils nécessaires, **compile Gradbot** (10 à 20 minutes la première fois),
installe Whisper et Phonon, et règle Ollama. Il demande votre mot de passe pour les paquets
du système. Il peut être relancé sans risque.

### La voix : jeton Hugging Face

La voix française Phonon est gratuite (usage non commercial) mais demande d'accepter ses
conditions : sur [huggingface.co/gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr),
acceptez les conditions, puis créez un jeton de type « Read » sur
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Copiez-le dans
`alloia.yaml`, à la ligne `jeton_huggingface:`.

### Le son

Branchez la carte son USB, puis listez les cartes reconnues :

```bash
./jetson/lancer.sh --lister-audio
```

Repérez le nom de votre carte USB (par exemple « USB Audio Device ») et mettez un morceau de
ce nom dans `alloia.yaml`, aux lignes `entree:` et `sortie:` de la partie `audio:`. Puis testez :

```bash
./jetson/lancer.sh --test-audio
```

Vous devez entendre un bip, puis le terminal indique si le micro capte votre voix. Si le micro
est trop faible, montez son volume avec `alsamixer` (touche F6 pour choisir la carte USB).

## Lancer AlloIA

```bash
./jetson/lancer.sh
```

Au premier lancement, AlloIA télécharge le cerveau (3 Go), l'écoute et la voix. Quand le
terminal affiche **« En attente d'un appel »**, décrochez et dites « Allô ? ».
Pour les essais, la touche **Entrée** décroche et raccroche aussi. Pour arrêter : **Ctrl + C**.

---

## Personnaliser

- **`alloia_conduite.md`** : la personnalité d'AlloIA et ce qu'elle doit faire raconter.
  Relu à chaque appel : une modification s'applique tout de suite.
- **`alloia_modele_doc.md`** : le modèle de la documentation produite.
- **`alloia.yaml`** : les réglages (voix, durée des silences, dossier des docs, carte son…).

## Mémoire de la carte

La Jetson Orin Nano a 8 Go partagés entre le processeur et la puce graphique. AlloIA en
utilise environ 4,5 Go. Si la carte manque de mémoire (lenteurs, arrêt brutal), désactivez
le bureau graphique, qui occupe environ 1 Go :

```bash
sudo systemctl set-default multi-user.target
```

Puis redémarrez. Pour retrouver le bureau : `sudo systemctl set-default graphical.target`.

## En cas de problème

| Ce que vous voyez | Que faire |
|---|---|
| « Aucun micro ou haut-parleur utilisable » | Vérifiez que la carte son USB apparaît avec `arecord -l` et `aplay -l`, puis son nom dans `alloia.yaml`. |
| « Ollama ne répond pas » | `sudo systemctl restart ollama`, puis relancez. |
| « Téléchargement refusé par Hugging Face » | Acceptez les conditions de Phonon et vérifiez le jeton dans `alloia.yaml`. |
| AlloIA se répond à elle-même | Gardez `couper_micro_quand_agent_parle: true`, et éloignez le micro du haut-parleur. |
| AlloIA coupe la personne trop tôt | Augmentez `pause_fin_de_phrase_s` (par exemple `1.3`) dans `alloia.yaml`. |
| L'appel démarre tout seul (bruit de l'atelier) | Augmentez `parole_pour_decrocher_s` (par exemple `1.2`). |

Envoyez les messages du terminal pour qu'on corrige ensemble.

## Pour les développeurs

```
alloia.yaml            réglages
alloia_conduite.md     personnalité et consignes de l'agent
alloia_modele_doc.md   modèle de la documentation
alloia/
  __main__.py          boucle : attente → appel → documentation
  appel.py             un appel (Gradbot, fin d'appel, transcription)
  audio.py             le combiné : carte son USB (ou simulé pour les tests)
  doc.py               rédaction et enregistrement de la documentation
  reglages.py          lecture d'alloia.yaml
agent/                 briques partagées : serveur vocal local (protocole Gradium),
                       Whisper, Phonon, Ollama
jetson/                installation et lancement sur la Jetson
tests/                 tests automatiques (combiné simulé)
```

**Licences** : le code et Gradbot sont libres. Phonon 40M est sous licence CC BY-NC 4.0
(usage non commercial) ; vérifiez les licences des autres modèles sur leurs pages.
