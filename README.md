# Directeur de mémoire : un agent vocal qui vous fait travailler votre mémoire

Vous parlez, il vous répond **à voix haute, en français**. Il lit les notes de votre
mémoire dans Obsidian et vous interroge comme un **directeur de mémoire exigeant** :
il relève ce qui reste flou et vous propose de nouvelles questions.
Quand vous le lui demandez, il écrit une **fiche récapitulative de la séance**.

- **Gratuit et 100 % local** : tout tourne sur votre ordinateur. Il ne faut aucun abonnement,
  et rien n'est envoyé sur internet (la connexion sert seulement au premier lancement).
- **Vos notes ne sont jamais modifiées.** L'agent les lit directement dans votre coffre,
  sans en faire de copie. Il écrit seulement de nouvelles fiches, dans un dossier séparé.

---

## Ce qu'il vous faut

- Un PC sous **Windows 10 ou 11**, avec **16 Go de mémoire vive** (RAM) conseillés (8 Go au minimum).
- Environ **8 Go d'espace libre** sur le disque.
- Un **micro**. Un **casque** est conseillé ; sans casque, le micro est coupé pendant que le
  directeur parle, pour qu'il ne s'entende pas lui-même.
- **Pas besoin de carte graphique** : tout tourne sur le processeur. Le directeur répond
  quelques secondes après la fin de votre phrase (plus vite sur un PC récent).
- Une connexion internet **pour l'installation et le premier lancement seulement**.

---

## Étape 1 : télécharger l'agent

1. Sur la page GitHub de ce projet, cliquez sur le bouton vert **Code**, puis sur **Download ZIP**.
2. Faites un clic droit sur le fichier téléchargé, puis cliquez sur **Extraire tout…**
3. Placez le dossier obtenu où vous voulez, par exemple dans `Documents`.

## Étape 2 : installer (une seule fois)

1. Ouvrez le dossier, puis **double-cliquez sur `installer.bat`**.
2. Si Windows affiche « Windows a protégé votre ordinateur », cliquez sur
   **Informations complémentaires**, puis sur **Exécuter quand même**.
3. Attendez le message **« Installation terminée ! »**. Cela prend 10 à 30 minutes.

L'installation met en place Python, les modules nécessaires, **llama.cpp** (le programme qui
fait tourner le « cerveau ») et télécharge le cerveau lui-même, **Gemma 4 E2B** (environ 3,8 Go,
rangé dans `C:\Users\VotreNom\DirecteurMemoire\modeles`). **Ollama n'est plus utilisé** :
vous pouvez le quitter (icône près de l'horloge) ou le désinstaller.

**Mise à jour depuis une version précédente** : remplacez les fichiers par ceux du nouveau ZIP,
**remettez votre `config.yaml`** (vos dossiers, votre sujet et votre jeton restent valables ;
les anciens réglages Ollama et Whisper sont simplement ignorés), puis relancez `installer.bat`.

## Étape 3 : obtenir l'accès à la voix (gratuit, une seule fois)

La voix française (Phonon de Gradium) est gratuite pour un usage personnel. Elle demande
toutefois d'accepter ses conditions sur le site Hugging Face.

1. Créez un compte gratuit sur [huggingface.co](https://huggingface.co/join).
2. Ouvrez la page [huggingface.co/gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr),
   puis cliquez sur le bouton pour **accepter les conditions** et accéder au modèle.
3. Ouvrez [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), cliquez sur
   **Create new token**, choisissez le type **Read**, donnez-lui un nom (par exemple `agent`),
   puis **copiez le jeton** (une suite de caractères qui commence par `hf_`).

## Étape 4 : régler l'agent

Faites un clic droit sur **`config.yaml`**, puis cliquez sur **Ouvrir avec**, puis sur **Bloc-notes**.
Modifiez seulement ces lignes :

```yaml
dossier_notes: 'C:\Users\VotreNom\Documents\Obsidian\MonCoffre\Mémoire'
dossier_recaps: 'C:\Users\VotreNom\Documents\Obsidian\MonCoffre\Recaps'
sujet_memoire: 'Les tiers-lieux et le travail en milieu rural'
jeton_huggingface: 'hf_xxxxxxxxxxxxxxxxxxxx'
```

- **`dossier_notes`** : le dossier de votre coffre Obsidian qui contient les notes du mémoire.
- **`dossier_recaps`** : le dossier où l'agent rangera ses fiches. Il est créé automatiquement.
  Il doit se trouver **en dehors** du dossier des notes. Le mettre dans votre coffre Obsidian
  est une bonne idée : les fiches apparaîtront dans Obsidian.
- **`sujet_memoire`** : votre sujet en une phrase. Ce réglage est facultatif.
- **`jeton_huggingface`** : le jeton copié à l'étape 3.

> **Astuce pour copier le chemin d'un dossier** : dans l'Explorateur de fichiers, faites
> **Maj + clic droit** sur le dossier, puis cliquez sur **Copier en tant que chemin d'accès**.
> Collez le chemin, puis **remplacez les guillemets `"` par des apostrophes `'`**.

Enregistrez le fichier (**Ctrl + S**) et fermez le Bloc-notes.

## Étape 5 : lancer une séance

1. **Double-cliquez sur `lancer.bat`.** Une fenêtre noire s'ouvre : laissez-la ouverte
   pendant toute la séance.
2. Le directeur **démarre son cerveau, puis lit vos consignes, vos notes et vos fiches**
   (une à deux minutes). C'est fait une fois au lancement : ensuite, chaque séance commence
   tout de suite.
3. Quand la fenêtre affiche **« Prêt ! »**, la page de l'agent s'ouvre dans votre navigateur.
   L'adresse est affichée dans la fenêtre, en général <http://127.0.0.1:8000>.
4. Cliquez sur le **bouton micro** et **autorisez l'accès au micro** si le navigateur le demande.
5. Le directeur vous salue et vous pose une première question. **Répondez simplement à voix haute**,
   puis marquez une courte pause : c'est ce qui lui indique que vous avez fini votre phrase.
   Pendant qu'il parle, il ne vous entend pas : attendez « Je vous écoute ».

### Obtenir la fiche de séance

Dites par exemple **« slash fiche »**, **« fais-moi la fiche »** ou **« on termine »**, ou cliquez
sur le bouton **Rédiger la fiche de séance**. La fiche suit le modèle de votre mode de conduite
(voir ci-dessous). Elle est enregistrée dans votre dossier de fiches sous un nom comme
`2026-11-05_fiche_role-des-sap.md`. Au début de la séance suivante, l'agent relit les fiches les
plus récentes pour reprendre le fil.

### Changer le comportement de l'agent : le « mode de conduite »

Le fichier **`mode_de_conduite.md`** décrit comment l'agent se comporte : son rôle, son ton,
ce que vous attendez de lui, ses règles et le modèle de la fiche. Ouvrez-le avec le Bloc-notes,
modifiez-le comme vous voulez et enregistrez. Il est relu **au début de chaque séance** : inutile
de relancer l'agent, cliquez simplement de nouveau sur le micro.

### Arrêter

Cliquez de nouveau sur le bouton (carré rouge) pour finir la séance.
Pour quitter complètement, **fermez la fenêtre noire**.

---

## Problèmes fréquents

| Ce que vous voyez | Que faire |
|---|---|
| « Le dossier des notes n'existe pas » | Vérifiez le chemin `dossier_notes` dans `config.yaml`, avec ses apostrophes `' '`. |
| « config.yaml est mal écrit » | Vérifiez que chaque chemin est entre apostrophes `' '` et que vous n'avez pas supprimé d'espaces en début de ligne. |
| « Téléchargement refusé par Hugging Face » | Refaites l'étape 3 : acceptez les conditions du modèle et recopiez le jeton. |
| « llama.cpp est introuvable » | Relancez `installer.bat`, puis fermez et rouvrez `lancer.bat`. Voir aussi « Si llama.cpp ne s'installe pas » plus bas. |
| « Fichier introuvable pour le modèle » | Relancez `installer.bat` : le téléchargement de Gemma 4 reprend où il s'était arrêté. |
| L'agent ne vous entend pas | Autorisez le micro dans le navigateur (icône à gauche de l'adresse) et vérifiez le micro choisi dans Windows. |
| L'agent se répond à lui-même | Utilisez un casque, ou baissez le volume des haut-parleurs. |
| Les réponses sont lentes | La fenêtre noire affiche « réflexion : premier mot après … s ». Fermez les programmes gourmands ; dans `config.yaml`, vous pouvez essayer `modele: 'gemma-4-E2B-it-Q3_K_M.gguf'` (plus léger : à télécharger avec `.venv\Scripts\python.exe -m agent.telecharger gemma-4-E2B-it-Q3_K_M.gguf`). |
| L'agent répond alors que vous n'aviez pas fini | Une pause de 1,2 s marque la fin de votre phrase. Augmentez `pause_fin_de_phrase_s` (par exemple `1.8`) dans la partie `conversation:` de `config.yaml`. |
| La page ne s'ouvre pas toute seule | Copiez dans Chrome ou Edge l'adresse affichée sur la ligne « Prêt ! » (en général <http://127.0.0.1:8000>), sans fermer la fenêtre noire. |
| « Le port 8000 est déjà pris » | Rien à faire : l'agent prend un autre port. Fermez quand même les anciennes fenêtres noires de l'agent. |

## Réglages utiles (dans `config.yaml`)

- **`voix > nom`** : choisit une autre voix française, par exemple `nom: 'Hélène'`.
  Voix féminines : Hélène (grave, posée), Tiphaine, Orianne, Ninon, Noémie, Coralie, Maëlys, Solène, Lilou.
  Voix masculines : Arnaud (grave, direct), Lucien (posé), Augustin, Jules, Baptiste, Marius.
  La voix est téléchargée (quelques Ko) au lancement suivant.
- **`voix > vitesse`** : `-0.2` pour parler plus vite, `0.2` pour parler plus lentement.
- **`cerveau > modele`** : la version de Gemma 4 E2B. `gemma-4-E2B-it-Q4_K_M.gguf` par défaut ;
  `gemma-4-E2B-it-Q3_K_M.gguf` est un peu plus rapide et un peu moins fin.
- **`conversation > pause_fin_de_phrase_s`** : la pause qui marque la fin de votre phrase.
- **`conversation > silence_avant_relance_s`** : le nombre de secondes de silence avant que l'agent vous relance.

---

### Si llama.cpp ne s'installe pas

`installer.bat` l'installe avec la commande Windows `winget install llama.cpp`. Si `winget`
n'est pas disponible sur votre PC, téléchargez llama.cpp pour Windows sur
[github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
(fichier dont le nom contient `bin-win-cpu-x64.zip`), extrayez-le, puis indiquez l'emplacement
de `llama-server.exe` dans `config.yaml`, partie `cerveau:` :
`serveur: 'C:\chemin\vers\llama-server.exe'`.

---

## Vos notes sont protégées

- L'agent **ouvre vos notes en lecture seule**, directement là où elles sont, sans en faire de copie.
- Il n'a **aucune commande** pour modifier, renommer ou supprimer une note.
- Le dossier des recaps doit être en dehors du dossier des notes. Sinon, l'agent refuse de démarrer.
- Une fiche n'en écrase jamais une autre. Si un nom est déjà pris, l'agent ajoute « (2) ».

---

## Pour les curieux : comment ça marche

Le directeur fonctionne comme le téléphone AlloIA sur la carte Jetson :

- **les oreilles et le cerveau** : [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
  de Google, dans [llama.cpp](https://github.com/ggml-org/llama.cpp). Il **écoute directement
  votre voix** : chaque phrase lui est envoyée telle quelle, sans transcription préalable ;
- **la bouche** : [Phonon](https://huggingface.co/gradium/phonon-40m-fr) de Gradium (`phonon-40m-fr`) ;
- **le détecteur de parole** : Silero, qui repère la fin de vos phrases.

Pour que les réponses soient rapides sur un simple processeur, llama.cpp garde en mémoire
vos consignes, vos notes et vos fiches d'un tour à l'autre (il ne les relit pas à chaque
phrase). Il a deux « espaces » : l'un pour la conversation, l'autre pour retranscrire vos
phrases (pour l'historique et la fiche) et rédiger la fiche, sans effacer le premier.

Organisation des fichiers :

```
config.yaml           vos réglages
mode_de_conduite.md   le comportement de l'agent (votre prompt)
installer.bat         installation (Python, llama.cpp, Gemma 4)
lancer.bat            lancement
agent/
  __main__.py         démarrage : cerveau, voix, lecture des consignes, page web
  cerveau.py          Gemma 4 dans llama.cpp (lancé par l'agent)
  seance.py           une séance à la voix avec la page web
  application.py      les outils de l'agent (notes, fiches) et la fiche de séance
  consignes.py        le mode de conduite, l'adaptation à l'oral et la fiche
  coffre.py           la lecture des notes (lecture seule) et l'écriture des fiches
  phrases.py          découpe de votre voix en phrases (Silero)
  moteurs.py          la voix Phonon et le détecteur de parole
  telecharger.py      téléchargement de Gemma 4
  static/index.html   la page web (micro, voix, conversation)
tests/                les tests automatiques (pour les développeurs, avec pytest)
```

**Licences** : le code de ce projet est libre. Le modèle de voix
Phonon 40M est publié sous licence **CC BY-NC 4.0** : il est gratuit pour un usage
personnel, mais **pas pour un usage commercial**.
