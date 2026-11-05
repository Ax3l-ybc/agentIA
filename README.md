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

- Un PC sous **Windows 10 ou 11**, avec **16 Go de mémoire vive** (RAM) au minimum.
- Environ **15 Go d'espace libre** sur le disque.
- Un **micro**. Un **casque** est conseillé : sinon l'agent risque de s'entendre parler.
- Une **carte graphique NVIDIA** (GeForce RTX avec au moins 8 Go de mémoire, par exemple).
  **Elle est indispensable pour une conversation fluide** : sans elle, l'agent démarre, mais il
  met trop de temps à comprendre ce que vous dites.
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
3. Attendez le message **« Installation terminée ! »**. Cela prend 10 à 20 minutes.

L'installation met en place Python, les modules nécessaires et
[Ollama](https://ollama.com), le programme qui fait tourner le « cerveau » de l'agent.
Si Ollama ne s'installe pas tout seul, téléchargez-le sur
[ollama.com/download](https://ollama.com/download), installez-le, puis relancez `installer.bat`.

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
2. **Au premier lancement**, l'agent télécharge son cerveau, son écoute et sa voix
   (environ 8 Go). Cela peut prendre un moment. Les lancements suivants sont rapides.
3. Quand la fenêtre affiche **« Prêt ! »**, la page de l'agent s'ouvre dans votre navigateur
   (adresse : <http://localhost:8000>).
4. Cliquez sur le **bouton micro** et **autorisez l'accès au micro** si le navigateur le demande.
5. Le directeur vous salue et vous pose une première question. **Répondez simplement à voix haute.**
   Vous pouvez lui couper la parole.

### Obtenir la fiche de séance

Dites par exemple : **« Fais-moi la fiche de la séance »**, ou cliquez sur le bouton
**Rédiger la fiche de séance**. Après quelques secondes, un fichier nommé par exemple
`Recap 2026-11-04 15h30.md` apparaît dans votre dossier de recaps. La fiche contient un résumé,
les points travaillés, les zones floues, les nouvelles questions, les prochaines étapes et
la transcription de la séance.

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
| « Ollama ne répond pas » | Lancez **Ollama** depuis le menu Démarrer, puis relancez `lancer.bat`. |
| L'agent ne vous entend pas | Autorisez le micro dans le navigateur (icône à gauche de l'adresse) et vérifiez le micro choisi dans Windows. |
| L'agent se coupe ou se répond à lui-même | Utilisez un casque, ou laissez la case **Anti-écho** cochée. |
| Les réponses sont lentes | Si la fenêtre noire affiche « aucune carte graphique NVIDIA utilisable », c'est l'écoute qui est trop lente : il faut une carte NVIDIA. Sinon, essayez un cerveau plus léger dans `config.yaml` : `modele: 'qwen2.5:3b'`. |
| La page ne s'ouvre pas toute seule | Ouvrez <http://localhost:8000> dans votre navigateur. |

## Réglages utiles (dans `config.yaml`)

- **`voix > nom`** : choisit une autre voix. La liste des voix s'affiche dans la fenêtre noire au démarrage.
- **`voix > vitesse`** : `-0.2` pour parler plus vite, `0.2` pour parler plus lentement.
- **`ollama > modele`** : le cerveau. `qwen2.5:7b` convient à la plupart des PC.
  `mistral-small3.2` donne de meilleures réponses, mais demande un PC très puissant.
- **`conversation > silence_avant_relance_s`** : le nombre de secondes de silence avant que l'agent vous relance.

---

## Vos notes sont protégées

- L'agent **ouvre vos notes en lecture seule**, directement là où elles sont, sans en faire de copie.
- Il n'a **aucune commande** pour modifier, renommer ou supprimer une note.
- Le dossier des recaps doit être en dehors du dossier des notes. Sinon, l'agent refuse de démarrer.
- Une fiche n'en écrase jamais une autre. Si un nom est déjà pris, l'agent ajoute « (2) ».

---

## Pour les curieux : comment ça marche

L'agent est construit sur [Gradbot](https://github.com/gradium-ai/gradbot), un cadre open source
pour agents vocaux. Gradbot coordonne trois éléments :

- **les oreilles** : [Kyutai STT](https://github.com/kyutai-labs/delayed-streams-modeling)
  (`kyutai/stt-1b-en_fr`). Il transcrit en continu et détecte quand vous avez fini de parler ;
- **le cerveau** : un modèle de langage qui tourne dans [Ollama](https://ollama.com) ;
- **la bouche** : [Phonon](https://huggingface.co/gradium/phonon-40m-fr) de Gradium (`phonon-40m-fr`).

Normalement, Gradbot utilise les services en ligne de Gradium pour l'écoute et la voix.
Ici, un petit serveur local (`agent/serveur_voix.py`) parle le même langage que ces services,
mais il fait tout le travail sur votre PC avec Kyutai STT et Phonon.

Organisation des fichiers :

```
config.yaml           vos réglages
installer.bat         installation
lancer.bat            lancement
agent/
  application.py      la séance vocale, les outils de l'agent et les fiches
  consignes.py        le personnage du directeur de mémoire
  coffre.py           la lecture des notes (lecture seule) et l'écriture des fiches
  serveur_voix.py     l'écoute et la voix en local (protocole Gradium)
  moteurs.py          Kyutai STT et Phonon
  ollama.py           la communication avec Ollama
  static/index.html   la page web
tests/                les tests automatiques (pour les développeurs, avec pytest)
```

**Licences** : le code de ce projet et Gradbot sont libres. Le modèle de voix
Phonon 40M est publié sous licence **CC BY-NC 4.0** : il est gratuit pour un usage
personnel, mais **pas pour un usage commercial**.
