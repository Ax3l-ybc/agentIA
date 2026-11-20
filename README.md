# Mon Assistant : votre assistant personnel, à l'écrit ou à la voix

Une page comme ChatGPT ou Claude, mais **qui tourne entièrement sur votre ordinateur** :

- **Écrivez-lui**, ou **cliquez sur le micro pour lui parler** : il vous répond à voix haute,
  en français. Il peut aussi lire à voix haute ses réponses écrites.
- **Créez vos propres assistants**, sans toucher au code : un nom, des consignes (son rôle,
  son ton, ce qu'il doit faire), une voix, et si vous voulez, un dossier de documents qu'il
  peut consulter (vos notes, vos cours…). Deux exemples sont fournis : *Assistant du
  quotidien* et *Coach d'écriture*.
- **Joignez des documents** (PDF, Word, texte) : « résume ce rapport », « réponds à ce mail »…
  Et récupérez les réponses : **copier** en un clic, ou **enregistrer en Word**.
- **Comptes rendus de réunion** : enregistrez une réunion (ou importez un enregistrement) ;
  l'assistant la retranscrit et rédige le compte rendu (décisions, actions, questions en suspens).
- **Vos conversations sont gardées** sur votre ordinateur : vous les retrouvez dans la colonne
  de gauche, pour les reprendre plus tard.
- **Gratuit et 100 % local** : aucun abonnement, rien n'est envoyé sur internet (la connexion
  sert seulement à l'installation). Vos documents sont lus, **jamais modifiés**.

---

## Ce qu'il vous faut

- Un PC sous **Windows 10 ou 11**, avec **16 Go de mémoire vive** conseillés (8 Go au minimum).
- Environ **8 Go d'espace libre** sur le disque.
- Pour parler à l'assistant : un **micro** (un **casque** est conseillé).
- **Pas besoin de carte graphique** : tout tourne sur le processeur. Les réponses écrites
  commencent à s'afficher en une seconde environ ; à la voix, l'assistant répond quelques
  secondes après la fin de votre phrase.

---

## Étape 1 : télécharger

1. Sur la page GitHub du projet, choisissez la branche **assistant-personnel**, cliquez sur le
   bouton vert **Code**, puis sur **Download ZIP**.
   (Lien direct : <https://github.com/Ax3l-ybc/agentIA/archive/refs/heads/assistant-personnel.zip>)
2. Faites un clic droit sur le fichier téléchargé, puis **Extraire tout…**
3. Placez le dossier obtenu où vous voulez, par exemple dans `Documents`.

## Étape 2 : installer (une seule fois)

1. Ouvrez le dossier, puis **double-cliquez sur `installer.bat`**.
2. Si Windows affiche « Windows a protégé votre ordinateur », cliquez sur
   **Informations complémentaires**, puis sur **Exécuter quand même**.
3. Attendez le message **« Installation terminée ! »** (10 à 30 minutes).

À la fin, un raccourci **« Meilleur collègue »** est créé sur votre Bureau : c'est lui qui
lance l'assistant. (S'il a disparu, double-cliquez sur `creer_raccourci.bat` dans le dossier
du programme pour le recréer.)

L'installation met en place Python, **llama.cpp** (le programme qui fait tourner le « cerveau »)
et télécharge le cerveau lui-même, **Gemma 4 E2B** de Google (environ 3,8 Go, rangé dans
`C:\Users\VotreNom\MonAssistant\modeles`). Si vous avez déjà installé le *Directeur de
mémoire*, Gemma 4 est réutilisé : rien n'est téléchargé une deuxième fois.

## Étape 3 : la voix (facultatif, gratuit, une seule fois)

Sans cette étape, l'assistant marche **à l'écrit**. Pour qu'il parle, la voix française
(Phonon de Gradium, gratuite pour un usage personnel) demande d'accepter ses conditions sur
le site Hugging Face :

1. Créez un compte gratuit sur [huggingface.co](https://huggingface.co/join).
2. Ouvrez la page [huggingface.co/gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr)
   et cliquez sur le bouton pour **accepter les conditions**.
3. Ouvrez [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), cliquez sur
   **Create new token**, type **Read**, donnez-lui un nom, puis **copiez le jeton** (il commence par `hf_`).
4. Ouvrez **`config.yaml`** avec le Bloc-notes et collez votre jeton entre les apostrophes :
   `jeton_huggingface: 'hf_...'`. Enregistrez.

## Étape 4 : lancer

1. **Double-cliquez sur « Meilleur collègue » sur votre Bureau** (ou sur `lancer.bat` dans le
   dossier du programme). Une fenêtre noire s'ouvre : laissez-la ouverte tant que vous
   utilisez l'assistant (pour arrêter, fermez-la).
2. Après une à deux minutes, la fenêtre affiche **« Prêt ! »** et la page s'ouvre dans votre
   navigateur (en général <http://127.0.0.1:8000>).

---

## Utiliser l'assistant

- **Écrire** : tapez votre message en bas, puis **Entrée** (Maj + Entrée pour aller à la ligne).
  La réponse s'affiche au fur et à mesure. Le bouton carré **arrête** une réponse en cours.
- **Parler** : cliquez sur le **micro** (autorisez-le si le navigateur le demande). Parlez
  normalement, puis marquez une courte pause : l'assistant répond à voix haute. Le micro est
  coupé pendant qu'il parle. Cliquez à nouveau sur le micro pour arrêter.
- **Lire à voix haute** : activez-le à gauche de la zone de saisie pour entendre aussi les
  réponses écrites.
- **Changer d'assistant** : cliquez sur un assistant dans la colonne de gauche : une nouvelle
  conversation commence avec lui.
- **Retrouver une conversation** : elles sont classées par date dans la colonne de gauche.
  Au survol, le crayon la renomme et la poubelle la supprime.

## Joindre un document

Cliquez sur le **trombone** à gauche de la zone de saisie, ou **glissez-déposez** vos fichiers
dans la page. Formats acceptés : **PDF**, **Word (.docx)** et texte (.txt, .md, .csv…). Écrivez
ensuite votre demande (« résume ce document », « quelles sont les dates importantes ? »,
« améliore ce mail »…) et envoyez.

- Le document est lu **sur votre ordinateur**. Seul son texte est gardé, dans la conversation :
  vous pouvez continuer à poser des questions dessus.
- Un document long (au-delà d'une dizaine de pages) n'est lu qu'en partie : son début. Le
  trombone l'indique (« long : début seulement »). Pour un long rapport, joignez plutôt le
  passage qui vous intéresse.
- Un PDF **scanné** (une image de papier) ne contient pas de texte : il ne peut pas être lu.
- Un fichier Word ancien (.doc) : ouvrez-le dans Word et enregistrez-le en .docx.

## Copier une réponse, l'enregistrer en Word

Sous chaque réponse : **Copier** (pour la coller dans un mail, par exemple) et **Word** (un
document Word est enregistré dans votre dossier Téléchargements). En haut à droite,
**Exporter en Word** enregistre toute la conversation.

## Compte rendu de réunion

Cliquez sur **Compte rendu de réunion**, en haut de la colonne de gauche :

- **Enregistrer** : posez l'ordinateur au milieu de la table et lancez l'enregistrement. À la
  fin, cliquez sur **Terminer et rédiger le compte rendu**. La transcription avance déjà pendant
  la réunion.
- **Importer un fichier audio** : un enregistrement déjà fait, par exemple avec un téléphone
  (mp3, m4a, wav, ogg…).

L'assistant retranscrit la réunion, puis rédige le compte rendu : **résumé, points abordés,
décisions, actions à mener (qui, quoi, quand), questions en suspens**. Le compte rendu s'ouvre
comme une conversation : vous pouvez poser des questions sur la réunion (« qui doit faire
quoi ? ») et l'exporter en Word. La transcription complète est aussi gardée en texte dans
`C:\Users\VotreNom\MonAssistant\reunions`. L'enregistrement sonore, lui, n'est pas gardé.

À savoir :
- **Le temps** : sur un simple processeur, comptez souvent entre un tiers et la moitié de la
  durée de la réunion pour la transcription, plus une à deux minutes pour la rédaction. Pendant
  ce temps, l'assistant ne répond pas aux conversations.
- **La qualité** du compte rendu dépend surtout de celle de l'enregistrement : un micro proche,
  peu de bruit, une personne qui parle à la fois. Relisez toujours le compte rendu.
- Les intervenants ne sont **pas reconnus** : un nom n'apparaît que s'il est prononcé.

## Créer ou modifier un assistant

Dans la colonne de gauche, cliquez sur **Créer un assistant**, ou passez la souris sur un
assistant et cliquez sur le **crayon**. Vous réglez :

- **Nom** et **description** ;
- **Consignes** : décrivez en français, comme à une personne, son rôle, son ton (tutoiement ou
  vouvoiement), ce qu'il doit faire et éviter. Les consignes s'appliquent à chaque nouvelle
  conversation ;
- **Voix** : Hélène, Tiphaine, Orianne, Ninon, Noémie, Coralie, Maëlys, Solène, Lilou
  (féminines) ; Arnaud, Lucien, Augustin, Jules, Baptiste, Marius (masculines) ;
- **Dossier de documents** (facultatif) : un dossier de fichiers `.md` ou `.txt` (par exemple
  un coffre Obsidian). L'assistant peut les consulter pour répondre et cite leur nom ; il les
  lit **sans jamais rien y modifier**.

Chaque assistant est aussi un simple fichier texte dans le dossier **`assistants`** du
programme : vous pouvez l'ouvrir et le modifier avec le Bloc-notes, le copier pour en faire un
autre, ou le partager. Un assistant supprimé depuis la page est rangé dans `assistants\corbeille`
(rien n'est vraiment effacé).

**Conseils pour de bonnes consignes** (Gemma 4 E2B est un petit modèle) : soyez concret et
positif (« pose une seule question à la fois » plutôt que « ne pose pas plusieurs questions »),
donnez le rôle dès la première phrase, et restez court.

---

## Problèmes fréquents

| Ce que vous voyez | Que faire |
|---|---|
| « La voix n'a pas pu être téléchargée » | Refaites l'étape 3 (accepter les conditions, recopier le jeton). En attendant, l'assistant marche à l'écrit. |
| « config.yaml est mal écrit » | Vérifiez les apostrophes `' '` autour des textes et les espaces en début de ligne. |
| « llama.cpp est introuvable » | Relancez `installer.bat`, puis fermez et rouvrez `lancer.bat`. Voir aussi plus bas. |
| « Fichier introuvable pour le modèle » | Relancez `installer.bat` : le téléchargement reprend où il s'était arrêté. |
| « Le dossier de documents n'existe pas » | Vérifiez le chemin : copiez-le depuis la barre d'adresse de l'Explorateur Windows. |
| L'assistant ne vous entend pas | Autorisez le micro dans le navigateur (icône à gauche de l'adresse) et vérifiez le micro choisi dans Windows. |
| Il vous coupe pendant que vous réfléchissez | Augmentez `pause_fin_de_phrase_s` (par exemple `1.8`) dans `config.yaml`. |
| Les réponses sont lentes | Fermez les programmes gourmands. Dans `config.yaml`, `modele: 'gemma-4-E2B-it-Q3_K_M.gguf'` est plus rapide (à télécharger avec `.venv\Scripts\python.exe -m assistant.telecharger gemma-4-E2B-it-Q3_K_M.gguf`). Avec une carte graphique, essayez `couches_gpu: 99`. |
| La page ne s'ouvre pas toute seule | Copiez dans Chrome ou Edge l'adresse de la ligne « Prêt ! », sans fermer la fenêtre noire. |

### Si llama.cpp ne s'installe pas

`installer.bat` l'installe avec la commande Windows `winget install llama.cpp`. Si `winget`
n'est pas disponible, téléchargez llama.cpp pour Windows sur
[github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
(fichier dont le nom contient `bin-win-cpu-x64.zip`), extrayez-le, puis indiquez l'emplacement
de `llama-server.exe` dans `config.yaml`, partie `cerveau:` : `serveur: 'C:\chemin\vers\llama-server.exe'`.

### Mettre à jour

Téléchargez le nouveau ZIP et copiez son contenu par-dessus l'ancien dossier, **sauf**
`config.yaml` (votre jeton) et le dossier `assistants` (vos assistants). Vos conversations et
le modèle, rangés dans `C:\Users\VotreNom\MonAssistant`, ne bougent pas.

---

## Pour les curieux : comment ça marche

- **Le cerveau** : [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) de Google,
  dans [llama.cpp](https://github.com/ggml-org/llama.cpp). Il lit vos messages écrits et
  **écoute directement votre voix**, sans transcription préalable.
- **La voix** : [Phonon](https://huggingface.co/gradium/phonon-40m-fr) de Gradium ;
  **le détecteur de parole** : Silero, qui repère la fin de vos phrases.
- **Pour répondre vite** (le même moteur que le *Directeur de mémoire*) : llama.cpp garde la
  conversation en mémoire d'un tour à l'autre ; à la voix, la réponse est préparée dès 0,4 s de
  silence et la voix commence dès la première virgule.

```
config.yaml           réglages techniques (cerveau, voix, page web)
assistants/           un fichier par assistant (nom, voix, documents, consignes)
installer.bat         installation (Python, llama.cpp, Gemma 4, raccourci sur le Bureau)
lancer.bat            lancement (c'est ce que lance le raccourci « Meilleur collègue »)
creer_raccourci.bat   recrée le raccourci « Meilleur collègue » sur le Bureau
assistant/
  __main__.py         démarrage : cerveau, voix, page web
  application.py      le serveur de la page (assistants, conversations)
  session.py          une conversation, à l'écrit ou à la voix
  profils.py          lecture et écriture des fichiers d'assistants
  conversations.py    l'historique des conversations
  documents.py        les documents d'un assistant (lecture seule)
  fichiers.py         lecture des fichiers joints (PDF, Word, texte) et export Word
  reunion.py          comptes rendus de réunion (transcription par morceaux, puis rédaction)
  raccourci.ps1       création du raccourci sur le Bureau
  cerveau.py          Gemma 4 dans llama.cpp
  moteurs.py          la voix (Phonon) et le détecteur de parole (Silero)
  phrases.py          découpe de votre voix en phrases
  static/index.html   la page
tests/                tests automatiques (faux cerveau, voix simulée)
```

**Licences** : le code est libre. Phonon 40M est sous licence CC BY-NC 4.0 (usage non
commercial) ; Gemma 4 suit les conditions d'utilisation de Google.
