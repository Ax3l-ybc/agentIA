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
- **Chacun son profil** : qui vous êtes, vos besoins particuliers, vos préférences. Tous les
  assistants en tiennent compte, et chaque personne retrouve ses propres conversations.
- **Un cerveau adapté au PC** : Gemma 4 E2B par défaut, E4B plus fin, et sur un PC puissant un
  second modèle plus grand pour rédiger ; la carte graphique est utilisée si le PC en a une.
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

## Étape 3 : la voix (rien à faire)

La voix française **Kokoro** (voix « Siwis ») est installée avec le reste : elle est libre,
utilisable aussi en entreprise, et ne demande aucun compte.

**En option, la voix Phonon** (Gradium) propose 15 voix françaises, féminines et masculines,
mais seulement pour un **usage personnel, non commercial**, et demande un compte Hugging Face :

1. Créez un compte gratuit sur [huggingface.co](https://huggingface.co/join).
2. Ouvrez la page [huggingface.co/gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr)
   et cliquez sur le bouton pour **accepter les conditions**.
3. Ouvrez [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), cliquez sur
   **Create new token**, type **Read**, donnez-lui un nom, puis **copiez le jeton** (il commence par `hf_`).
4. Ouvrez **`config.yaml`** avec le Bloc-notes, collez votre jeton entre les apostrophes
   (`jeton_huggingface: 'hf_...'`) et choisissez Phonon dans la partie `voix:` (`moteur: 'phonon'`).
   Enregistrez et relancez. Avec Phonon, la voix se choisit pour chaque assistant, dans la page.

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

## Votre profil

En bas à gauche, cliquez sur **Mon profil** (ou sur « Présentez-vous… » dans la page d'accueil)
et décrivez-vous : prénom, métier ou rôle, qui vous êtes, **vos besoins particuliers** (par
exemple « je suis dyslexique : phrases courtes, sans jargon ») et **vos préférences** (« vouvoiement,
réponses courtes avec un exemple concret »…). Tous les assistants en tiennent compte, et
vos préférences passent avant leur ton habituel. Le profil s'applique aux nouvelles conversations.

**Plusieurs personnes sur le même PC** : cliquez sur votre prénom en bas à gauche, puis
**Ajouter une personne**. On passe de l'une à l'autre au même endroit ; chacun retrouve ses
propres conversations. Les profils restent sur le PC
(`C:\Users\VotreNom\MonAssistant\utilisateurs`) et ne sont **pas protégés par un mot de passe** :
toute personne qui utilise ce PC peut les voir.

## Le cerveau : choisir selon votre PC

En bas à gauche, le bouton **Cerveau** montre le PC (mémoire, carte graphique) et les modèles
possibles, avec celui qui est **conseillé pour ce PC** :

| Modèle | Rôle | Taille | Pour quel PC |
|---|---|---|---|
| **Gemma 4 E2B** | converse, écoute la voix | 4 Go | tous ; le bon choix sans carte graphique |
| **Gemma 4 E4B** | converse, écoute la voix | 6 Go | plus fin ; 32 Go de mémoire, ou une carte graphique de 8 Go |
| **Gemma 4 12B** | rédige (n'écoute pas la voix) | 7 Go | une carte graphique de 16 Go |
| **Gemma 4 26B-A4B** | rédige | 17 Go | 48 Go de mémoire, ou une grosse carte graphique |
| **Gemma 4 31B** | rédige, le plus fin | 18 Go | une carte graphique de 24 Go |

- **Le modèle « pour rédiger » est facultatif.** S'il est choisi, il répond aux messages écrits et
  rédige les comptes rendus ; la voix et la transcription restent au modèle qui écoute.
- **Utiliser la carte graphique** : à cocher si le PC en a une (NVIDIA, ou AMD Radeon RX / Intel Arc).
  C'est bien plus rapide quand le modèle tient dans sa mémoire. `installer.bat` prépare déjà la
  bonne version de llama.cpp s'il détecte une carte ; sinon elle est téléchargée en appliquant.
- Cliquez sur **Appliquer** : les fichiers qui manquent sont téléchargés (la progression s'affiche),
  puis le cerveau redémarre en une à deux minutes, sans fermer le programme. Si le nouveau choix ne
  démarre pas (mémoire insuffisante…), l'ancien est remis en route et la page explique pourquoi.
- Sur un simple processeur, E4B répond environ deux fois plus lentement qu'E2B (premier mot en
  3 s au lieu de 1,5 s, mesuré sur un PC à 4 cœurs), mais suit mieux les consignes et le profil.

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
| « La voix n'a pas pu être chargée » | Vérifiez votre connexion internet au premier lancement (la voix Kokoro, environ 180 Mo, se télécharge une fois), puis relancez. Avec Phonon : refaites l'étape 3. En attendant, l'assistant marche à l'écrit. |
| « config.yaml est mal écrit » | Vérifiez les apostrophes `' '` autour des textes et les espaces en début de ligne. |
| « llama.cpp est introuvable » | Relancez `installer.bat`, puis fermez et rouvrez `lancer.bat`. Voir aussi plus bas. |
| « Fichier introuvable pour le modèle » | Relancez `installer.bat` : le téléchargement reprend où il s'était arrêté. |
| « Le dossier de documents n'existe pas » | Vérifiez le chemin : copiez-le depuis la barre d'adresse de l'Explorateur Windows. |
| L'assistant ne vous entend pas | Autorisez le micro dans le navigateur (icône à gauche de l'adresse) et vérifiez le micro choisi dans Windows. |
| Il vous coupe pendant que vous réfléchissez | Augmentez `pause_fin_de_phrase_s` (par exemple `1.8`) dans `config.yaml`. |
| Les réponses sont lentes | Fermez les programmes gourmands. Bouton **Cerveau** : revenez à Gemma 4 E2B, sans modèle de rédaction ; avec une carte graphique, cochez « Utiliser la carte graphique ». |
| Le cerveau choisi ne démarre plus | Au lancement, l'assistant revient tout seul à Gemma 4 E2B. Pour tout remettre à zéro : supprimez `C:\Users\VotreNom\MonAssistant\cerveau.json`. |
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
- **La voix** : [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (version ONNX), voix
  française « Siwis » ; ou, en option, [Phonon](https://huggingface.co/gradium/phonon-40m-fr) de Gradium ;
  **le détecteur de parole** : Silero, qui repère la fin de vos phrases.
- **Pour répondre vite** (le même moteur que le *Directeur de mémoire*) : llama.cpp garde la
  conversation en mémoire d'un tour à l'autre ; à la voix, la réponse est préparée dès 0,4 s de
  silence et la voix commence dès la première virgule.
- **Pour que la voix démarre vite** : Kokoro fabrique chaque morceau de phrase d'un coup, en un temps
  proportionnel à sa longueur. Le premier morceau est donc court (première virgule, ou cinq mots),
  puis les morceaux grandissent ; les débuts de réponse fréquents (« Bien sûr, », « Bonjour ! », la
  salutation avec votre prénom…) sont préparés à l'avance et sortent aussitôt ; et llama.cpp tourne
  avec une priorité un peu plus basse, pour que la voix passe d'abord quand les deux travaillent.

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
  utilisateurs.py     les profils des personnes (qui je suis, besoins, préférences)
  modeles.py          les modèles possibles, le conseil selon le PC, le téléchargement
  cerveaux.py         le cerveau d'écoute et le cerveau de rédaction, changement sans redémarrer
  materiel.py         mémoire vive et carte graphique du PC
  llamacpp_gpu.py     llama.cpp pour la carte graphique (CUDA ou Vulkan)
  fichiers.py         lecture des fichiers joints (PDF, Word, texte) et export Word
  reunion.py          comptes rendus de réunion (transcription par morceaux, puis rédaction)
  raccourci.ps1       création du raccourci sur le Bureau
  cerveau.py          Gemma 4 dans llama.cpp
  moteurs.py          la voix (Phonon) et le détecteur de parole (Silero)
  phrases.py          découpe de votre voix en phrases
  static/index.html   la page
tests/                tests automatiques (faux cerveau, voix simulée)
```

**Licences et crédits** : Gemma 4 (Google) et Kokoro-82M (hexgrad) sont sous licence
Apache 2.0. La voix française de Kokoro, « Siwis », a été entraînée sur le corpus SIWIS
(Université d'Édimbourg, licence CC BY 4.0). Phonon 40M (Gradium, en option) est sous licence
CC BY-NC 4.0 : usage non commercial seulement. llama.cpp, Silero, espeak-ng (via espeakng-loader)
et les autres bibliothèques sont des logiciels libres.
