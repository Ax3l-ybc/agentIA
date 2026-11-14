# AlloIA : pistes à retravailler plus tard

Points notés après la mise en route de la version rapide (Gemma 4 E2B dans llama.cpp,
voix Phonon), qui fonctionne bien sur la Jetson Orin Nano.

## 1. Démarrage automatique

AlloIA se lance toute seule à l'allumage de la Jetson, sans clavier ni terminal : on branche
la carte, et le téléphone est prêt.

- Créer un service systemd (par exemple `alloia.service`) qui lance `jetson/lancer.sh`
  au démarrage, et le relance s'il s'arrête.
- Vérifier que le son (carte KT USB) est accessible depuis le service.
- Prévoir un moyen simple de l'arrêter ou de voir ses messages (`journalctl -u alloia`).

## 2. Récupérer les documentations facilement

Aujourd'hui, les documentations sont écrites dans `~/AlloIA/docs` sur la Jetson.

- Les écrire dans un dossier partagé du réseau du FabLab, ou sur une clé USB
  (réglage `dossier_docs` dans `alloia.perso.yaml`).
- Ou les envoyer ailleurs (wiki du FabLab, e-mail au FabManager…) si besoin.

## 3. Affiner les questions

Ajuster `alloia_conduite.md` à partir de vrais appels :

- insister davantage sur les réglages des machines (laser, impression 3D, fraiseuse…) ;
- demander les fichiers sources à partager et leur licence ;
- vérifier que la documentation produite (`alloia_modele_doc.md`) correspond à ce que le
  FabLab attend.

## 4. Faire de la branche AlloIA-rapide la branche principale

Pour que la version qui fonctionne soit celle qu'on trouve en premier sur GitHub :

- fusionner `AlloIA-rapide` dans la branche principale du dépôt, ou la définir comme branche
  par défaut ;
- garder les anciennes branches (`AlloIA`, `AlloIA-omni`) pour mémoire, ou les supprimer.
