

Mise en place
1. Cloner le projet
2. Creer un environement virtuel (via anaconda navigator par exemple)
3. Installer les requirements
3. Ouvrir Jupyter ou une console de commande
4. (optionnel si votre connexion AWS est déjà configurée) ajouter un fichier credentials.txt contenant uniquement {aws_access_key_id}/{aws_secret_access_key} dans le dossier du projet

Utilisation 
- Soit dans le notebook NBA_Cote_predictor.ipynb (recommandée !), soit en console via main.py

Execution
- La description détaillée de l'execution du notebook se trouve dans le notebook et dans le fichier nba_cote_computer.py
- Il faut 4 à 5 heures pour l'extraction des variables. (Uniquement au premier lancement. Les données sont stockées en local au fur et à mesure pour permettre leur exploration et leur réutilisation.)
- Entre 1h et 2h pour les simulations.

Output 
- Deux fichiers contenant les cotes de chaque équipe avant et après la saison régulière. 
- 3 graphiques en plus si lancé depuis le notebook.

Ce que ce programme ne fait jamais
- Utiliser des données du match ou postérieur à celui-ci pour prédire l'issu du match.

Bugs connus
- Si votre connexion à AWS n'est pas configuré et que vous n'ajoutez pas de credentials.txt

Fiabilité du système
- Ce projet a été testé sur deux PC différents en utilisant conda pour l'installation des "requirements" après 
  un clone du repo et la création d'un environement virtuel neuf.

Auteur : Théophile Ravillion