# Analyse de sentiments
Groupe | Date | Sujet | UE | UCE | Encadrants
:---|:---|:---|:---|:---|:---
Jarod Duret, Jonathan Heno | 14/01/2021 | Prédiction de notes de films | Innovations et recherche pour la société du numérique | Application d'innovation | Richard Dufour, Vincent Labatut, Mickaël Rouvier


## Installation de l'environnement conda
L'environnement utilisé pour la réalisation de ce projet peut-être généré via la commande :
```shell
conda env create -f environment.yml
```

## Attentes
L'idée est de faire une analyse descriptive du corpus.

Il y a pour moi une première partie ou on explique la tâche, puis le corpus. Et on donne des informations sous forme de tableau (train/dev/test) :
- Nombre de commentaire
- Répartition des commentaires par classe (3 et 10)
- Nombre d'utilisateur
- Nombre de film
- Film ayant le plus de commentaire
- Utilisateur ayant fait le plus de film
- etc....

Après on peut mettre des graphiques :
- **Nuage de mot :** Mot les plus utilisés par classe (en filtrant les stop-words)
- **Moyenne des notes par film :** moyenne + écart-type)
- **Moyenne des notes par utilisateur :** (moyenne + écart-type)


On peut faire des graphiques par film et utilisateur spécifique :
- Afficher un nuage de mot pour le film ayant eu la meilleure note (et un autre pour le film ayant eu la pire note)
- Afficher un nuage de mot pour l'utilisateur ayant fait le plus de commentaire (et l'inverse)

L'idée est qu'à partir de ses graphiques vous ayez une idée des paramètres à extraire pour résoudre la tâche.


Rendu attendu le 2 Décembre
