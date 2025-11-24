## esther-apache-spark

Ce dépôt est un projet Python minimal basé sur Pandas et Apache Spark (via PySpark) pour expérimenter le traitement de données à grande échelle.
Il fournit une structure légère (dépendances, configuration et squelette de code) pour charger des données, effectuer des transformations avec Spark et valider les traitements au moyen de tests automatisés.

--------------------------------------------------------------------
Workflow :
- import des fichiers  du dossier data/march-input
- nettoyage et aggrégations de ces données dans des dataframes
- export des données dans data/out en .csv et dans une bdd sql
- pipeline de test qui compare les dataframes aggrégés pandas & pyspark

Notes : les notebooks .ipynb sont des fichiers de travail, ce ne sont pas les versions finales du pipeline. Les fichiers à exécuter sont bien les .py
--------------------------------------------------------------------


--------------------------------------------------------------------
Ce projet fonctionne avec uv.
Pour installer uv : "pip install uv" ou "brew install uv"

Pour lancer le projet :
- pour créer l'environnement d'après la config : "uv sync"
- pandas : depuis la racine, "uv run src pandas_files/pipeline_pandas.py"
- pyspark : depuis la racine, "uv run src pandas_files/pipeline_pyspark.py"

-> note  : les deux pipelines exportent leurs documents dans le même dossier (data/out), donc visuellement rien ne changera entre les deux si les data/out est déjà remplis
--------------------------------------------------------------------


--------------------------------------------------------------------
Pour les tests :
- le fichier conftest.py permet de configurer l'affichage des tests, il est automatiquement pris en compte par pytest
- le fichier conftest.py modifie le sys.path pour permettre l'import des fonctions des pipelines pandas & pyspark comme des modules, si le fichier test_pipeline_equivalences.py est lancé directement, et que conftest.py n'a jamais été lancé (tout seul ou via pytest), il est probable que test_pipeline_equivalence.py retourne une erreur d'import impossible


Pour lancer les tests, deux possibilités :
- uv run pytest -> lance uniquement le test
- uv run pre-commit (évite de faire des commit pour rien, mais nécessite git add . avant) -> lance toutes les actions du fichier .pre-commit-config.yaml

Le pre-commit se déclenche lors des commit, donc :
- git add .
- git commit -m "nom du commit" --> lance le pre-commit

-> note : le pre-commit est un peu chargé en vérifications, ruff notamment
est un peu difficile à contenter, il peut faire échouter le pre-commit facilement
-> si besoin : commenter la partie du du .pre-commit-config.yaml qui bloque
-> puis relancer git add . // git commit -m "mon commit"
