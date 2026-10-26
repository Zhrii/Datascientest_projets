# Stratégie de réentraînement

Projet : classer des produits e-commerce en 27 catégories, à partir du texte (TF-IDF + SVM linéaire).
Score de référence : le **F1-macro** (il tient compte de toutes les classes, même les petites).

## 1. Pourquoi réentraîner ?

Un modèle vieillit. Il y a de nouveaux produits, les vendeurs changent leurs mots, la répartition des catégories bouge. Le modèle devient moins bon petit à petit, sans qu'aucune erreur ne s'affiche.

Il faut donc décider **quand** on réentraîne, **comment** on le fait sans risque, et **comment** on revient en arrière si ça se passe mal.

## 2. Quand réentraîner ?

| Déclencheur | Seuil proposé | Statut |
|---|---|---|
| Nouvelles données étiquetées | au moins 10 % de produits en plus depuis le dernier entraînement | à valider en équipe |
| Le modèle devient moins bon | F1-macro < 0,75 sur des produits récents étiquetés | il faudra récupérer des retours (Phase 4) |
| Rythme régulier | une fois par mois | à valider en équipe |
| Code d'entraînement modifié | à chaque changement | manuel |

Ces déclencheurs sont **automatisés** (Phase 3) : le container `scheduler` (`pipelines/scheduler.py`) vérifie toutes les heures l'âge du modèle (> 30 jours) et la quantité de données (+10 %), et lance alors le réentraînement via `POST /retrain`. On peut toujours le lancer à la main avec `.\make.ps1 train`. En Phase 4, Prometheus surveille le F1 du modèle en production (alerte `ModelF1Low` sous 0,75) et son âge (alerte `ModelTooOld`).

## 3. Comment réentraîner sans risque ?

La règle : **un nouvel entraînement ne remplace jamais le modèle en production directement.** On utilise deux dossiers.

```
models/                  <- modèle EN PRODUCTION (lu par l'API)
├── tfidf_vectorizer.pkl
├── svm_text_model.pkl
├── label_encoder.pkl
├── metrics.json         <- carte d'identité : score, date, numéro du run MLflow
├── candidate/           <- nouveau modèle à tester (le "candidat")
└── archive/<date>/      <- anciens modèles de production (pour revenir en arrière)
```

Les trois étapes sont celles du pipeline ZenML `rakuten_training_pipeline`, lancé par `.\make.ps1 train` (ou `POST /retrain` sur l'API) :

0. **Contrôle des données** (étapes `collect_data` puis `validate_data`) : colonnes, 27 catégories, doublons, désignations vides. Si un contrôle échoue, le pipeline s'arrête : on n'entraîne pas sur des données cassées. La version des données (empreintes DVC) est relevée.
1. **Entraînement** (étapes `preprocess_data` puis `train_model`) : le nouveau modèle est écrit dans `models/candidate/`, le run est noté dans MLflow avec la version des données et le rapport qualité, et le modèle devient une nouvelle version de `rakuten-text-svm` dans le Model Registry.
2. **Promotion** (étape `promote_model`) : on compare le F1-macro du candidat avec celui de la production.
   - Candidat au moins aussi bon : l'ancien modèle est archivé, le candidat prend sa place, sa version reçoit l'alias `production` dans le Model Registry, et le run MLflow reçoit le tag `promoted_to_production=true`.
   - Candidat moins bon : refusé, la production ne bouge pas, le tag est `false`.
3. **Mise en service** : chaque réplique de l'API vérifie toutes les minutes l'alias `production` du Model Registry et charge la nouvelle version toute seule, sans redémarrage ni coupure.

Les protections :
- le candidat doit être complet (3 fichiers + `metrics.json`, écrit en dernier), sinon la promotion s'arrête ;
- les fichiers sont copiés sous un nom temporaire puis renommés d'un coup, donc l'API ne lit jamais un fichier à moitié copié ;
- `--dry-run` simule sans rien modifier, `--min-gain` exige une amélioration minimale, `--force` promeut quoi qu'il arrive (à éviter) ;
- le script renvoie un code de sortie (0 promu, 2 refusé, 1 erreur) qu'un outil d'automatisation pourra lire plus tard.

## 4. Revenir en arrière (rollback)

**Méthode principale : déplacer l'alias dans MLflow.** Onglet *Models* → `rakuten-text-svm` → poser l'alias `production` sur la version précédente. En moins d'une minute, toutes les répliques de l'API la chargent (aucun redémarrage).

**Copie locale.** Chaque promotion garde aussi l'ancien modèle dans `models/archive/<date>/`. Après un rollback, recopier l'archive correspondante dans `models/` : c'est la référence à laquelle `promote_model` compare le prochain candidat, et le modèle que l'API utilise si MLflow est indisponible.

```powershell
dir models\archive
copy models\archive\<date>\*.pkl models\
copy models\archive\<date>\metrics.json models\
```

MLflow garde aussi les 3 fichiers de chaque entraînement (onglet Artifacts, stockés dans MinIO), et la version des données utilisée : `git checkout` du commit correspondant puis `.\make.ps1 data-pull` redonne exactement ces données.

## 5. Limites (à dire honnêtement dans le rapport)

- **Les deux modèles ne sont pas testés sur exactement les mêmes produits.** Chaque entraînement tire ses 20 % de test au hasard (avec une graine fixe) dans les données du moment. Si les données changent, la comparaison n'est pas parfaite. Amélioration possible : un jeu de test fixe, jamais utilisé pour entraîner.
- **Le F1 en production n'est pas mesuré directement.** Il ne se calcule que sur des produits dont on connaît la vraie catégorie. À la place, on surveille un signe avant-coureur : la **dérive des données** (Evidently, container `drift-monitor`) compare les produits reçus aux données d'entraînement — forme des textes et répartition des catégories prédites — et déclenche l'alerte `DataDrift`. Une dérive n'est pas réentraînée automatiquement : sans nouvelles données étiquetées, réentraîner ne changerait rien ; elle signale qu'il faut en collecter.
- **Le déclencheur « F1 < 0,75 sur des produits récents » n'est pas automatisé.** Il faudrait des produits récents dont on connaît la vraie catégorie (retours des modérateurs). Le scheduler ne gère que l'âge du modèle et la quantité de données ; l'alerte `ModelF1Low` porte sur le F1 mesuré à l'entraînement.
- **Un seul critère de décision (le F1-macro).** On pourrait aussi regarder le score par classe, pour ne pas promouvoir un modèle qui s'effondre sur une petite classe.
