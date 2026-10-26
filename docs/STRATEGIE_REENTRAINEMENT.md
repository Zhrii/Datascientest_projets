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
| Le modèle devient moins bon | F1-macro < 0,75 sur les retours des modérateurs (`POST /feedback`) | alerte `ProductionF1Low` (voir § 6) |
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

0. **Contrôle des données** (étapes `collect_data` puis `validate_data`) : 3 % des produits sont mis de côté une fois pour toutes (*holdout*, jamais utilisés pour entraîner) ; puis contrôle des colonnes, des 27 catégories, des doublons et des désignations vides. Si un contrôle échoue, le pipeline s'arrête : on n'entraîne pas sur des données cassées. La version des données (empreintes DVC) est relevée.
1. **Entraînement** (étapes `preprocess_data` puis `train_model`) : le nouveau modèle est écrit dans `models/candidate/`, le run est noté dans MLflow avec la version des données et le rapport qualité, et le modèle devient une nouvelle version de `rakuten-text-svm` dans le Model Registry.
2. **Promotion** (étape `promote_model`) : on note le candidat **et** le modèle en production sur les **mêmes produits mis de côté** (2 476 produits jamais vus), et on compare leurs F1-macro.
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

- **Comparaison équitable, mais sur un jeu fixe.** Candidat et production sont notés sur les mêmes 2 476 produits mis de côté (`--holdout`). Ce jeu ne change pas : s'il vieillit (nouveaux types de produits), il faut le renouveler, et refaire une fois la transition décrite au § 6.
- **Le F1 en production dépend des retours des modérateurs.** Sans retours, on n'a que des signes avant-coureurs : la **dérive des données** (Evidently, alerte `DataDrift`) et la **confiance du modèle** (alerte `LowConfidenceSpike`). Une dérive n'est pas réentraînée automatiquement : sans nouvelles données étiquetées, réentraîner ne changerait rien.
- **Une baisse du F1 en production déclenche une alerte, pas un réentraînement.** Il faudrait d'abord ajouter les retours des modérateurs (texte + vraie catégorie) aux données d'entraînement ; c'est l'étape suivante logique.
- **Un seul critère de décision (le F1-macro).** On pourrait aussi regarder le score par classe, pour ne pas promouvoir un modèle qui s'effondre sur une petite classe.

## 6. Mesurer le modèle en production (retours des modérateurs et confiance)

**Pourquoi.** Le F1-macro d'entraînement dit si le modèle était bon le jour où il a été entraîné. En production, on ne connaît pas la vraie catégorie des produits reçus : on ne peut pas calculer de F1 sans aide.

**Retours des modérateurs.** Chaque réponse de `POST /predict` contient un `prediction_id`. Un modérateur qui vérifie le produit renvoie la vraie catégorie avec `POST /feedback`. Le drift-monitor calcule le **F1-macro en production** sur les 500 derniers retours (à partir de 100) ; sous 0,75, l'alerte `ProductionF1Low` se déclenche. Grafana affiche le F1 en production à côté du F1 d'entraînement.

**Confiance du modèle.** Chaque prédiction a une confiance : la marge entre le meilleur score du modèle et le deuxième. Le seuil de « prédiction peu sûre » est calibré sur les produits mis de côté (`scripts/calibrate_confidence.py`) : sous ce seuil, le modèle se trompe au moins une fois sur deux. Ces prédictions sont à faire vérifier en priorité par un modérateur ; si leur part explose, l'alerte `LowConfidenceSpike` se déclenche. Les retours vérifient que ce signal tient : Grafana compare l'accuracy des prédictions sûres et peu sûres.

**Transition vers le jeu mis de côté (une seule fois).** Le modèle en production avant l'ajout du holdout (version 4) avait été entraîné sur ces produits : sur le holdout, il obtenait 0,873, un score gonflé. Le premier modèle entraîné sans eux (version 5, 0,769 sur le holdout, cohérent avec ses 0,773 en validation) a donc été promu avec `--force`. Depuis, tous les modèles sont entraînés sans le holdout et comparés équitablement dessus.
