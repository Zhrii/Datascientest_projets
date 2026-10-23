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

Pour l'instant (Phase 2), on lance tout **à la main** avec une commande. L'automatisation viendra en Phase 3, et la détection automatique d'une baisse de qualité en Phase 4 (Prometheus et Grafana).

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

1. **Entraînement** (étapes `preprocess_data` puis `train_model`) : le nouveau modèle est écrit dans `models/candidate/`, le run est noté dans MLflow et le modèle devient une nouvelle version de `rakuten-text-svm` dans le Model Registry.
2. **Promotion** (étape `promote_model`) : on compare le F1-macro du candidat avec celui de la production.
   - Candidat au moins aussi bon : l'ancien modèle est archivé, le candidat prend sa place, sa version reçoit l'alias `production` dans le Model Registry, et le run MLflow reçoit le tag `promoted_to_production=true`.
   - Candidat moins bon : refusé, la production ne bouge pas, le tag est `false`.
3. **Mise en service** : l'API charge `rakuten-text-svm@production` depuis le Model Registry (`make train` redémarre l'API ; `POST /retrain` recharge le modèle tout seul).

Les protections :
- le candidat doit être complet (3 fichiers + `metrics.json`, écrit en dernier), sinon la promotion s'arrête ;
- les fichiers sont copiés sous un nom temporaire puis renommés d'un coup, donc l'API ne lit jamais un fichier à moitié copié ;
- `--dry-run` simule sans rien modifier, `--min-gain` exige une amélioration minimale, `--force` promeut quoi qu'il arrive (à éviter) ;
- le script renvoie un code de sortie (0 promu, 2 refusé, 1 erreur) qu'un outil d'automatisation pourra lire plus tard.

## 4. Revenir en arrière (rollback)

Chaque promotion garde l'ancien modèle dans `models/archive/<date>/`. Pour revenir à une version précédente, on recopie ses fichiers dans `models/` puis on redémarre l'API :

```powershell
dir models\archive
copy models\archive\<date>\*.pkl models\
copy models\archive\<date>\metrics.json models\
docker compose restart api
```

MLflow garde aussi les 3 fichiers de chaque entraînement (onglet Artifacts, stockés dans MinIO). Autre façon de revenir en arrière : dans MLflow (onglet Models), déplacer l'alias `production` sur une version précédente, puis redémarrer l'API.

## 5. Limites (à dire honnêtement dans le rapport)

- **Les deux modèles ne sont pas testés sur exactement les mêmes produits.** Chaque entraînement tire ses 20 % de test au hasard (avec une graine fixe) dans les données du moment. Si les données changent, la comparaison n'est pas parfaite. Amélioration possible : un jeu de test fixe, jamais utilisé pour entraîner.
- **Pas de détection automatique d'une baisse de qualité.** Le F1 ne se calcule que sur des produits dont on connaît la vraie catégorie. Piste pour la Phase 4 : surveiller la répartition des catégories prédites et la confiance du modèle avec Prometheus et Grafana.
- **Déclenchement manuel.** À automatiser en Phase 3.
- **Un seul critère de décision (le F1-macro).** On pourrait aussi regarder le score par classe, pour ne pas promouvoir un modèle qui s'effondre sur une petite classe.
