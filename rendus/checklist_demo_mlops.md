# Checklist démo live — Stack MLOps

Durée cible : **6–7 minutes** (après la partie data), dans une présentation de 20 minutes.

## Avant la démo (la veille et 15 min avant)

```powershell
cd "F:\Projets\Projet ecole\Datascientest_projets"
# Docker Desktop démarré (baleine verte) !
.\make.ps1 up              # attendre ~3 min (chaque réplique de l'API charge CLIP / ResNet)
docker compose ps          # tout Up / healthy (minio-init : Exited (0), normal)
.\make.ps1 health          # text_ready / image_ready / fusion_ready = true
.\make.ps1 test            # 57 tests verts
```

Si aucun modèle texte n'est encore dans le Model Registry : `.\make.ps1 train` (≈ 1 min 30).
Pour que la dérive soit analysée, il faut au moins 200 prédictions texte : le bouton de simulation en envoie 250.

Onglets prêts dans le navigateur :
1. http://localhost:8501 — Streamlit
2. http://localhost:8000/docs — FastAPI via nginx (bouton *Authorize* : `admin` / `rakuten`)
3. http://localhost:8888 — ZenML (`default` / `zenml`)
4. http://localhost:5000 — MLflow (onglets *Experiments* et *Models*)
5. http://localhost:3002 — Grafana (`admin` / `rakuten`), dashboard *Rakuten Inference API*
6. http://localhost:9092/alerts — Prometheus, règles d'alerte

## Script (ordre recommandé)

### 1. Streamlit — prédiction (≈ 1 min)
- Sidebar : **API FastAPI 🟢 online**
- Page **4. Assistant Texte** : `Livre Harry Potter à l'école des sorciers édition poche` → Livre, **via API**
- Page **6. Moteur Multimodal** : texte + image produit → modèle fusion
- Message clé : *Streamlit n'est qu'un client ; le service métier est l'API*

### 2. Sécurité et scalabilité (≈ 1 min 30) — Phase 3
- Page **8. MLOps / API**, section *Sécurité* :
  - **/predict sans mot de passe** → **401** (authentification)
  - **/retrain avec le compte user** → **403** (autorisation : rôle admin requis)
  - **40 prédictions d'affilée** → ≈ 30 × 200 puis des **429** (limite 30 / minute)
- Section *Scalabilité* : le graphique montre la rafale répartie entre **les 2 répliques** de l'API
- Message clé : *2 répliques derrière nginx ; Redis partage la limite de requêtes et le verrou de réentraînement*

### 3. Données : versioning, qualité, dérive (≈ 1 min 30) — Phases 2 et 4
- **Versioning** : MLflow, un run → paramètres `data_X_train_update_md5` = empreinte DVC des données ; montrer `data brut/X_train_update.csv.dvc`
- **Qualité** : MLflow, artefact `data/quality_report.json` ; ZenML, étape `validate_data`
- **Dérive** : page 8, **Simuler une dérive** (250 produits courts, en anglais, pleins de chiffres) → à la prochaine analyse (≤ 2 min) : **🔴 Dérive détectée**, rapport Evidently dans l'expander
- Message clé : *on surveille aussi les données, pas seulement le service*

### 4. Pipeline de réentraînement (≈ 1 min 30) — Phases 2 et 3
- ZenML : run `rakuten_training_pipeline` → `collect_data` → `validate_data` → `preprocess_data` → `train_model` → `promote_model`
- MLflow *Models* : `rakuten-text-svm`, alias **production**
- Message clé : *un nouveau modèle n'entre en production que s'il est au moins aussi bon ; les répliques le chargent seules, sans redémarrage*
- Automatisation : le container `scheduler` relance tout seul si le modèle a plus de 30 jours ou s'il y a +10 % de données
- (Option live) **Déclencher /retrain (compte admin)** → ≈ 1 min 30 → nouveau run ZenML + MLflow, nouvelle version chargée

### 5. Monitoring et alertes (≈ 1 min) — Phase 4
- Grafana : requêtes, latence p95, **requêtes refusées (401 / 403 / 429)**, F1, âge du modèle, **répliques en service**, **dérive des données**
- Panneau **Alertes actives** : après les étapes 2 et 3, **RateLimitHit**, **DataDrift** (et AuthFailures / ForbiddenAccess si on a insisté)
- Prometheus `/alerts` : les 12 règles
- Message clé : *on surveille le service et les données en production, pas seulement le F1 du notebook*

### 6. Qualité du code (≈ 20 s)
- Onglet *Actions* de GitHub : 57 tests lancés à chaque push (CI)

## Phrase de conclusion

> On est partis d'un modèle de notebook à F1 0,84. Aujourd'hui c'est une API sécurisée et répliquée, un pipeline ZenML qui contrôle les données, réentraîne et promeut le modèle tout seul, des données et des modèles versionnés, et un monitoring qui surveille le service comme la dérive des données — chaque brique du schéma sert réellement.

## Si un service est down

```powershell
.\make.ps1 up              # relance ce qui manque
docker compose logs -f api # suivre le chargement des modèles
```

- Une réplique de l'API tombe : nginx bascule sur l'autre, la démo continue (alerte `ApiReplicaDown`).
- Streamlit sans API : la prédiction passe en **local** (fallback), la démo continue.
- MLflow indisponible : l'API charge le modèle texte depuis `models/` (fallback).
- Dérive « en attente » : moins de 200 prédictions texte → bouton *Simuler une dérive*, puis `.\make.ps1 drift` pour analyser tout de suite.
