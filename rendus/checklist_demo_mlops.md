# Checklist démo live — Stack MLOps

Durée cible : **5–6 minutes** (après la partie data), dans une présentation de 20 minutes.

## Avant la démo (la veille et 15 min avant)

```powershell
cd "F:\Projets\Projet ecole\Datascientest_projets"
# Docker Desktop démarré (baleine verte) !
.\make.ps1 up              # attendre ~2 min (chargement CLIP / ResNet dans l'API)
docker compose ps          # tout Up / healthy (minio-init : Exited (0), normal)
.\make.ps1 health          # text_ready / image_ready / fusion_ready = true
.\make.ps1 test            # 30 tests verts
```

Si aucun modèle texte n'est encore dans le Model Registry : `.\make.ps1 train` (≈ 1 min).

Onglets prêts dans le navigateur :
1. http://localhost:8501 — Streamlit
2. http://localhost:8000/docs — FastAPI (bouton *Authorize* : `admin` / `rakuten`)
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

### 2. Sécurité de l'API (≈ 1 min) — Phase 3
- Page **8. MLOps / API** :
  - **Appeler /predict sans mot de passe** → **401**
  - **Envoyer 40 prédictions d'affilée** → environ 30 × 200 puis des **429** (limite 30 / minute)
- Message clé : *login / mot de passe et limite de requêtes, chaque refus est compté dans Prometheus*

### 3. Pipeline de réentraînement (≈ 1 min 30) — Phases 2 et 3
- ZenML : run `rakuten_training_pipeline` → étapes `preprocess_data` → `train_model` → `promote_model`
- MLflow *Experiments* : paramètres + F1 de chaque run ; *Models* : `rakuten-text-svm`, alias **production**
- Message clé : *un nouveau modèle n'entre en production que s'il est au moins aussi bon ; l'ancien est archivé (rollback)*
- Automatisation : le container `scheduler` relance tout seul si le modèle a plus de 30 jours ou s'il y a +10 % de données
- (Option live) bouton **Déclencher /retrain via API** sur la page 8 → ≈ 1 min → nouveau run dans ZenML et MLflow

### 4. Monitoring et alertes (≈ 1 min) — Phase 4
- Grafana : requêtes, latence p95, **Sécurité : requêtes refusées** (les 401 / 429 de l'étape 2 apparaissent), F1 des modèles, âge du modèle
- Panneau **Alertes actives** : après la rafale, l'alerte **RateLimitHit** se déclenche (Prometheus évalue toutes les 15 s)
- Prometheus `/alerts` : les 8 règles (API down, modèle non chargé, erreurs, latence, 429, mots de passe refusés, F1 < 0,75, modèle > 30 jours)
- Message clé : *on surveille le service en production, pas seulement le F1 du notebook*

### 5. Qualité (≈ 20 s)
- `.\make.ps1 test` ou l'onglet *Actions* de GitHub : tests lancés à chaque push

## Phrase de conclusion

> On est partis d'un modèle de notebook à F1 0,84. Aujourd'hui c'est une API sécurisée, un pipeline ZenML qui réentraîne et promeut le modèle tout seul, un Model Registry, et un monitoring avec alertes — chaque brique du schéma sert réellement.

## Si un service est down

```powershell
.\make.ps1 up              # relance ce qui manque
docker compose logs -f api # suivre le chargement des modèles
```

- Streamlit sans API : la prédiction passe en **local** (fallback), la démo continue.
- MLflow indisponible : l'API charge le modèle texte depuis `models/` (fallback).
