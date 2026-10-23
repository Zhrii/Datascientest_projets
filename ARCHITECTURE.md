# Architecture MLOps — Projet Rakuten

Workspace : **`Datascientest_projets`**

La stack reprend le schéma *Architecture Overview* (ZenML) bloc par bloc, et chaque bloc sert réellement.

## Schéma → containers

| Bloc du schéma | Service Docker | Port hôte | Ce qu'il fait concrètement |
|----------------|----------------|-----------|----------------------------|
| Pipeline Runner (`make train`) | `train` (profil `jobs`) | — | Lance le pipeline ZenML `rakuten_training_pipeline` |
| preprocess_data → train_model | (étapes du pipeline) | — | `scripts/collecte.py` + `process.py`, puis `training.py` (TF-IDF + SVM) |
| Model Registry | `mlflow` | **5000** | Chaque entraînement = une version de `rakuten-text-svm` ; l'alias `production` désigne celle servie |
| FastAPI Inference | `api` | **8000** | Charge `rakuten-text-svm@production` (fallback `models/`) + image + fusion ; `/predict` 🔒 `/retrain` 🔒 `/model/info` 🔒 `/health` `/metrics` |
| ZenML Server | `zenml` | **8888** | Historique des runs du pipeline, étapes, artefacts |
| MySQL | `mysql` | — | Base de métadonnées du serveur ZenML |
| MinIO S3 Artifact Store | `minio` | 9000 / **9001** | Bucket `zenml` (artefacts entre étapes) + bucket `mlflow` (fichiers des modèles) |
| MLflow Experiments | `mlflow` | **5000** | Paramètres, F1, fichiers de chaque entraînement |
| Prometheus | `prometheus` | **9092** | Scrape `/metrics` de l'API + règles d'alerte (`monitoring/alerts.yml`) |
| Grafana | `grafana` | **3002** | Dashboard *Rakuten Inference API* (trafic, latence, sécurité, F1, alertes actives) |
| Automatisation (en plus du schéma) | `scheduler` | — | Déclenche `/retrain` tout seul (modèle > 30 jours, +10 % de données) |
| UI (en plus du schéma) | `streamlit` | 8501 | Démo, appelle l'API |

```
You
 ├─ make train / POST /retrain → Pipeline Runner (ZenML)
 │     1. preprocess_data ─────────────► MinIO (artefact processed_data)
 │     2. train_model ──── Log Metrics ► MLflow (+ fichiers dans MinIO)
 │                    └─ Register ────► Model Registry (nouvelle version)
 │     3. promote_model ── si F1 ≥ prod ► alias "production" + models/ (archive de l'ancien)
 │     Track Run ─────────────────────► ZenML Server ─► MySQL
 ├─ API calls      → FastAPI :8000 ── Load Production Model ◄ Model Registry
 ├─ View pipelines → ZenML :8888
 └─ View experiments → MLflow :5000

FastAPI ─ Metrics ► Prometheus :9092 ► Grafana :3002
```

Stratégie de réentraînement (déclencheurs, promotion, rollback, limites) : [docs/STRATEGIE_REENTRAINEMENT.md](docs/STRATEGIE_REENTRAINEMENT.md).

## Correspondance avec les phases du projet

| Phase | Exigence | Réalisation |
|-------|----------|-------------|
| 1 — Fondation | Code hors notebook (collecte, processing, training) | `scripts/collecte.py`, `process.py`, `training.py` |
| | Environnement virtuel | `setup_env.ps1`, `environment.yml`, `requirements*.txt` |
| | Mini API | `api/main.py` (FastAPI) |
| | Git | dépôt GitHub, branches de travail, CI `.github/workflows/tests.yml` |
| 2 — Micro-services | Docker, 1 container = 1 tâche | `docker-compose.yml` (11 services) |
| | MLflow | Experiments + Model Registry |
| | Stratégie de réentraînement | candidat → promotion si F1 ≥ production → archive / rollback |
| 3 — Orchestration et sécurité | Orchestration | pipeline ZenML `rakuten_training_pipeline` |
| | Tests basiques | `tests/` (pytest : API, sécurité, promotion, scheduler, preprocessing) |
| | Login / mot de passe | HTTP Basic (`api/security.py`) |
| | Limite de requêtes | 30 `/predict` / minute, 2 `/retrain` / heure par utilisateur → 429 |
| | Automatisation des entraînements | container `scheduler` (`pipelines/scheduler.py`) |
| 4 — Monitoring et présentation | Prometheus + Grafana, dashboards | `monitoring/` |
| | Alertes | `monitoring/alerts.yml` (8 règles) → Prometheus `/alerts` + panneau Grafana |
| | Interface de présentation | Streamlit (`app.py`, page *8. MLOps / API* pour la démo) |

## Sécurité de l'API

- **Login / mot de passe** (HTTP Basic) sur `/predict`, `/model/info`, `/retrain`. Identifiants : variables `API_USERNAME` / `API_PASSWORD` (défaut `admin` / `rakuten`, à surcharger dans un fichier `.env`). Comparaison en temps constant (`secrets.compare_digest`).
- **Limite de requêtes** par utilisateur, fenêtre glissante : `PREDICT_RATE_LIMIT=30/minute`, `RETRAIN_RATE_LIMIT=2/hour`. Au-delà : **429** + en-tête `Retry-After`.
- `/health` (healthcheck Docker) et `/metrics` (Prometheus) restent publics.
- Chaque refus est compté (`auth_failures_total`, `rate_limited_total`) et peut déclencher une alerte.

```powershell
curl.exe -X POST http://localhost:8000/predict -F "text=Chaise en bois"                    # 401
curl.exe -u admin:rakuten -X POST http://localhost:8000/predict -F "text=Chaise en bois"   # 200
```

## Automatisation des entraînements

Le container `scheduler` vérifie toutes les heures (`CHECK_INTERVAL_SECONDS`) les déclencheurs de la stratégie de réentraînement :

| Déclencheur | Réglage |
|-------------|---------|
| aucun modèle en production | — |
| modèle trop vieux | `RETRAIN_MAX_AGE_DAYS=30` |
| nouvelles données | `NEW_DATA_THRESHOLD=0.10` (+10 % de produits) |

S'il faut réentraîner, il appelle `POST /retrain` (avec login) : l'API lance le pipeline ZenML et recharge le modèle s'il est promu. La règle « jamais moins bon que la production » s'applique toujours.

## Alertes (`monitoring/alerts.yml`)

| Alerte | Condition | Gravité |
|--------|-----------|---------|
| ApiDown | Prometheus ne lit plus `/metrics` depuis 1 min | critical |
| ModelNotLoaded | un pipeline (texte / image / fusion) non chargé depuis 2 min | critical |
| HighErrorRate | > 10 % de prédictions en erreur sur 5 min | warning |
| HighLatency | p95 de `/predict` > 2 s sur 5 min | warning |
| RateLimitHit | > 10 requêtes bloquées (429) en 5 min | warning |
| AuthFailures | > 5 mots de passe refusés en 5 min | warning |
| ModelF1Low | F1-macro texte < 0,75 | warning |
| ModelTooOld | modèle texte entraîné il y a > 30 jours | info |

Visibles sur http://localhost:9092/alerts et dans le panneau *Alertes actives* de Grafana. Pas d'Alertmanager : aucune notification e-mail / Slack n'est envoyée (piste d'amélioration).

## Tests

```powershell
.\make.ps1 test     # pytest dans l'image de l'API
```

`tests/` couvre : routes publiques / protégées, mauvais mot de passe, prédiction, validation (400), limite de requêtes (429), promotion (premier modèle, meilleur, moins bon, `--min-gain`, `--dry-run`, candidat incomplet), déclencheurs du scheduler, nettoyage HTML. Les mêmes tests tournent sur GitHub à chaque push (`.github/workflows/tests.yml`).

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et **démarré**

## Démarrage

```powershell
.\make.ps1 up      # toute la stack
.\make.ps1 train   # pipeline ZenML, puis restart de l'API
```

| URL | Accès |
|-----|-------|
| http://localhost:8000/docs | API Swagger (bouton *Authorize* : `admin` / `rakuten`) |
| http://localhost:8501 | Streamlit |
| http://localhost:8888 | ZenML (`default` / `zenml`) |
| http://localhost:5000 | MLflow (Experiments + Models) |
| http://localhost:9001 | MinIO (`minioadmin` / `minioadmin`) |
| http://localhost:9092 | Prometheus |
| http://localhost:3002 | Grafana (`admin` / `rakuten`) |

Stop : `.\make.ps1 down`

## Mode local (sans Docker ni ZenML)

```powershell
.\make.ps1 install-mlops
.\make.ps1 train-local   # collecte -> process -> training (models/candidate)
.\make.ps1 promote       # candidat -> models/ s'il est meilleur
.\make.ps1 api
.\make.ps1 streamlit
```

## Code

| Fichier | Rôle |
|---------|------|
| `pipelines/rakuten_pipeline.py` | Pipeline ZenML + création de la stack `rakuten` (orchestrateur local + artifact store MinIO) |
| `scripts/collecte.py` | CSV bruts → `data/raw/merged.csv` |
| `scripts/process.py` | nettoyage texte → `data/processed/processed_data.csv` |
| `scripts/training.py` | TF-IDF + SVM → `models/candidate/` + run MLflow + version dans le Model Registry |
| `scripts/promote_model.py` | candidat ≥ production ? → `models/` + archive + alias `production` (codes 0 promu / 2 refusé / 1 erreur) |
| `scripts/predict.py` | prédiction / évaluation en ligne de commande |
| `scripts/export_fusion.py` | exporte le modèle fusion (TF-IDF + CLIP + SVM) dans `models/` |
| `src/inference/predictor.py` | chargement des modèles + routage texte / image / fusion (API et Streamlit) |
| `api/main.py` | API FastAPI |
| `api/security.py` | login / mot de passe + limite de requêtes |
| `pipelines/scheduler.py` | déclenchement automatique du réentraînement |
| `monitoring/alerts.yml` | règles d'alerte Prometheus |
| `tests/` | tests pytest |

## Artefacts modèles

- **Texte (production)** : Model Registry `rakuten-text-svm@production` ; copie locale dans `models/` (`tfidf_vectorizer.pkl`, `svm_text_model.pkl`, `label_encoder.pkl`, `metrics.json`) utilisée si MLflow est indisponible ; anciennes versions dans `models/archive/<date>/`
- **Image** : `models/image_logistic_regression_27_classes_baseline.pkl` (téléchargé depuis HF au besoin)
- **Fusion** : `models/tfidf_fusion.pkl`, `svd_fusion.pkl`, `fusion_svm.pkl`, `label_encoder_fusion.pkl` (via `.\make.ps1 export-fusion`)

Les modèles image et fusion ne sont pas réentraînés par le pipeline (CLIP / ResNet : trop lourds pour un réentraînement CPU) ; ils restent des fichiers locaux.
