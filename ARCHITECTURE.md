# Architecture MLOps — Projet Rakuten

Workspace : **`Datascientest_projets`**

La stack reprend le schéma *Architecture Overview* (ZenML) bloc par bloc, et chaque bloc sert réellement.

## Schéma → containers

| Bloc du schéma | Service Docker | Port hôte | Ce qu'il fait concrètement |
|----------------|----------------|-----------|----------------------------|
| Pipeline Runner (`make train`) | `train` (profil `jobs`) | — | Lance le pipeline ZenML `rakuten_training_pipeline` |
| preprocess_data → train_model | (étapes du pipeline) | — | `scripts/collecte.py` + `process.py`, puis `training.py` (TF-IDF + SVM) |
| Model Registry | `mlflow` | **5000** | Chaque entraînement = une version de `rakuten-text-svm` ; l'alias `production` désigne celle servie |
| FastAPI Inference | `api` | **8000** | Charge `rakuten-text-svm@production` (fallback `models/`) + image + fusion ; `/predict` `/retrain` `/health` `/model/info` `/metrics` |
| ZenML Server | `zenml` | **8888** | Historique des runs du pipeline, étapes, artefacts |
| MySQL | `mysql` | — | Base de métadonnées du serveur ZenML |
| MinIO S3 Artifact Store | `minio` | 9000 / **9001** | Bucket `zenml` (artefacts entre étapes) + bucket `mlflow` (fichiers des modèles) |
| MLflow Experiments | `mlflow` | **5000** | Paramètres, F1, fichiers de chaque entraînement |
| Prometheus | `prometheus` | **9092** | Scrape `/metrics` de l'API |
| Grafana | `grafana` | **3002** | Dashboard *Rakuten Inference API* |
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

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et **démarré**

## Démarrage

```powershell
.\make.ps1 up      # toute la stack
.\make.ps1 train   # pipeline ZenML, puis restart de l'API
```

| URL | Accès |
|-----|-------|
| http://localhost:8000/docs | API Swagger |
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

## Artefacts modèles

- **Texte (production)** : Model Registry `rakuten-text-svm@production` ; copie locale dans `models/` (`tfidf_vectorizer.pkl`, `svm_text_model.pkl`, `label_encoder.pkl`, `metrics.json`) utilisée si MLflow est indisponible ; anciennes versions dans `models/archive/<date>/`
- **Image** : `models/image_logistic_regression_27_classes_baseline.pkl` (téléchargé depuis HF au besoin)
- **Fusion** : `models/tfidf_fusion.pkl`, `svd_fusion.pkl`, `fusion_svm.pkl`, `label_encoder_fusion.pkl` (via `.\make.ps1 export-fusion`)

Les modèles image et fusion ne sont pas réentraînés par le pipeline (CLIP / ResNet : trop lourds pour un réentraînement CPU) ; ils restent des fichiers locaux.
