# Architecture MLOps — Projet Rakuten

Workspace : **`Datascientest_projets`**

## Schéma → containers

| Bloc schéma | Service Docker | Port hôte |
|-------------|----------------|-----------|
| FastAPI Inference | `api` | **8000** |
| UI | `streamlit` | 8501 |
| Pipeline `make train` | `train` (profil) | — |
| ZenML Server | `zenml` | **8888** |
| Prometheus | `prometheus` | **9092** |
| Grafana | `grafana` | **3002** |
| MLflow Experiments | `mlflow` | 5000 |
| MinIO S3 Artifact Store | `minio` | 9000 / **9001** |
| MySQL ZenML Metadata | `mysql` | 3306 |

```
You
 ├─ make train / train-docker / POST /retrain  → Pipeline
 ├─ API calls                                 → FastAPI :8000
 ├─ View pipelines                            → ZenML :8888
 └─ View experiments                          → MLflow :5000

Pipeline → artefacts (models/, MinIO) + métriques MLflow
FastAPI  → charge modèle production + /metrics
Prometheus :9092 ← scrape FastAPI
Grafana :3002
```

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et **démarré**

## Démarrage

```powershell
.\make.ps1 up
```

| URL | Accès |
|-----|-------|
| http://localhost:8000/docs | API Swagger |
| http://localhost:8501 | Streamlit |
| http://localhost:8888 | ZenML (default / zenml) |
| http://localhost:9092 | Prometheus |
| http://localhost:3002 | Grafana (`admin` / `rakuten`) — dashboard *Rakuten Inference API* |
| http://localhost:5000 | MLflow |
| http://localhost:9001 | MinIO (`minioadmin` / `minioadmin`) |

Retrain dans Docker :

```powershell
.\make.ps1 train-docker
```

Stop :

```powershell
.\make.ps1 down
```

## Mode local (sans Docker)

```powershell
.\make.ps1 install-mlops
.\make.ps1 api
.\make.ps1 streamlit
```

## Artefacts modèles

- Texte local : `models/tfidf_vectorizer.pkl`, `label_encoder.pkl`, `svm_linear_baseline_final.pkl`
- Image : téléchargé depuis HF au besoin
- Fusion : exporter via `notebooks/12_multimodal_fusion.ipynb`
