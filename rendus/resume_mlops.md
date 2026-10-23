# Résumé du travail MLOps — Projet Rakuten

**Équipe :** Morgane Bernard, Tristan Pruvost, Nicolas Neguiral  
**Projet :** Classification multimodale de produits e-commerce (27 classes `prdtypecode`)  
**Dossier de référence :** `Datascientest_projets`

---

## 1. Contexte : d’où on partait

Avant cette phase, le projet était déjà solide **côté data science** :

- Notebooks d’exploration, preprocessing, baseline et modèles avancés (texte / image / fusion)
- Code modulaire dans `src/`
- Application Streamlit de démonstration (`app.py`)
- Rapports dans `rendus/` (texte, image, fusion)
- Modèles champions identifiés (notamment fusion TF-IDF + CLIP + SVM, F1-macro ≈ **0,84**)

En revanche, l’ensemble restait une **approche laboratoire** : entraînement via notebooks, modèles chargés localement dans Streamlit, pas d’API de production, pas de suivi d’expériences industrialisé, pas de monitoring.

Le schéma d’architecture cible (ZenML / FastAPI / Prometheus / Grafana / MLflow / MinIO) demandait de passer à une **approche MLOps**.

---

## 2. Objectif de cette phase

**Industrialiser** le modèle sans réécrire la science :

| Avant | Après |
|-------|--------|
| Notebooks + Streamlit seul | Pipeline + API + UI + monitoring |
| Modèles “à la main” | Chargement via service d’inférence + registry locale |
| Pas de métriques runtime | Prometheus + Grafana |
| Pas de tracking d’expériences en stack | MLflow (+ ZenML pour l’orchestration) |
| Tout sur la machine locale | Stack **Docker Compose** alignée sur le schéma |

En une phrase : **garder les modèles et le preprocessing, ajouter l’emballage production**.

---

## 3. Ce qui a été mis en place

### 3.1 Couche d’inférence partagée (`src/inference/`)

**Quoi :** extraction de la logique de prédiction hors de Streamlit.

**Pourquoi :** une seule source de vérité pour l’API et l’UI ; évite de dupliquer le code de prédiction (et les bugs).

**Routage intelligent :**

1. Texte + image → modèle **fusion** (si artefacts présents)
2. Texte seul → **SVM TF-IDF** (F1 ≈ 0,78)
3. Image seule → **ResNet50 + Logistic Regression**
4. Si fusion absente mais texte+image → **fallback texte**

Fichiers clés :

- `src/inference/predictor.py` — chargement des modèles + `predict()`
- `src/inference/categories.py` — libellés des 27 classes Rakuten
- `src/inference/api_client.py` — client HTTP pour Streamlit → FastAPI

### 3.2 API FastAPI (`api/main.py`) — port **8000**

**Quoi :** vrai point d’entrée “production”.

| Endpoint | Rôle |
|----------|------|
| `POST /predict` | Classification (texte et/ou image) |
| `GET /health` | Santé API + état des pipelines |
| `GET /model/info` | Métadonnées des modèles |
| `POST /retrain` | Déclenche le pipeline d’entraînement (async) |
| `GET /metrics` | Métriques au format Prometheus |

**Pourquoi :** séparer la **vitrine** (Streamlit) du **service métier** (API). C’est le cœur du schéma “Inference Layer”.

### 3.3 Streamlit refactorisé (`app.py`)

**Quoi :** Streamlit appelle l’API si elle est dispo, sinon fallback local.  
Ajout d’une page **« 8. MLOps / API »**.

**Pourquoi :** la démo reste simple pour le jury, tout en passant réellement par l’architecture MLOps.

### 3.4 Pipeline d’entraînement (`pipelines/train_pipeline.py`)

**Quoi :** 2 steps alignés sur le schéma :

1. `preprocess_data` — nettoyage / `text_combined` à partir de `data brut/`
2. `train_model` — SVM Linear TF-IDF, sauvegarde des artefacts, log MLflow

Fonctionne en **standalone** (sans ZenML) et avec ZenML si installé.

**Pourquoi :** remplacer “relancer le notebook” par une commande reproductible (`make train` / `train-docker` / `POST /retrain`).

### 3.5 Stack Docker Compose (schéma complet)

**Quoi :** conteneurisation de tous les blocs du schéma d’architecture.

| Service | Port | Rôle |
|---------|------|------|
| **FastAPI** (`api`) | 8000 | Inférence |
| **Streamlit** | 8501 | Démo UI |
| **ZenML** | 8888 | Orchestration / suivi des pipelines |
| **Prometheus** | 9092 | Collecte des métriques API |
| **Grafana** | 3002 | Visualisation (dashboard *Rakuten Inference API*) |
| **MLflow** | 5000 | Tracking d’expériences |
| **MinIO** | 9000 / 9001 | Stockage artefacts type S3 |
| **MySQL** | 3306 | Métadonnées ZenML |
| **train** (profil) | — | Exécution one-shot du pipeline |

Fichiers associés :

- `docker-compose.yml`, `Dockerfile`, `.dockerignore`
- `requirements-docker.txt` (image plus légère que le requirements notebooks)
- `monitoring/prometheus.yml`
- `monitoring/grafana/provisioning/` (datasource + dashboard)
- `Makefile` / `make.ps1`
- `ARCHITECTURE.md`

**Pourquoi :** pouvoir démarrer toute l’architecture d’un coup, de façon reproductible, et coller au schéma fourni pour le rendu / la soutenance.

### 3.6 Documentation projet

- `ARCHITECTURE.md` — mapping schéma ↔ containers / ports / commandes
- `README.md` — section MLOps ajoutée
- Ce document (`rendus/resume_mlops.md`) — synthèse “quoi / pourquoi”

---

## 4. Pourquoi chaque brique (vue “soutenance”)

```
Utilisateur / Jury
    │
    ├─ Streamlit (:8501)     → démo visuelle
    ├─ FastAPI (:8000)       → prédictions réelles
    ├─ ZenML (:8888)         → “comment on orchestre l’entraînement”
    ├─ MLflow (:5000)        → “comment on compare les expériences”
    ├─ Grafana (:3002)       → “comment on surveille la prod”
    └─ MinIO (:9001)         → “où vivent les artefacts”
```

- **Sans FastAPI** : Streamlit reste un script de démo, pas un service.
- **Sans Prometheus/Grafana** : pas de preuve de monitoring.
- **Sans MLflow/ZenML** : pas de traçabilité des entraînements.
- **Sans Docker** : installation fragile, difficile à reproduire pour l’équipe / le jury.

---

## 5. Comment démarrer (rappel)

Prérequis : Docker Desktop démarré (WSL2 requis sous Windows).

```powershell
cd "F:\Projets\Projet ecole\Datascientest_projets"
.\make.ps1 up
```

| URL | Accès |
|-----|--------|
| http://localhost:8000/docs | API Swagger |
| http://localhost:8501 | Streamlit |
| http://localhost:3002 | Grafana (`admin` / `rakuten`) |
| http://localhost:9092 | Prometheus |
| http://localhost:8888 | ZenML (`default` / `zenml`) |
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

---

## 6. État actuel (limites connues)

| Élément | État |
|---------|------|
| API texte | OK (`models/svm_linear_baseline_final.pkl` + vectorizer + encoder) |
| API image | OK (baseline ResNet / HF) |
| API fusion | OK via `pipelines/export_fusion.py` (échantillon ; F1 notebook complet ~0.84) |
| Stack Docker | Opérationnelle |
| Dashboard Grafana | Provisionné (*Rakuten Inference API*) |
| MLflow | Run texte SVM loggée (`rakuten-classification`) |
| MinIO | Service up ; modèles via volume `./models` |

---

## 7. Prochaines étapes recommandées

1. Exporter les artefacts **fusion** depuis `notebooks/12_multimodal_fusion.ipynb` vers `models/`
2. Lancer un `.\make.ps1 train-docker` pour avoir une run visible dans MLflow
3. Préparer la démo orale : Streamlit → prédiction → Grafana → (optionnel) ZenML / MLflow
4. Committer / pousser le code MLOps sur le dépôt d’équipe

---

## 8. Synthèse en 3 phrases

1. Le travail data (modèles, preprocessing, rapports) existait déjà.  
2. Cette phase a **industrialisé** le projet selon le schéma ZenML : API, pipeline, monitoring, tracking, stockage, le tout en Docker.  
3. Le but n’était pas de refaire la modélisation, mais de montrer une **architecture MLOps crédible** autour du champion multimodal Rakuten.
