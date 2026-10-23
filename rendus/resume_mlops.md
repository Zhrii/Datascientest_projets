# Résumé du travail MLOps — Projet Rakuten

**Équipe :** Morgane Bernard, Tristan Pruvost, Nicolas Neguiral  
**Projet :** Classification multimodale de produits e-commerce (27 classes `prdtypecode`)  
**Dossier de référence :** `Datascientest_projets`

---

## 1. Contexte : d'où on partait

Avant cette phase, le projet était déjà solide **côté data science** :

- Notebooks d'exploration, preprocessing, baseline et modèles avancés (texte / image / fusion)
- Code modulaire dans `src/`
- Application Streamlit de démonstration (`app.py`)
- Rapports dans `rendus/` (texte, image, fusion)
- Modèles champions identifiés (notamment fusion TF-IDF + CLIP + SVM, F1-macro ≈ **0,84**)

En revanche, l'ensemble restait une **approche laboratoire** : entraînement via notebooks, modèles chargés localement dans Streamlit, pas d'API, pas de suivi d'expériences, pas de monitoring.

L'objectif MLOps, découpé en 4 phases, était de transformer ce travail en **service déployable, sécurisé, réentraînable et surveillé**, sur le modèle du schéma d'architecture ZenML.

---

## 2. Objectif de cette phase

**Industrialiser** le modèle sans réécrire la science :

| Avant | Après |
|-------|--------|
| Notebooks + Streamlit seul | Scripts + pipeline ZenML + API + UI + monitoring |
| Modèles « à la main » | Model Registry MLflow, alias `production`, promotion automatique |
| Pas de métriques runtime | Prometheus + Grafana + 8 règles d'alerte |
| Pas de suivi d'expériences | MLflow (paramètres, F1, fichiers des modèles) |
| API ouverte à tous | Login / mot de passe + limite de requêtes |
| Réentraînement manuel | Scheduler qui déclenche le pipeline tout seul |
| Tout sur la machine locale | Stack **Docker Compose** (11 services) |

En une phrase : **garder les modèles et le preprocessing, ajouter l'emballage production**.

---

## 3. Phase 1 — Fondation

| Exigence | Réalisation |
|----------|-------------|
| Sortir du notebook | `scripts/collecte.py` (CSV bruts → fichier fusionné), `scripts/process.py` (nettoyage HTML, normalisation), `scripts/training.py` (TF-IDF + SVM linéaire), `scripts/predict.py` (prédiction / évaluation en ligne de commande) |
| Environnement virtuel | `setup_env.ps1` (venv + dépendances + spaCy + kernel Jupyter), `environment.yml` (Conda), `requirements*.txt` par usage |
| Mini API | `api/main.py` (FastAPI) |
| Git | dépôt GitHub de l'équipe, une branche par chantier, Pull Requests |

**Couche d'inférence partagée** (`src/inference/`) : une seule source de vérité pour l'API, Streamlit et les scripts.

1. Texte + image → modèle **fusion** (TF-IDF + CLIP + SVM)
2. Texte seul → **SVM TF-IDF** (F1 ≈ 0,79 sur les 84 916 produits)
3. Image seule → **ResNet50 + régression logistique**
4. Fusion absente mais texte + image → **fallback texte**

---

## 4. Phase 2 — Micro-services

### 4.1 Un container = une tâche (`docker-compose.yml`)

| Bloc du schéma | Service | Port | Rôle |
|----------------|---------|------|------|
| FastAPI Inference | `api` | 8000 | Prédictions texte / image / fusion |
| UI | `streamlit` | 8501 | Démo, client de l'API |
| Pipeline Runner | `train` | — | Lance le pipeline ZenML (`make train`) |
| ZenML Server | `zenml` | 8888 | Historique des runs du pipeline |
| MySQL | `mysql` | — | Métadonnées ZenML |
| MinIO S3 | `minio` | 9001 | Artefacts ZenML (bucket `zenml`) et MLflow (bucket `mlflow`) |
| MLflow | `mlflow` | 5000 | Expériences + Model Registry |
| Prometheus | `prometheus` | 9092 | Métriques + alertes |
| Grafana | `grafana` | 3002 | Dashboards |
| (automatisation) | `scheduler` | — | Déclenche le réentraînement |

### 4.2 MLflow

Chaque entraînement est un *run* : paramètres (TF-IDF, taille du jeu…), métriques (F1-macro, accuracy…) et les 3 fichiers du modèle (stockés dans MinIO). Chaque modèle devient une **version** de `rakuten-text-svm` dans le **Model Registry**.

### 4.3 Stratégie de réentraînement (`docs/STRATEGIE_REENTRAINEMENT.md`)

- Un entraînement produit un **candidat** (`models/candidate/`), jamais directement la production.
- `promote_model` compare son F1-macro à celui de la production : **au moins aussi bon → promu** (alias `production`, ancien modèle archivé dans `models/archive/<date>/`), sinon **refusé**.
- **Rollback** : recopier une archive, ou déplacer l'alias `production` sur une version précédente dans MLflow.

---

## 5. Phase 3 — Orchestration et sécurité

### 5.1 Orchestration : pipeline ZenML (`pipelines/rakuten_pipeline.py`)

`preprocess_data` → `train_model` → `promote_model`, exécuté par le container `train` (`make train`) ou par l'API (`POST /retrain`). Chaque run est visible dans ZenML ; les données passées d'une étape à l'autre sont stockées dans MinIO.

### 5.2 Tests (`tests/`, pytest — 30 tests)

| Fichier | Ce qui est vérifié |
|---------|--------------------|
| `test_api.py` | routes publiques / protégées, mauvais mot de passe, prédiction texte et fusion, erreur 400, limite 429 |
| `test_security.py` | lecture des limites, quota par utilisateur, fenêtre glissante |
| `test_promote.py` | premier modèle, meilleur candidat (+ archive), moins bon (refusé), `--min-gain`, `--dry-run`, candidat incomplet |
| `test_scheduler.py` | déclencheurs : pas de modèle, modèle trop vieux, +10 % de données |
| `test_preprocessing.py` | nettoyage HTML, minuscules, description manquante |

Lancement : `.\make.ps1 test`. Les mêmes tests tournent sur GitHub à chaque push (`.github/workflows/tests.yml`).

### 5.3 Sécurité de l'API (`api/security.py`)

- **Login / mot de passe** (HTTP Basic) sur `/predict`, `/model/info`, `/retrain` ; identifiants dans les variables d'environnement `API_USERNAME` / `API_PASSWORD`.
- **Limite de requêtes** par utilisateur : 30 prédictions / minute, 2 réentraînements / heure ; au-delà, erreur **429** avec le délai d'attente.
- `/health` et `/metrics` restent publics (healthcheck Docker, Prometheus).

### 5.4 Automatisation des entraînements (`pipelines/scheduler.py`)

Le container `scheduler` vérifie toutes les heures les déclencheurs de la stratégie : **aucun modèle**, **modèle de plus de 30 jours**, **+10 % de produits** depuis le dernier entraînement. Il appelle alors `POST /retrain` ; la promotion garde la règle « jamais moins bon ».

---

## 6. Phase 4 — Monitoring et présentation

### 6.1 Métriques exposées par l'API (`/metrics`)

| Métrique | Sens |
|----------|------|
| `predict_requests_total{modality,status}` | prédictions par modalité et résultat |
| `predict_latency_seconds` | latence (histogramme → p95) |
| `model_ready{pipeline}` | modèle chargé (1/0) |
| `model_f1_macro{pipeline}` | F1-macro du modèle chargé |
| `model_trained_timestamp_seconds` | date d'entraînement du modèle texte |
| `retrain_running` | réentraînement en cours |
| `auth_failures_total` / `rate_limited_total` | refus 401 / 429 |

### 6.2 Dashboard Grafana *Rakuten Inference API*

Requêtes, latence p95, modèles prêts, réentraînement en cours, total des prédictions, **requêtes refusées (401 / 429)**, **F1 des modèles**, **âge du modèle**, **alertes actives**.

### 6.3 Alertes (`monitoring/alerts.yml`)

ApiDown, ModelNotLoaded, HighErrorRate (> 10 %), HighLatency (p95 > 2 s), RateLimitHit, AuthFailures, ModelF1Low (< 0,75), ModelTooOld (> 30 jours). Visibles dans Prometheus (`/alerts`) et dans Grafana.

### 6.4 Interface de présentation

Streamlit (8 pages) ; la page **8. MLOps / API** sert à la démo : architecture, appel sans mot de passe (401), rafale de requêtes (429), déclenchement du réentraînement. Script de démo : `rendus/checklist_demo_mlops.md`.

---

## 7. Démarrage

Prérequis : Docker Desktop démarré (WSL2 requis sous Windows).

```powershell
cd "F:\Projets\Projet ecole\Datascientest_projets"
.\make.ps1 up      # toute la stack
.\make.ps1 train   # pipeline ZenML (premier modèle texte)
.\make.ps1 test    # tests
.\make.ps1 down    # arrêt
```

| URL | Accès |
|-----|--------|
| http://localhost:8501 | Streamlit |
| http://localhost:8000/docs | API Swagger (`admin` / `rakuten`) |
| http://localhost:8888 | ZenML (`default` / `zenml`) |
| http://localhost:5000 | MLflow |
| http://localhost:9001 | MinIO (`minioadmin` / `minioadmin`) |
| http://localhost:9092 | Prometheus |
| http://localhost:3002 | Grafana (`admin` / `rakuten`) |

---

## 8. Limites connues (à dire honnêtement)

| Élément | Limite | Piste |
|---------|--------|-------|
| Réentraînement | seul le modèle **texte** est réentraîné ; image et fusion (CLIP / ResNet) restent des fichiers figés | GPU + pipeline dédié |
| Fusion servie | export sur échantillon (F1 ≈ 0,74) ; F1 0,84 obtenu dans le notebook complet | export sur toutes les données |
| Comparaison des modèles | pas de jeu de test fixe : chaque entraînement tire ses 20 % de validation dans les données du moment | jeu de test gelé |
| Dérive des données | pas de détection automatique d'une baisse de qualité en production (il faudrait des vraies catégories récentes) | retours modérateurs + alerte sur la répartition des prédictions |
| Alertes | pas d'Alertmanager : pas d'e-mail / Slack | Alertmanager |
| Sécurité | un seul compte, mots de passe en variables d'environnement, HTTP sans TLS | comptes multiples, secrets Docker, HTTPS |
| Limite de requêtes | compteur en mémoire : remis à zéro au redémarrage, valable pour une seule instance de l'API | Redis |

---

## 9. Synthèse en 3 phrases

1. Le travail data (modèles, preprocessing, rapports) existait déjà.
2. Les 4 phases MLOps l'ont transformé en service : scripts et API (1), micro-services Docker + MLflow + stratégie de réentraînement (2), pipeline ZenML, tests, sécurité et réentraînement automatique (3), monitoring avec alertes et démo Streamlit (4).
3. Chaque brique du schéma d'architecture sert réellement : un nouveau modèle est entraîné, tracé, comparé, promu et surveillé sans intervention manuelle.
