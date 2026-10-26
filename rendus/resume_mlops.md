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
| API ouverte à tous | Login / mot de passe, rôles user / admin, limite de requêtes |
| Données non versionnées, non contrôlées | DVC + contrôle qualité dans le pipeline |
| Aucune surveillance des données | Détection de dérive (Evidently) |
| Une seule API | 2 répliques derrière nginx, état partagé dans Redis |
| Réentraînement manuel | Scheduler qui déclenche le pipeline tout seul |
| Tout sur la machine locale | Stack **Docker Compose** (14 services) |

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
| FastAPI Inference | `api` (×2) | — | Prédictions texte / image / fusion |
| UI | `streamlit` | 8501 | Démo, client de l'API |
| Pipeline Runner | `train` | — | Lance le pipeline ZenML (`make train`) |
| ZenML Server | `zenml` | 8888 | Historique des runs du pipeline |
| MySQL | `mysql` | — | Métadonnées ZenML |
| MinIO S3 | `minio` | 9001 | Artefacts ZenML (`zenml`), MLflow (`mlflow`), données DVC (`dvc`) |
| MLflow | `mlflow` | 5000 | Expériences + Model Registry |
| Prometheus | `prometheus` | 9092 | Métriques + alertes |
| Grafana | `grafana` | 3002 | Dashboards |
| (automatisation) | `scheduler` | — | Déclenche le réentraînement |
| (scalabilité) | `api-gateway` (nginx) + `redis` | 8000 | Répartition de charge, état partagé des répliques |
| (surveillance des données) | `drift-monitor` | — | Dérive des données (Evidently) |

### 4.2 MLflow

Chaque entraînement est un *run* : paramètres (TF-IDF, taille du jeu…), métriques (F1-macro, accuracy…) et les 3 fichiers du modèle (stockés dans MinIO). Chaque modèle devient une **version** de `rakuten-text-svm` dans le **Model Registry**.

### 4.3 Versioning des données (DVC)

Les CSV de `data brut/` sont suivis par **DVC** : Git ne garde que les fichiers `*.csv.dvc` (empreinte md5), les CSV sont stockés dans MinIO. Chaque run MLflow enregistre ces empreintes : pour chaque modèle, on sait avec quelles données il a été entraîné (`git checkout` + `.\make.ps1 data-pull` pour les retrouver). Code (Git) + données (DVC) + modèle (MLflow) : versioning de bout en bout.

### 4.4 Stratégie de réentraînement (`docs/STRATEGIE_REENTRAINEMENT.md`)

- Un entraînement produit un **candidat** (`models/candidate/`), jamais directement la production.
- `promote_model` compare son F1-macro à celui de la production : **au moins aussi bon → promu** (alias `production`, ancien modèle archivé dans `models/archive/<date>/`), sinon **refusé**.
- **Rollback** : recopier une archive, ou déplacer l'alias `production` sur une version précédente dans MLflow.

---

## 5. Phase 3 — Orchestration et sécurité

### 5.1 Orchestration : pipeline ZenML (`pipelines/rakuten_pipeline.py`)

`collect_data` → `validate_data` → `preprocess_data` → `train_model` → `promote_model`, exécuté par le container `train` (`make train`) ou par l'API (`POST /retrain`). Chaque run est visible dans ZenML ; les données passées d'une étape à l'autre sont stockées dans MinIO.

### 5.2 Tests (`tests/`, pytest — 57 tests)

| Fichier | Ce qui est vérifié |
|---------|--------------------|
| `test_api.py` | routes publiques / protégées, rôles (403), mauvais mot de passe, prédiction texte et fusion, erreur 400, limite 429, verrou du retrain (409), journal des prédictions |
| `test_security.py` | comptes, rôles, quota par utilisateur et par rôle, fenêtre glissante, limite partagée entre répliques (Redis) |
| `test_validate_data.py` | contrôle qualité : données propres, colonne manquante, catégorie inconnue ou absente, doublons, désignations vides |
| `test_data_version.py` | lecture des empreintes DVC, calcul md5 |
| `test_drift.py` | variables de texte, pas de dérive sur des données identiques, dérive détectée sur des données différentes (Evidently) |
| `test_promote.py` | premier modèle, meilleur candidat (+ archive), moins bon (refusé), `--min-gain`, `--dry-run`, candidat incomplet |
| `test_scheduler.py` | déclencheurs : pas de modèle, modèle trop vieux, +10 % de données |
| `test_preprocessing.py` | nettoyage HTML, minuscules, description manquante |

Lancement : `.\make.ps1 test`. Les mêmes tests tournent sur GitHub à chaque push (`.github/workflows/tests.yml`).

### 5.3 Sécurité de l'API (`api/security.py`)

- **Authentification** (HTTP Basic) : comptes `demo` (rôle *user*) et `admin` (rôle *admin*) ; mauvais identifiants → **401**.
- **Autorisation** : seul un *admin* peut réentraîner ou lire `/model/info` ; un *user* reçoit **403**.
- **Limite de requêtes** par utilisateur : 30 prédictions / minute pour un *user*, 600 pour un *admin*, 2 réentraînements / heure ; au-delà, erreur **429** avec le délai d'attente.
- `/health` et `/metrics` restent publics (healthcheck Docker, Prometheus).

### 5.4 Automatisation des entraînements (`pipelines/scheduler.py`)

Le container `scheduler` vérifie toutes les heures les déclencheurs de la stratégie : **aucun modèle**, **modèle de plus de 30 jours**, **+10 % de produits** depuis le dernier entraînement. Il appelle alors `POST /retrain` ; la promotion garde la règle « jamais moins bon ». **Déploiement automatique** : chaque réplique de l'API vérifie toutes les minutes l'alias `production` du Model Registry et charge seule le nouveau modèle, sans redémarrage.

### 5.5 Scalabilité légère

2 répliques de l'API (`API_REPLICAS`) derrière **nginx**, qui répartit la charge. **Redis** partage leur état : limite de requêtes commune, un seul réentraînement à la fois (verrou), statut du réentraînement, journal des prédictions. Une réplique tombe → nginx bascule sur l'autre, alerte `ApiReplicaDown`.

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
| `auth_failures_total` / `forbidden_total` / `rate_limited_total` | refus 401 / 403 / 429 |
| `model_registry_version` | version du Model Registry chargée par chaque réplique |
| `data_drift_share`, `data_drift_detected`, `data_drift_column_*` | dérive des données (drift-monitor) |

### 6.2 Qualité et dérive des données

- **Qualité** : l'étape `validate_data` du pipeline vérifie colonnes, 27 catégories, doublons, désignations vides et taille ; si un contrôle échoue, le pipeline s'arrête avant l'entraînement. Le rapport est enregistré dans MLflow.
- **Dérive** : le container `drift-monitor` compare toutes les 2 minutes les 300 derniers produits reçus par l'API (journal dans Redis) à 5 000 produits d'entraînement, avec **Evidently** (distances de Wasserstein et Jensen-Shannon, seuils calibrés sur de vrais produits Rakuten) : longueur du texte, nombre de mots, part de chiffres, catégorie prédite. Résultat : métriques Prometheus, alerte `DataDrift`, rapport HTML affiché dans Streamlit.

### 6.3 Dashboard Grafana *Rakuten Inference API*

Requêtes, latence p95, modèles prêts, réentraînement en cours, total des prédictions, **requêtes refusées (401 / 403 / 429)**, **F1 des modèles**, **âge du modèle**, **répliques en service**, **dérive des données** (globale et par variable), **alertes actives**.

### 6.4 Alertes (`monitoring/alerts.yml`, 12 règles)

ApiDown, ApiReplicaDown, ModelNotLoaded, HighErrorRate (> 10 %), HighLatency (p95 > 2 s), RateLimitHit, ForbiddenAccess, AuthFailures, ModelF1Low (< 0,75), ModelTooOld (> 30 jours), DataDrift, DriftMonitorDown. Visibles dans Prometheus (`/alerts`) et dans Grafana.

### 6.5 Interface de présentation

Streamlit (8 pages) ; la page **8. MLOps / API** sert à la démo : appel sans mot de passe (401), `/retrain` avec le compte user (403), rafale de requêtes (429) et sa répartition entre les répliques, simulation d'une dérive et rapport Evidently, déclenchement du réentraînement. Script de démo : `rendus/checklist_demo_mlops.md`.

---

## 7. Démarrage

Prérequis : Docker Desktop démarré (WSL2 requis sous Windows).

```powershell
cd "F:\Projets\Projet ecole\Datascientest_projets"
.\make.ps1 up      # toute la stack
.\make.ps1 data-pull  # CSV versionnés par DVC (si absents)
.\make.ps1 train   # pipeline ZenML (premier modèle texte)
.\make.ps1 test    # tests
.\make.ps1 down    # arrêt
```

| URL | Accès |
|-----|--------|
| http://localhost:8501 | Streamlit |
| http://localhost:8000/docs | API Swagger (`admin` / `rakuten` ou `demo` / `demo`) |
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
| Performance en production | le F1 n'est pas mesuré en production (il faudrait les vraies catégories des produits récents) ; on surveille la dérive des données à la place | retours des modérateurs |
| Dérive | variables simples (forme du texte, catégories) ; pas de dérive sémantique du vocabulaire | embeddings de texte (descripteurs Evidently) |
| Versioning des données | stockage DVC = MinIO local à chaque poste ; images non versionnées | stockage S3 partagé (AWS, DagsHub) |
| Alertes | pas d'Alertmanager : pas d'e-mail / Slack | Alertmanager |
| Sécurité | comptes et mots de passe en variables d'environnement, HTTP sans TLS | base d'utilisateurs, jetons (OAuth2 / JWT), secrets Docker, HTTPS |
| Scalabilité | répliques sur une seule machine ; chaque réplique charge CLIP / ResNet en mémoire | Kubernetes / Docker Swarm, serveur de modèles dédié |

---

## 9. Synthèse en 3 phrases

1. Le travail data (modèles, preprocessing, rapports) existait déjà.
2. Les 4 phases MLOps l'ont transformé en service : scripts et API (1), micro-services Docker, MLflow, versioning des données avec DVC et stratégie de réentraînement (2), pipeline ZenML avec contrôle qualité, CI, API sécurisée par rôles, réentraînement automatique et scalabilité (3), monitoring, détection de dérive et alertes (4).
3. Chaque brique du schéma d'architecture sert réellement : un nouveau modèle est entraîné, tracé, comparé, promu et surveillé sans intervention manuelle.
