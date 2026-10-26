# Architecture MLOps — Projet Rakuten

Workspace : **`Datascientest_projets`**

La stack reprend le schéma *Architecture Overview* (ZenML) bloc par bloc, et chaque bloc sert réellement. Elle couvre les 4 phases du kick-off MLOps (voir la correspondance plus bas).

## Containers

| Bloc | Service Docker | Port hôte | Ce qu'il fait concrètement |
|------|----------------|-----------|----------------------------|
| Pipeline Runner (`make train`) | `train` (profil `jobs`) | — | Lance le pipeline ZenML `rakuten_training_pipeline` |
| Étapes du pipeline | (dans `train`) | — | collect → **validate** → preprocess → train → promote |
| Model Registry | `mlflow` | **5000** | Chaque entraînement = une version de `rakuten-text-svm` ; l'alias `production` désigne celle servie |
| Répartiteur de charge | `api-gateway` (nginx) | **8000** | Répartit les requêtes sur les répliques de l'API |
| FastAPI Inference | `api` (×2) | — | Charge `rakuten-text-svm@production` + image + fusion ; `/predict` `/retrain` `/model/info` `/health` `/metrics` |
| État partagé | `redis` | — | Limite de requêtes, verrou et statut du `/retrain`, journal des prédictions |
| ZenML Server | `zenml` | **8888** | Historique des runs du pipeline, étapes, artefacts |
| MySQL | `mysql` | — | Base de métadonnées du serveur ZenML |
| MinIO S3 | `minio` | 9000 / **9001** | Buckets `zenml` (artefacts entre étapes), `mlflow` (fichiers des modèles), `dvc` (données versionnées) |
| MLflow Experiments | `mlflow` | **5000** | Paramètres, F1, version des données, rapport qualité, fichiers de chaque entraînement |
| Dérive des données | `drift-monitor` | — | Compare les produits reçus aux données d'entraînement (Evidently) |
| Automatisation | `scheduler` | — | Déclenche `/retrain` tout seul (modèle > 30 jours, +10 % de données) |
| Prometheus | `prometheus` | **9092** | Scrape les répliques de l'API et le drift-monitor + 12 règles d'alerte |
| Grafana | `grafana` | **3002** | Dashboard *Rakuten Inference API* |
| UI | `streamlit` | 8501 | Démo, appelle l'API |

```
You
 ├─ make train / POST /retrain / scheduler → Pipeline Runner (ZenML)
 │     1. collect_data ──── version des données (empreintes DVC)
 │     2. validate_data ─── arrêt du pipeline si les données sont cassées
 │     3. preprocess_data ─────────────► MinIO (artefact processed_data)
 │     4. train_model ──── Log Metrics ► MLflow (+ version des données, rapport qualité, fichiers dans MinIO)
 │                    └─ Register ────► Model Registry (nouvelle version)
 │     5. promote_model ── si F1 ≥ prod ► alias "production" + models/ (archive de l'ancien)
 │     Track Run ─────────────────────► ZenML Server ─► MySQL
 ├─ API calls ─► nginx :8000 ─► FastAPI ×2 ── Load Production Model ◄ Model Registry (rechargé seul)
 │                                  └─ journal des prédictions ─► Redis ─► drift-monitor (Evidently)
 ├─ dvc push / pull ─► MinIO (bucket dvc)
 ├─ View pipelines → ZenML :8888
 └─ View experiments → MLflow :5000

FastAPI ×2 + drift-monitor ─ Metrics ► Prometheus :9092 (alertes) ► Grafana :3002
```

Stratégie de réentraînement (déclencheurs, promotion, rollback, limites) : [docs/STRATEGIE_REENTRAINEMENT.md](docs/STRATEGIE_REENTRAINEMENT.md).

## Correspondance avec les phases du kick-off

| Phase | Composant clé | Réalisation |
|-------|---------------|-------------|
| **1 — Fondations** | Versioning du code | Git / GitHub, une branche par chantier |
| | Environnements reproductibles | `setup_env.ps1`, `environment.yml`, `requirements*.txt`, Docker |
| | Pipelines données + entraînement | `scripts/collecte.py`, `process.py`, `training.py` |
| | API basique, documentation | `api/main.py` (Swagger `/docs`), ce document |
| **2 — Microservices, suivi, versioning** | Suivi des expériences | MLflow Experiments |
| | Versioning des données | **DVC** (CSV de `data brut/`, stockage MinIO) ; empreinte dans chaque run MLflow |
| | Versioning des modèles | MLflow Model Registry, alias `production`, archives |
| | Isolation des services | 14 services Docker (dont l'API en 2 répliques), 1 container = 1 tâche |
| **3 — Orchestration et déploiement** | Orchestration de bout en bout | pipeline ZenML en 5 étapes |
| | API sécurisée | authentification (HTTP Basic) + **autorisation par rôle** (user / admin) + limite de requêtes |
| | CI | GitHub Actions : 57 tests pytest à chaque push |
| | Réentraînement et déploiement automatisés | `scheduler` → `/retrain` → promotion → les répliques chargent le nouveau modèle seules |
| | Scalabilité légère | **2 répliques** de l'API derrière **nginx**, état partagé dans **Redis** |
| **4 — Surveillance et maintenance** | Monitoring des performances | Prometheus + Grafana (trafic, latence, erreurs, F1, âge du modèle) |
| | Qualité des données | étape `validate_data` du pipeline (rapport dans MLflow) |
| | Détection de dérive | **Evidently** (`drift-monitor`) → métriques, alerte `DataDrift`, rapport HTML |
| | Système d'alertes | 12 règles Prometheus |
| | Front-end | Streamlit, page *8. MLOps / API* pour la démo |

## Sécurité de l'API (`api/security.py`)

| Compte | Rôle | Droits | Limite `/predict` |
|--------|------|--------|-------------------|
| `demo` / `demo` | user | `/predict` | 30 / minute |
| `admin` / `rakuten` | admin | `/predict`, `/model/info`, `/retrain` | 600 / minute |

- **Authentification** HTTP Basic, comparaison en temps constant ; mauvais identifiants → **401**.
- **Autorisation** : un *user* sur une route *admin* → **403**.
- **Limite de requêtes** par utilisateur, fenêtre glissante, partagée par les répliques (Redis) ; au-delà → **429** + `Retry-After`. `/retrain` : 2 / heure.
- Comptes dans `API_USERS` (`login:motdepasse:role,...`) ; mots de passe à changer dans un fichier `.env` (`ADMIN_PASSWORD`, `DEMO_PASSWORD`).
- `/health` (healthcheck Docker) et `/metrics` (Prometheus) restent publics.
- Chaque refus est compté (`auth_failures_total`, `forbidden_total`, `rate_limited_total`) et peut déclencher une alerte.

```powershell
curl.exe -X POST http://localhost:8000/predict -F "text=Chaise en bois"                   # 401
curl.exe -u demo:demo -X POST http://localhost:8000/retrain                               # 403
curl.exe -u demo:demo -X POST http://localhost:8000/predict -F "text=Chaise en bois"      # 200
```

## Scalabilité (`deploy/nginx.conf`, `src/state.py`)

- `API_REPLICAS` répliques de l'API (défaut 2), derrière **nginx** (`api-gateway`) qui répartit la charge et retrouve les répliques par le DNS de Docker. L'en-tête `X-API-Instance` indique la réplique qui a répondu.
- **Redis** partage l'état : limite de requêtes commune, un seul `/retrain` à la fois (verrou), statut du dernier réentraînement visible depuis n'importe quelle réplique, journal des prédictions.
- **Rechargement à chaud** : chaque réplique vérifie toutes les `MODEL_REFRESH_SECONDS` (60 s) l'alias `production` du Model Registry et recharge le modèle texte s'il a changé. Pas de redémarrage, pas de coupure.
- Prometheus découvre chaque réplique (`dns_sd_configs`) ; alerte `ApiReplicaDown` si l'une tombe.

```powershell
$env:API_REPLICAS=3; docker compose up -d   # passer à 3 répliques
```

## Données : versioning, qualité, dérive

**Versioning (DVC).** Les CSV de `data brut/` sont suivis par DVC : Git ne contient que les fichiers `*.csv.dvc` (empreinte md5), les CSV sont dans MinIO (bucket `dvc`). Chaque run MLflow enregistre ces empreintes (`data_X_train_update_md5`, `data_Y_train_md5`) : pour un modèle donné, on sait avec quelles données il a été entraîné, et `git checkout <commit>` puis `.\make.ps1 data-pull` redonne exactement ces données.

```powershell
.\make.ps1 data-push    # après une mise à jour des CSV : python -m dvc add "data brut/X_train_update.csv" puis push
.\make.ps1 data-pull    # récupérer les CSV (MinIO doit tourner : .\make.ps1 up)
```

**Qualité (`scripts/validate_data.py`, étape `validate_data`).** Contrôles bloquants : colonnes attendues, au moins 1 000 produits, catégories connues, 27 catégories présentes, ≤ 1 % de désignations vides, ≤ 1 % de `productid` en double. Avertissements : descriptions manquantes > 50 %, catégorie de moins de 50 produits. Si un contrôle échoue, le pipeline s'arrête avant l'entraînement : la production n'est pas touchée. Le rapport est enregistré dans MLflow (`data/quality_report.json`).

**Dérive (`drift-monitor`, Evidently).** Toutes les 2 minutes, les 300 derniers produits reçus par l'API (au moins 200) sont comparés à 5 000 produits d'entraînement sur : longueur du texte, nombre de mots, part de chiffres, catégorie (vraie en référence, prédite en production). Distances de Wasserstein (numériques, seuil 0,2) et de Jensen-Shannon (catégorie, seuil 0,25), calibrées sur de vrais produits Rakuten (≤ 0,11) et sur des produits inhabituels (0,7 à 6) ; le jeu est déclaré en dérive si au moins la moitié des variables ont dérivé. Les textes reçus sont nettoyés exactement comme le fait l'API avant de prédire (`src/preprocessing/serving.py`). Résultats : métriques `data_drift_*` (Grafana, alerte `DataDrift`) et rapport HTML `reports/drift_report.html` (affiché dans Streamlit). Démo : bouton *Simuler une dérive* de Streamlit.

## Automatisation des entraînements

Le container `scheduler` vérifie toutes les heures (`CHECK_INTERVAL_SECONDS`) les déclencheurs de la stratégie de réentraînement :

| Déclencheur | Réglage |
|-------------|---------|
| aucun modèle en production | — |
| modèle trop vieux | `RETRAIN_MAX_AGE_DAYS=30` |
| nouvelles données | `NEW_DATA_THRESHOLD=0.10` (+10 % de produits) |

S'il faut réentraîner, il appelle `POST /retrain` (compte admin) : l'API lance le pipeline ZenML ; si le modèle est promu, toutes les répliques le chargent. La règle « jamais moins bon que la production » s'applique toujours.

## Alertes (`monitoring/alerts.yml`)

| Alerte | Condition | Gravité |
|--------|-----------|---------|
| ApiDown | plus aucune réplique ne répond depuis 1 min | critical |
| ApiReplicaDown | une réplique ne répond plus depuis 1 min | warning |
| ModelNotLoaded | un pipeline (texte / image / fusion) non chargé depuis 2 min | critical |
| HighErrorRate | > 10 % de prédictions en erreur sur 5 min | warning |
| HighLatency | p95 de `/predict` > 2 s sur 5 min | warning |
| RateLimitHit | > 10 requêtes bloquées (429) en 5 min | warning |
| ForbiddenAccess | > 5 accès refusés (403) en 5 min | warning |
| AuthFailures | > 5 mots de passe refusés en 5 min | warning |
| ModelF1Low | F1-macro texte < 0,75 | warning |
| ModelTooOld | modèle texte entraîné il y a > 30 jours | info |
| DataDrift | dérive détectée par Evidently | warning |
| DriftMonitorDown | aucune analyse de dérive depuis 15 min | warning |

Visibles sur http://localhost:9092/alerts et dans le panneau *Alertes actives* de Grafana. Pas d'Alertmanager : aucune notification e-mail / Slack n'est envoyée (piste d'amélioration).

## Tests

```powershell
.\make.ps1 test     # pytest dans l'image de l'API
```

57 tests (`tests/`) : routes publiques / protégées, rôles (403), mauvais mot de passe, prédiction, validation (400), limite de requêtes (429, par rôle, partagée via Redis), verrou du `/retrain` (409), journal des prédictions, promotion (premier modèle, meilleur, moins bon, `--min-gain`, `--dry-run`, candidat incomplet), contrôle qualité des données, version des données, détection de dérive (Evidently), déclencheurs du scheduler, nettoyage HTML. Les mêmes tests tournent sur GitHub à chaque push (`.github/workflows/tests.yml`).

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et **démarré** (≈ 8 Go de mémoire pour la stack)
- Pour DVC depuis le poste : `pip install -r requirements-mlops.txt`

## Démarrage

```powershell
.\make.ps1 up         # toute la stack
.\make.ps1 data-pull  # si les CSV de "data brut/" manquent (voir Données)
.\make.ps1 train      # pipeline ZenML ; les répliques chargent seules le modèle promu
```

| URL | Accès |
|-----|-------|
| http://localhost:8000/docs | API Swagger (bouton *Authorize* : `admin` / `rakuten` ou `demo` / `demo`) |
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

Sans `REDIS_URL`, l'API garde son état en mémoire (une seule réplique).

## Code

| Fichier | Rôle |
|---------|------|
| `pipelines/rakuten_pipeline.py` | Pipeline ZenML + création de la stack `rakuten` (orchestrateur local + artifact store MinIO) |
| `pipelines/scheduler.py` | déclenchement automatique du réentraînement |
| `pipelines/drift_monitor.py` | service de détection de dérive (Evidently → Prometheus, rapport HTML) |
| `scripts/collecte.py` | CSV bruts → `data/raw/merged.csv` |
| `scripts/validate_data.py` | contrôle qualité des données |
| `scripts/data_version.py` | version des données (empreintes DVC) |
| `scripts/process.py` | nettoyage texte → `data/processed/processed_data.csv` |
| `scripts/training.py` | TF-IDF + SVM → `models/candidate/` + run MLflow + version dans le Model Registry |
| `scripts/promote_model.py` | candidat ≥ production ? → `models/` + archive + alias `production` (codes 0 promu / 2 refusé / 1 erreur) |
| `scripts/predict.py` | prédiction / évaluation en ligne de commande |
| `scripts/export_fusion.py` | exporte le modèle fusion (TF-IDF + CLIP + SVM) dans `models/` |
| `src/inference/predictor.py` | chargement des modèles + routage texte / image / fusion (API et Streamlit) |
| `src/monitoring/drift.py` | calcul de la dérive (Evidently) |
| `src/state.py` | état partagé entre répliques (Redis) |
| `api/main.py` | API FastAPI |
| `api/security.py` | authentification, rôles, limite de requêtes |
| `deploy/nginx.conf` | répartiteur de charge |
| `monitoring/` | Prometheus (scrape + alertes), Grafana (datasource + dashboard) |
| `tests/` | tests pytest |

## Artefacts modèles

- **Texte (production)** : Model Registry `rakuten-text-svm@production` ; copie locale dans `models/` (`tfidf_vectorizer.pkl`, `svm_text_model.pkl`, `label_encoder.pkl`, `metrics.json`) utilisée si MLflow est indisponible ; anciennes versions dans `models/archive/<date>/`
- **Image** : `models/image_logistic_regression_27_classes_baseline.pkl` (téléchargé depuis HF au besoin)
- **Fusion** : `models/tfidf_fusion.pkl`, `svd_fusion.pkl`, `fusion_svm.pkl`, `label_encoder_fusion.pkl` (via `.\make.ps1 export-fusion`)

Les modèles image et fusion ne sont pas réentraînés par le pipeline (CLIP / ResNet : trop lourds pour un réentraînement CPU) ; ils restent des fichiers locaux.
