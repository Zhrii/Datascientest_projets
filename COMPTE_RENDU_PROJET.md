# Compte rendu du projet MLOps — Classification de produits Rakuten

> **À qui s'adresse ce document ?** Aux membres de l'équipe qui vont rédiger le rapport final et préparer la soutenance, sans avoir suivi le détail technique.
> **Ce qu'il contient :** tout ce qui a été fait, *pourquoi* ça a été fait, *où* le trouver dans le projet, et *comment* le montrer au jury, en suivant les exigences du kick-off DataScientest (4 phases).
> Chaque terme technique est expliqué la première fois qu'il apparaît ; un glossaire complet est à la fin (§ 12).

---

## Sommaire

1. [Le projet en 2 minutes](#1-le-projet-en-2-minutes)
2. [Ce qui était demandé](#2-ce-qui-était-demandé)
3. [L'architecture en une image](#3-larchitecture-en-une-image)
4. [Phase 1 — Fondations](#4-phase-1--fondations)
5. [Phase 2 — Microservices, suivi des expériences, versioning](#5-phase-2--microservices-suivi-des-expériences-versioning)
6. [Phase 3 — Orchestration, déploiement, sécurité](#6-phase-3--orchestration-déploiement-sécurité)
7. [Phase 4 — Surveillance et maintenance](#7-phase-4--surveillance-et-maintenance)
8. [Les chiffres à citer](#8-les-chiffres-à-citer)
9. [Problèmes rencontrés et solutions](#9-problèmes-rencontrés-et-solutions)
10. [Limites et pistes d'amélioration](#10-limites-et-pistes-damélioration)
11. [Démo, commandes et adresses](#11-démo-commandes-et-adresses)
12. [Glossaire](#12-glossaire)
13. [Où trouver quoi dans le projet](#13-où-trouver-quoi-dans-le-projet)
14. [Plan de rapport suggéré](#14-plan-de-rapport-suggéré)

---

## 1. Le projet en 2 minutes

### Le problème

Rakuten (site e-commerce) reçoit des milliers de fiches produit de vendeurs. Chaque produit doit être rangé dans **une des 27 catégories** du catalogue (code `prdtypecode` : livres, jeux vidéo, mobilier, piscine…). Le faire à la main est lent et source d'erreurs : on veut le **prédire automatiquement** à partir du **texte** (titre + description) et de l'**image** du produit.

### Les données (challenge Rakuten)

| Fichier | Contenu |
|---------|---------|
| `X_train_update.csv` | 84 916 produits : `designation` (titre), `description`, `productid`, `imageid` |
| `Y_train.csv` | la vraie catégorie (`prdtypecode`) de chacun de ces produits |
| `X_test_update.csv` | 13 812 produits sans catégorie (fichier de soumission du challenge) |
| `data/processed/image_train/` | une image par produit |

Particularités : 35 % des descriptions manquent, beaucoup contiennent du HTML, et les catégories sont déséquilibrées (de 764 à plus de 10 000 produits par catégorie).

### Deux grandes parties

1. **Partie data science** (notebooks 01 à 12, terminée en mars 2026) : trouver le meilleur modèle.
2. **Partie MLOps** (septembre–octobre 2026, sujet de la soutenance) : transformer ce modèle de notebook en **service fiable, sécurisé, surveillé et réentraînable**.

### Résultats de la partie data science (à rappeler dans le rapport)

| Approche | Meilleur modèle | F1-macro |
|----------|-----------------|----------|
| Texte seul | TF-IDF + SVM linéaire | **≈ 0,78** |
| Image seule | CLIP + SVM | ≈ 0,66 (ResNet50 + régression logistique : ≈ 0,51) |
| **Fusion texte + image** | TF-IDF + CLIP image + CLIP texte + SVM | **≈ 0,84** (champion) |
| Recherche texte ↔ image (matching CLIP) | — | Recall@1 ≈ 40 %, Recall@10 ≈ 69 % |

> **Le F1-macro** est la note principale du projet : pour chaque catégorie, on calcule un score entre 0 et 1 qui combine « quand le modèle dit *mobilier*, a-t-il raison ? » et « retrouve-t-il tous les meubles ? », puis on fait la moyenne des 27 catégories **sans tenir compte de leur taille**. Une petite catégorie mal classée fait donc baisser la note autant qu'une grande : c'est plus exigeant que le simple pourcentage de bonnes réponses (accuracy).

Détails : `rendus/rapport_partie_texte.md`, `rapport_partie_image.md`, `rapport_partie_fusion.md`.

---

## 2. Ce qui était demandé

Le kick-off DataScientest (« MLOps KO Projet ») fixe l'objectif : *se confronter à un vrai problème de Machine Learning Engineering* — projets **reproductibles**, déploiements **fiables**, **surveillance** des modèles, solutions **évolutives**, et savoir **présenter** ses résultats. Soutenance : **20 minutes devant un jury, avec une démo Streamlit**, en expliquant les choix techniques et l'architecture.

Le travail est découpé en 4 phases. Tableau de synthèse (le détail de chaque ligne est dans les sections 4 à 7) :

| Phase | Composant clé demandé | Fait ? | Comment |
|-------|----------------------|:------:|---------|
| **1. Fondations** | Versioning du code (Git) | ✅ | GitHub, une branche par chantier |
| | Environnements reproductibles | ✅ | `requirements*.txt`, `environment.yml`, `setup_env.ps1`, Docker |
| | Pipelines données + entraînement | ✅ | `scripts/collecte.py`, `process.py`, `training.py` |
| | API basique | ✅ | FastAPI (`api/main.py`) |
| | Documentation initiale | ✅ | `README.md`, `ARCHITECTURE.md` |
| | Tests automatisés dès le début | ✅ | 73 tests `pytest` |
| **2. Microservices, suivi, versioning** | Suivi des expériences | ✅ | MLflow |
| | Versioning des données | ✅ | DVC (stockage MinIO) |
| | Versioning des modèles | ✅ | MLflow Model Registry |
| | Isolation des services | ✅ | 14 services Docker |
| | Stratégie de réentraînement | ✅ | candidat → promotion si meilleur → archive / retour arrière |
| **3. Orchestration et déploiement** | Orchestration de bout en bout | ✅ | pipeline ZenML en 5 étapes |
| | API sécurisée (authentification + autorisation) | ✅ | login / mot de passe + rôles user / admin |
| | Limite de requêtes | ✅ | 30 prédictions / minute pour un utilisateur |
| | CI (intégration continue) | ✅ | GitHub Actions : tests à chaque push |
| | Réentraînement et déploiement automatisés | ✅ | scheduler + rechargement automatique du modèle |
| | Scalabilité légère | ✅ | 2 copies de l'API derrière nginx + Redis |
| **4. Surveillance et maintenance** | Monitoring des performances | ✅ | Prometheus + Grafana, F1 en production |
| | Qualité des données | ✅ | étape `validate_data` du pipeline |
| | Détection de dérive des données | ✅ | Evidently |
| | Système d'alertes | ✅ | 14 règles d'alerte |
| | Processus de réentraînement | ✅ | voir phase 2 et 3 |
| | Front-end (« nice to have ») | ✅ | Streamlit, page *8. MLOps / API* pour la démo |

---

## 3. L'architecture en une image

Le projet suit le schéma *Architecture Overview* fourni (ZenML / FastAPI / Prometheus / Grafana / MLflow / MinIO / MySQL), complété pour couvrir tout le kick-off. **Chaque brique sert réellement** (aucun service « décoratif »).

```
                        ┌──────────────────────────── Utilisateurs ────────────────────────────┐
                        │  Streamlit (démo) · Swagger · modérateurs · scheduler (automatique)   │
                        └───────────────────────────────┬───────────────────────────────────────┘
                                                        │ HTTP + login / mot de passe
                                                ┌───────▼────────┐
                                                │ nginx (gateway) │  répartit la charge
                                                └───┬────────┬───┘
                                        ┌───────────▼┐      ┌▼───────────┐
                                        │ API copie 1 │      │ API copie 2 │  FastAPI : prédit, reçoit les retours
                                        └─────┬──────┘      └──────┬─────┘
                                              └──── Redis ─────────┘  mémoire partagée (limites, journal…)
                                                        │
              charge le modèle "production" ◄───────────┤ journal des prédictions + retours modérateurs
                                                        ▼
 ┌──────────────── Pipeline d'entraînement (ZenML) ────────────────┐      ┌──────── Surveillance ────────┐
 │ collect → validate → preprocess → train → promote               │      │ drift-monitor (Evidently,     │
 │   │          │                       │         │                │      │   F1 en production)           │
 │   ▼          ▼                       ▼         ▼                │      │ Prometheus (métriques,        │
 │ DVC     rapport qualité          MLflow    Model Registry       │      │   14 alertes)                 │
 │ (version  (MLflow)               (scores)  (alias "production") │      │ Grafana (tableaux de bord)    │
 └─────────────────────────────────────────────────────────────────┘      └───────────────────────────────┘
        Stockage : MinIO (fichiers : données DVC, modèles, artefacts) · MySQL (historique ZenML)
```

### Les 14 services, chacun avec son rôle

| Service | Rôle en une phrase | Analogie |
|---------|-------------------|----------|
| `api` (×2) | Reçoit un produit, renvoie sa catégorie | le guichet |
| `api-gateway` (nginx) | Répartit les clients entre les 2 guichets | l'hôtesse d'accueil |
| `redis` | Mémoire partagée entre les 2 guichets | le cahier commun |
| `streamlit` | Interface web de démonstration | la vitrine |
| `train` | Lance le pipeline d'entraînement | l'atelier |
| `scheduler` | Décide tout seul quand réentraîner | le réveil |
| `drift-monitor` | Surveille les données reçues et la qualité du modèle | le contrôleur qualité |
| `mlflow` | Carnet de bord des entraînements + registre des modèles | le carnet de labo + la bibliothèque des versions |
| `zenml` | Tableau de bord des exécutions du pipeline | le tableau de suivi de production |
| `mysql` | Base de données de ZenML | les archives de ZenML |
| `minio` | Stockage de fichiers (type Amazon S3) | l'entrepôt |
| `minio-init` | Crée les « dossiers » de l'entrepôt au démarrage | — |
| `prometheus` | Relève les compteurs toutes les 15 s, déclenche les alertes | le relevé des compteurs |
| `grafana` | Graphiques à partir des compteurs | le tableau de bord |

Tout se lance avec une seule commande (`.\make.ps1 up`) grâce à **Docker Compose** (voir § 5.4).

---

## 4. Phase 1 — Fondations

### 4.1 Sortir du notebook : des scripts Python

**Pourquoi ?** Un notebook est pratique pour explorer, mais on ne peut pas le relancer automatiquement, le tester ni le déployer. On découpe donc le travail en scripts indépendants, chacun avec une entrée et une sortie claires.

| Script | Fait quoi | Entrée → sortie |
|--------|-----------|-----------------|
| `scripts/collecte.py` | Fusionne les textes et les catégories | CSV bruts → `data/raw/merged.csv` |
| `scripts/process.py` | Nettoie le texte (HTML, minuscules, espaces) et colle titre + description | → `data/processed/processed_data.csv` |
| `scripts/training.py` | Entraîne le modèle TF-IDF + SVM et mesure son F1 | → 3 fichiers `.pkl` + `metrics.json` |
| `scripts/predict.py` | Prédit en ligne de commande (un texte ou un fichier) | texte → catégorie |

Ces scripts réutilisent le code du dossier `src/` écrit pendant la partie data science (nettoyage, vectorisation, modèles), sans le réécrire.

> **TF-IDF + SVM, c'est quoi ?** TF-IDF transforme un texte en nombres : chaque mot reçoit un poids d'autant plus fort qu'il est fréquent dans ce produit et rare dans les autres (« piscine » pèse plus que « le »). Le SVM linéaire est un classifieur qui apprend à séparer les 27 catégories à partir de ces nombres. Simple, rapide, et le meilleur modèle texte du projet.

### 4.2 Environnements reproductibles

**Pourquoi ?** « Ça marche sur ma machine » ne suffit pas : chacun doit pouvoir réinstaller exactement les mêmes outils.

- `requirements.txt` (notebooks), `environment.yml` (Conda), `setup_env.ps1` (installation en une commande sous Windows).
- Un fichier de dépendances par usage, pour garder des images Docker légères : `requirements-api.txt`, `requirements-train.txt`, `requirements-mlflow.txt`, `requirements-test.txt`, `requirements-mlops.txt`.
- Versions figées quand c'est nécessaire (ex. `mlflow==3.16.1` identique côté serveur et client, `pandas<3` pour la compatibilité avec Evidently).
- **Docker** (§ 5.4) : chaque service tourne dans un conteneur identique sur toutes les machines.

### 4.3 Une API

**Pourquoi ?** Le modèle doit être utilisable par d'autres programmes (le site, une appli…), pas seulement dans un notebook. Une **API** est un point d'entrée web : on lui envoie un produit, elle répond la catégorie.

L'API est écrite avec **FastAPI** (`api/main.py`). Elle génère automatiquement sa documentation interactive (**Swagger**, http://localhost:8000/docs), où l'on peut tester chaque route.

| Route | Rôle | Qui peut l'appeler |
|-------|------|--------------------|
| `POST /predict` | Prédit la catégorie (texte, image, ou les deux) + confiance | utilisateur connecté |
| `POST /feedback` | Un modérateur donne la vraie catégorie d'un produit déjà prédit | utilisateur connecté |
| `GET /model/info` | Quels modèles sont chargés, leur version, leur F1 | administrateur |
| `POST /retrain` | Lance un réentraînement | administrateur |
| `GET /health` | L'API va-t-elle bien ? (utilisé par Docker) | tout le monde |
| `GET /metrics` | Compteurs pour Prometheus | tout le monde |

**Routage intelligent** (`src/inference/predictor.py`) : texte + image → modèle fusion ; texte seul → SVM texte ; image seule → modèle image ; si le modèle fusion manque → repli sur le texte.

### 4.4 Git et collaboration

- Dépôt GitHub de l'équipe ; une **branche** par chantier (`feature/mlops-phase1-texte`, `feature/mlops-phase2-docker` de Nicolas, `feature/mlops-fusion` qui regroupe tout).
- La branche `feature/mlops-fusion` **fusionne** le travail MLOps de Nicolas (scripts, Docker, MLflow, stratégie de réentraînement) et de Tristan (API multimodale, Streamlit, monitoring), en gardant le meilleur de chacun (§ 9.1).
- `.gitignore` : les gros fichiers (modèles, images, données) ne vont pas dans Git.

### 4.5 Tests automatisés

**Pourquoi ?** Vérifier automatiquement, à chaque modification, que rien n'est cassé.

**73 tests** avec `pytest` (dossier `tests/`), lancés avec `.\make.ps1 test` et automatiquement sur GitHub à chaque envoi de code (§ 6.4) :

| Fichier | Ce qui est vérifié |
|---------|-------------------|
| `test_api.py` | routes publiques / protégées, 401 / 403 / 429, prédiction, retours modérateurs, un seul réentraînement à la fois |
| `test_security.py` | comptes, rôles, limite de requêtes (y compris partagée entre les 2 copies de l'API) |
| `test_promote.py` | promotion : premier modèle, meilleur, moins bon, comparaison équitable sur produits mis de côté |
| `test_validate_data.py` | contrôles qualité des données |
| `test_data_version.py` | version des données (DVC) |
| `test_drift.py` | détection de dérive (Evidently) |
| `test_performance.py` | F1 en production, retours en double |
| `test_confidence.py` | confiance du modèle, produits mis de côté, calibrage du seuil |
| `test_scheduler.py` | déclencheurs du réentraînement automatique |
| `test_preprocessing.py` | nettoyage du texte (y compris celui de l'API) |

Les tests utilisent un **faux modèle** pour l'API (pas besoin de charger 2 Go de modèles) et un **faux Redis** (`fakeredis`).

---

## 5. Phase 2 — Microservices, suivi des expériences, versioning

### 5.1 Suivi des expériences : MLflow

**Pourquoi ?** Quand on entraîne 10 modèles, il faut savoir lequel a donné quel score, avec quels réglages et quelles données. MLflow est le **carnet de bord** automatique de chaque entraînement.

Pour chaque entraînement (un *run*), MLflow enregistre :
- les **paramètres** (nombre de mots du TF-IDF, taille des jeux, nombre de catégories…) ;
- les **métriques** (F1-macro, F1 pondéré, accuracy) ;
- la **version des données** utilisées (empreinte DVC, § 5.2) ;
- le **rapport qualité** des données (§ 7.2) ;
- les **fichiers du modèle** (stockés dans MinIO).

Adresse : http://localhost:5000 (onglet *Experiments*).

### 5.2 Versioning des données : DVC

**Pourquoi ?** Git versionne le code, mais pas les gros fichiers de données. Or pour reproduire un modèle, il faut savoir **exactement avec quelles données** il a été entraîné.

**DVC** (*Data Version Control*) résout ça :
- les CSV de `data brut/` sont stockés dans MinIO (dans un « dossier » `dvc`) ;
- Git ne garde qu'un petit fichier par CSV (`X_train_update.csv.dvc`) contenant son **empreinte** (md5 : une suite de 32 caractères qui change dès qu'un seul octet du fichier change) ;
- chaque entraînement enregistre cette empreinte dans MLflow ;
- résultat : pour n'importe quel modèle, on sait quelles données il a vues, et `git checkout` + `.\make.ps1 data-pull` les redonne exactement.

C'est le **versioning de bout en bout** demandé : code (Git) + données (DVC) + modèle (MLflow).

> ⚠️ **Conséquence pour l'équipe** : les CSV ont été retirés de Git. En récupérant la branche, Git les supprime de votre dossier `data brut/` : il faut les recopier (sauvegarde ou archive DataScientest), car le stockage DVC est le MinIO de la machine de Tristan.

### 5.3 Versioning des modèles : MLflow Model Registry

Chaque modèle entraîné devient une **version** (1, 2, 3, 4, 5…) du modèle `rakuten-text-svm` dans le **registre de modèles** de MLflow (onglet *Models*). Une étiquette (*alias*) `production` désigne la version servie par l'API. Changer de version = déplacer l'étiquette, sans toucher au code.

### 5.4 Microservices et Docker

**Pourquoi ?** Plutôt qu'un gros programme qui fait tout, on découpe en **services indépendants**, un par tâche (§ 3). Si l'un tombe, les autres continuent ; on peut en mettre plusieurs copies ; chacun a ses propres dépendances.

**Docker** emballe chaque service dans un **conteneur** : une boîte qui contient le programme et tout ce dont il a besoin, identique sur toutes les machines. **Docker Compose** (`docker-compose.yml`) décrit les 14 services et les lance ensemble.

Trois images Docker sont construites : `Dockerfile.api` (API + Streamlit : modèles texte, image, fusion), `Dockerfile.training` (pipeline, scheduler, drift-monitor : léger, sans les modèles d'image), `Dockerfile.mlflow`.

### 5.5 Stratégie de réentraînement

Document : `docs/STRATEGIE_REENTRAINEMENT.md` (rédigé par Nicolas, complété depuis).

**Règle d'or : un nouveau modèle ne remplace jamais directement celui en production.**

1. L'entraînement produit un **candidat** (`models/candidate/`).
2. La **promotion** (`scripts/promote_model.py`) compare le candidat au modèle en production, **sur les mêmes 2 476 produits mis de côté** (§ 7.5) :
   - au moins aussi bon → **promu** : il prend la place, l'ancien est **archivé** (`models/archive/<date>/`), l'étiquette `production` du registre passe sur lui ;
   - moins bon → **refusé**, rien ne change.
3. **Retour arrière** (*rollback*) : déplacer l'étiquette `production` sur une version précédente dans MLflow ; l'API la recharge en moins d'une minute.

Cette règle a fonctionné en vrai : lors d'un entraînement, un candidat moins bon (0,7728 contre 0,7854) a été **refusé automatiquement**.

---

## 6. Phase 3 — Orchestration, déploiement, sécurité

### 6.1 Orchestration : le pipeline ZenML

**Pourquoi ?** Enchaîner les étapes dans le bon ordre, garder une trace de chaque exécution, et pouvoir tout relancer d'une commande.

**ZenML** exécute le pipeline `rakuten_training_pipeline` (`pipelines/rakuten_pipeline.py`) en **5 étapes** :

| Étape | Rôle |
|-------|------|
| 1. `collect_data` | Fusionne les CSV, met de côté 3 % des produits (jamais entraînés), relève la version des données (DVC) |
| 2. `validate_data` | Contrôle qualité : **arrête tout** si les données sont cassées (§ 7.2) |
| 3. `preprocess_data` | Nettoie le texte |
| 4. `train_model` | Entraîne, enregistre dans MLflow, crée une version dans le registre |
| 5. `promote_model` | Met en production seulement si c'est meilleur |

Chaque exécution est visible dans ZenML (http://localhost:8888) avec ses étapes, durées et statut ; les données échangées entre étapes sont stockées dans MinIO ; l'historique est dans MySQL. Une exécution complète prend environ 2 à 3 minutes.

### 6.2 Sécurité de l'API

**Pourquoi ?** Une API ouverte peut être utilisée par n'importe qui, surchargée, ou servir à lancer des réentraînements à tort.

| Mécanisme | Ce que c'est | Ce qui se passe sinon |
|-----------|--------------|----------------------|
| **Authentification** | Il faut un identifiant et un mot de passe (HTTP Basic) | erreur **401** |
| **Autorisation (rôles)** | Compte `demo` (rôle *user*) : prédire et envoyer des retours. Compte `admin` : en plus, réentraîner et voir `/model/info` | erreur **403** |
| **Limite de requêtes** | 30 prédictions par minute pour un *user*, 600 pour un *admin*, 2 réentraînements par heure | erreur **429** + délai d'attente indiqué |

Détails : mots de passe comparés en temps constant (pour ne rien laisser deviner), changeables dans un fichier `.env` ; chaque refus est compté et peut déclencher une alerte (§ 7.6). Code : `api/security.py`.

### 6.3 Réentraînement et déploiement automatisés

- **Le scheduler** (`pipelines/scheduler.py`) vérifie toutes les heures s'il faut réentraîner : aucun modèle en production, modèle de plus de **30 jours**, ou **10 % de produits en plus** depuis le dernier entraînement. Si oui, il appelle `POST /retrain`.
- **Déploiement automatique** : chaque copie de l'API vérifie toutes les minutes la version `production` du registre et **charge seule** le nouveau modèle, sans redémarrage ni coupure de service. Vérifié en vrai : les 2 copies sont passées seules de la version 3 à la 4, puis à la 5.

### 6.4 Intégration continue (CI)

À chaque envoi de code sur GitHub (*push*), **GitHub Actions** installe le projet sur une machine propre et lance les 73 tests (`.github/workflows/tests.yml`). Résultat visible dans l'onglet *Actions* du dépôt (vert = tout passe). Le déploiement continu (CD) était optionnel et n'a pas été fait.

### 6.5 Scalabilité légère

**Pourquoi ?** Une seule API = un seul point de panne et une capacité limitée.

- **2 copies de l'API** (*répliques*) tournent en parallèle (nombre réglable : `API_REPLICAS`).
- **nginx** (le *répartiteur de charge*) envoie chaque requête à l'une ou l'autre. Si une copie tombe, il bascule sur l'autre.
- **Redis** (une mémoire partagée très rapide) permet aux 2 copies de partager : la limite de requêtes (sinon on aurait 2 × 30 prédictions par minute), un **verrou** pour qu'un seul réentraînement tourne à la fois, le journal des prédictions et les retours des modérateurs.

Vérifié en vrai : une rafale de 40 requêtes est répartie moitié-moitié entre les 2 copies et la limite de 30 est bien commune ; une copie arrêtée → **30 requêtes sur 30 réussies** quand même, et l'alerte `ApiReplicaDown` se déclenche.

---

## 7. Phase 4 — Surveillance et maintenance

### 7.1 Monitoring : Prometheus et Grafana

- **Prometheus** relève toutes les 15 secondes les compteurs exposés par l'API (`/metrics`) et par le drift-monitor : nombre de prédictions, erreurs, temps de réponse, refus 401 / 403 / 429, modèles chargés, F1, âge du modèle, confiance, dérive, F1 en production…
- **Grafana** (http://localhost:3002) affiche ces compteurs dans le tableau de bord **Rakuten Inference API** (17 graphiques) : trafic, latence, sécurité, répliques en service, F1 d'entraînement vs production, confiance, dérive des données, alertes actives.

### 7.2 Qualité des données

**Pourquoi ?** Un CSV corrompu ne doit jamais produire un modèle en production.

L'étape `validate_data` (`scripts/validate_data.py`) vérifie, **avant** l'entraînement :

| Contrôle | Seuil | Bloquant ? |
|----------|-------|:---------:|
| Colonnes attendues présentes | toutes | oui |
| Nombre de produits | ≥ 1 000 | oui |
| Catégories connues | aucune inconnue | oui |
| Les 27 catégories présentes | 27 | oui |
| Titres vides | ≤ 1 % | oui |
| Identifiants en double | ≤ 1 % | oui |
| Descriptions manquantes | > 50 % | avertissement |
| Catégorie rare | < 50 produits | avertissement |

Si un contrôle bloquant échoue, le pipeline s'arrête : rien n'est entraîné, rien ne change en production. Le rapport est enregistré dans MLflow. Sur les vraies données : **tout passe** (82 440 produits d'entraînement, 27 catégories, 0 doublon).

### 7.3 Détection de dérive des données (Evidently)

**Pourquoi ?** Un modèle peut se dégrader **sans aucune erreur visible** si les produits reçus ne ressemblent plus à ceux de l'entraînement (nouveaux types de produits, vendeurs qui écrivent autrement…). C'est la **dérive** (*drift*).

Le service `drift-monitor` compare toutes les 2 minutes les **300 derniers produits reçus** par l'API à **5 000 produits d'entraînement**, avec la librairie **Evidently**, sur 4 variables : longueur du texte, nombre de mots, part de chiffres, catégorie prédite. Une variable a « dérivé » si la distance entre les deux distributions dépasse un seuil ; le jeu est en dérive si au moins la moitié des variables ont dérivé.

**Calibrage** sur de vraies données : de vrais produits Rakuten donnent des distances ≤ 0,11 (bruit normal) ; des produits inhabituels (titres courts en anglais, pleins de chiffres) donnent de 0,57 à 5. Seuils retenus : 0,2 et 0,25.

Résultats : métriques dans Grafana, alerte `DataDrift`, rapport HTML Evidently consultable dans Streamlit. Vérifié en vrai : 300 vrais produits → pas de dérive ; 250 produits inhabituels → dérive sur les 4 variables, alerte déclenchée.

### 7.4 Performance du modèle en production : le F1 réel

**Le problème :** le F1 affiché jusqu'ici est celui mesuré **à l'entraînement**. En production, on ne connaît pas la vraie catégorie des produits reçus (c'est justement ce qu'on prédit), donc on ne peut pas calculer de F1… sans aide.

**La solution : les retours des modérateurs.**
1. Chaque prédiction reçoit un identifiant (`prediction_id`).
2. Un modérateur qui vérifie un produit renvoie sa vraie catégorie avec `POST /feedback` (dans Streamlit : formulaire « Retour modérateur » sous chaque prédiction).
3. Le drift-monitor calcule le **F1-macro en production** sur les 500 derniers retours (à partir de 100).
4. Grafana affiche le F1 en production **à côté** du F1 d'entraînement ; sous 0,75, l'alerte `ProductionF1Low` se déclenche.

Garde-fous : un produit inconnu → 404, une catégorie inconnue → 400, un deuxième retour sur la même prédiction → 409 (même si les 2 copies de l'API le reçoivent en même temps).

**Test réel** (bouton de simulation de Streamlit : de vrais produits mis de côté, § 7.5, sont prédits puis « corrigés » avec leur vraie catégorie) :

| Situation | F1 en production | Accuracy | Prédictions peu sûres | Alertes déclenchées |
|-----------|:---------------:|:--------:|:--------------------:|---------------------|
| 300 retours sur des fiches normales | **0,823** | 85 % | 17 % | aucune |
| + 300 fiches bâclées (titre réduit à un mot) | **0,478** | 51 % | 44 % | `ProductionF1Low`, `LowConfidenceSpike`, `DataDrift` |

Les trois signaux de surveillance réagissent ensemble et de façon cohérente : le modèle se trompe (F1), il hésite (confiance), et les textes reçus ne ressemblent plus à l'entraînement (dérive). Le F1 calculé sur 300 retours fluctue de quelques points selon l'échantillon (0,77 sur l'ensemble des 2 476 produits mis de côté).

### 7.5 Confiance du modèle et produits mis de côté

**La confiance.** Pour chaque prédiction, le modèle donne un score à chaque catégorie. La **marge** entre le meilleur score et le deuxième mesure son hésitation : petite marge = il hésitait entre deux catégories. Chaque réponse de l'API contient cette confiance et un indicateur « peu sûre » ; Streamlit affiche « ⚠️ à faire vérifier par un modérateur ».

**Les produits mis de côté (*holdout*).** Pour calibrer la confiance et mesurer honnêtement, il faut des produits **avec leur vraie catégorie** que le modèle **n'a jamais vus**. Le pipeline met donc de côté une fois pour toutes **3 % de X_train (2 476 produits)**, toujours les mêmes, jamais utilisés pour l'entraînement. Ils servent à :
- **calibrer** le seuil « peu sûre » : la marge sous laquelle le modèle se trompe plus d'une fois sur deux ;
- **comparer équitablement** candidat et production lors de la promotion (§ 5.5) ;
- **simuler** des retours de modérateurs dans la démo.

Résultat du calibrage (le signal de confiance est très net) :

| Modèle | Prédictions « peu sûres » | Accuracy des peu sûres | Accuracy des sûres |
|--------|:------------------------:|:---------------------:|:-----------------:|
| Texte | 21 % | 42 % | **89 %** |
| Fusion | 16 % | 37 % | **86 %** |
| Image | 62 % | 42 % | **82 %** |

Pour le texte, l'accuracy passe de 42 % (marges les plus faibles) à 99,6 % (marges les plus fortes). Le modèle image étant faible, la majorité de ses prédictions sont incertaines : c'est cohérent. Si la part de prédictions peu sûres double (plus de 40 % en texte), l'alerte `LowConfidenceSpike` se déclenche.

### 7.6 Les 14 alertes

Visibles dans Prometheus (http://localhost:9092/alerts) et dans le panneau *Alertes actives* de Grafana.

| Alerte | Se déclenche quand… | Gravité |
|--------|---------------------|---------|
| ApiDown | plus aucune copie de l'API ne répond | critique |
| ApiReplicaDown | une copie de l'API manque | avertissement |
| ModelNotLoaded | un modèle (texte / image / fusion) n'est pas chargé | critique |
| HighErrorRate | plus de 10 % de prédictions en erreur | avertissement |
| HighLatency | 5 % des prédictions prennent plus de 2 s | avertissement |
| RateLimitHit | plus de 10 requêtes bloquées (429) en 5 min | avertissement |
| ForbiddenAccess | plus de 5 accès refusés (403) en 5 min | avertissement |
| AuthFailures | plus de 5 mauvais mots de passe en 5 min | avertissement |
| ModelF1Low | F1 d'entraînement sous 0,75 | avertissement |
| **ProductionF1Low** | **F1 en production sous 0,75** | critique |
| **LowConfidenceSpike** | plus de 40 % de prédictions texte peu sûres | avertissement |
| ModelTooOld | modèle entraîné il y a plus de 30 jours | info |
| **DataDrift** | dérive des données détectée | avertissement |
| DriftMonitorDown | plus d'analyse de dérive depuis 15 min | avertissement |

### 7.7 Front-end : Streamlit

Application de démonstration (http://localhost:8501), 8 pages : contexte, défis et modélisation, résultats (avec matrice de confusion), assistants texte / image / multimodal (avec confiance et formulaire de retour modérateur), conclusion, et **8. MLOps / API** qui sert de support à la démo : appel sans mot de passe (401), réentraînement avec le compte *user* (403), rafale de requêtes (429) et sa répartition entre les 2 copies, simulation d'une dérive et rapport Evidently, F1 en production et simulations de retours (normaux puis fiches bâclées), déclenchement d'un réentraînement.

---

## 8. Les chiffres à citer

| Sujet | Chiffre |
|-------|---------|
| Données | 84 916 produits, 27 catégories ; 82 440 pour l'entraînement, 2 476 mis de côté |
| Meilleur modèle (notebooks) | fusion texte + image : **F1-macro 0,84** |
| Modèle texte en production (version 5) | F1-macro **0,773** en validation, **0,769** sur les produits mis de côté |
| Modèle fusion servi | F1 ≈ 0,74 (export sur échantillon de 4 000 produits) |
| Confiance (texte) | 21 % de prédictions peu sûres (42 % justes) ; les sûres : 89 % justes |
| F1 en production (retours simulés) | 0,823 sur des fiches normales ; 0,478 avec des fiches bâclées (3 alertes déclenchées) |
| Services Docker | 14 (dont l'API en 2 copies) |
| Tests | 73, lancés à chaque push (CI GitHub verte) |
| Alertes | 14 règles |
| Graphiques Grafana | 17 |
| Pipeline | 5 étapes, ≈ 2 à 3 minutes |
| Sécurité | 401 / 403 / 429 ; 30 prédictions par minute pour un *user* |
| Dérive | vrais produits : distances ≤ 0,11 ; produits inhabituels : 0,57 à 5 |
| Bascule en cas de panne | 30 requêtes sur 30 réussies avec une copie de l'API arrêtée |

---

## 9. Problèmes rencontrés et solutions

Ces points sont précieux pour la partie « difficultés » du rapport : ils montrent une vraie démarche d'ingénierie.

### 9.1 Deux versions MLOps développées en parallèle

Nicolas (scripts, Docker, MLflow, stratégie de réentraînement) et Tristan (API multimodale, Streamlit, monitoring) avaient chacun construit une version. **Solution** : fusion en gardant le meilleur de chacun — les scripts et la promotion de Nicolas, l'API et le monitoring de Tristan.

### 9.2 Des briques « décoratives »

Au départ, ZenML, MinIO et MySQL démarraient mais ne servaient à rien. **Solution** : les brancher pour de vrai (pipeline ZenML, MinIO comme stockage de ZenML, MLflow et DVC, MySQL comme base de ZenML).

### 9.3 Comparaison de modèles injuste

Chaque modèle était évalué sur sa propre validation tirée au hasard : un écart de 1 point pouvait n'être que du hasard. **Solution** : les 2 476 produits mis de côté, sur lesquels candidat et production sont notés au moment de la comparaison. Transition : l'ancien modèle (version 4) avait été entraîné sur ces produits et obtenait un score gonflé (0,873) ; le premier modèle propre (version 5, 0,769) a donc été promu une fois de force, de façon documentée.

### 9.4 Un vrai défaut trouvé grâce à la détection de dérive

Sur de vrais produits, la dérive signalait à tort trop de chiffres dans les textes. **Cause** : l'API ne nettoyait pas le HTML des textes reçus (contrairement à l'entraînement) : le modèle voyait en production des codes comme `&#233;` jamais vus à l'entraînement. **Solution** : un nettoyage commun à l'entraînement et à la production (`src/preprocessing/serving.py`) + un test.

### 9.5 Seuils mal réglés

- Dérive : avec les réglages par défaut d'Evidently, de vrais produits déclenchaient une dérive. **Solution** : calibrage sur de vraies données (§ 7.3).
- Confiance : un premier critère de seuil marquait 84 % des prédictions image comme peu sûres. **Solution** : un critère « local » (là où le modèle se trompe plus souvent qu'il n'a raison), robuste au bruit.

### 9.6 Pannes trouvées en cassant volontairement le système

En arrêtant une copie de l'API : nginx renvoyait une erreur au moment de la panne, et l'alerte ne se déclenchait pas (une copie arrêtée disparaît de la liste au lieu d'apparaître « en panne »). **Solution** : nginx rejoue la prédiction sur l'autre copie, et l'alerte compte les copies présentes.

### 9.7 Autres corrections

- Prometheus ignorait les premiers refus 429 (compteurs créés trop tard) → compteurs créés à 0 au démarrage.
- Les tests restaient bloqués dans Docker (pytest parcourait 85 000 images) → `pytest.ini`.
- Les CSV étaient déjà dans Git, incompatible avec DVC → retirés de Git (avec l'accord de l'équipe, impact documenté § 5.2).
- Migration de la base ZenML (version 0.67 → 0.98) bloquée → base recréée (elle était vide).

---

## 10. Limites et pistes d'amélioration

À dire honnêtement au jury : c'est attendu et apprécié.

| Limite | Piste |
|--------|-------|
| Seul le modèle **texte** est réentraîné ; image et fusion (CLIP, ResNet) restent figés (trop lourds sans carte graphique) | serveur GPU, pipeline dédié |
| Le modèle fusion servi (F1 ≈ 0,74) est exporté sur un échantillon ; le 0,84 vient du notebook complet | export sur toutes les données |
| Le F1 en production dépend des retours des modérateurs (simulés dans la démo) | vraie interface de modération |
| Une baisse du F1 en production alerte mais ne réentraîne pas : il faudrait d'abord ajouter les retours aux données d'entraînement | boucle retours → données d'entraînement |
| Dérive mesurée sur des variables simples, pas sur le sens des mots | descripteurs de texte (embeddings) d'Evidently |
| Stockage DVC = MinIO local à chaque machine ; images non versionnées | stockage S3 partagé (AWS, DagsHub) |
| Sécurité simple : comptes dans des variables d'environnement, pas de HTTPS | base d'utilisateurs, jetons (OAuth2 / JWT), HTTPS |
| Toutes les copies de l'API tournent sur une seule machine | Kubernetes |
| Alertes visibles dans Prometheus / Grafana, mais aucune notification (e-mail, Slack) | Alertmanager |

---

## 11. Démo, commandes et adresses

### Lancer le projet

```powershell
cd "F:\Projets\Projet ecole\Datascientest_projets"
# Docker Desktop doit être démarré
.\make.ps1 up          # lance les 14 services (≈ 3 min)
.\make.ps1 train       # pipeline d'entraînement (≈ 3 min)
.\make.ps1 test        # les 73 tests
.\make.ps1 down        # tout arrêter
```

Autres commandes : `.\make.ps1 data-pull` / `data-push` (DVC), `.\make.ps1 drift` (analyse de dérive immédiate), `.\make.ps1 health`, `.\make.ps1 help`.

### Adresses

| Outil | Adresse | Identifiants |
|-------|---------|--------------|
| Streamlit (démo) | http://localhost:8501 | — |
| API (Swagger) | http://localhost:8000/docs | `admin` / `rakuten` ou `demo` / `demo` |
| ZenML | http://localhost:8888 | `default` / `zenml` |
| MLflow | http://localhost:5000 | — |
| MinIO | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Prometheus | http://localhost:9092 | — |
| Grafana | http://localhost:3002 | `admin` / `rakuten` |

### Scénario de démo

Le déroulé détaillé (≈ 7 minutes, quoi cliquer, quoi dire) est dans **`rendus/checklist_demo_mlops.md`**.

---

## 12. Glossaire

| Terme | Explication simple |
|-------|-------------------|
| **Accuracy** | Pourcentage de bonnes réponses. |
| **Alerte** | Règle qui se déclenche quand un compteur dépasse un seuil (ex. l'API ne répond plus). |
| **API** | Point d'entrée web d'un programme : on lui envoie une demande, il répond. |
| **Artefact** | Fichier produit par une étape (modèle, données nettoyées, rapport). |
| **Authentification / autorisation** | « Qui es-tu ? » (identifiant + mot de passe) / « As-tu le droit ? » (rôle). |
| **CI (intégration continue)** | Lancement automatique des tests à chaque envoi de code. |
| **Conteneur (Docker)** | Boîte contenant un programme et tout ce dont il a besoin ; identique partout. |
| **Dérive (drift)** | Les données reçues en production ne ressemblent plus à celles de l'entraînement. |
| **DVC** | Outil qui versionne les gros fichiers de données (Git ne garde que leur empreinte). |
| **Empreinte (md5)** | Code de 32 caractères calculé à partir d'un fichier ; change si le fichier change. |
| **Evidently** | Librairie qui compare deux jeux de données et détecte la dérive. |
| **F1-macro** | Note principale du projet : moyenne, sur les 27 catégories, d'un score qui combine précision et rappel (§ 1). |
| **FastAPI** | Librairie Python pour écrire des API. |
| **Feedback (retour)** | Vraie catégorie donnée par un modérateur pour un produit déjà prédit. |
| **Grafana** | Outil de tableaux de bord (graphiques). |
| **Holdout** | Produits mis de côté, jamais utilisés pour entraîner, pour mesurer honnêtement. |
| **Marge / confiance** | Écart entre le meilleur et le deuxième score du modèle : petit = le modèle hésite. |
| **MinIO** | Stockage de fichiers compatible Amazon S3, installé en local. |
| **MLflow** | Carnet de bord des entraînements + registre des versions de modèles. |
| **MLOps** | Pratiques pour mettre et maintenir un modèle de machine learning en production. |
| **Model Registry** | Bibliothèque des versions d'un modèle ; une étiquette désigne celle en production. |
| **nginx** | Serveur qui répartit les requêtes entre plusieurs copies de l'API. |
| **Pipeline** | Suite d'étapes automatisées (collecter → contrôler → nettoyer → entraîner → promouvoir). |
| **Prometheus** | Outil qui relève régulièrement des compteurs et déclenche les alertes. |
| **Promotion** | Mise en production d'un nouveau modèle, seulement s'il est au moins aussi bon. |
| **Redis** | Mémoire partagée très rapide entre plusieurs programmes. |
| **Réplique** | Copie identique d'un service (ici, 2 copies de l'API). |
| **Rollback** | Retour à une version précédente du modèle. |
| **Scheduler** | Programme qui déclenche une tâche automatiquement (ici, le réentraînement). |
| **SVM** | Modèle de classification qui sépare les catégories ; celui utilisé pour le texte. |
| **Swagger** | Page web de documentation interactive de l'API (`/docs`). |
| **TF-IDF** | Transformation d'un texte en nombres selon l'importance de chaque mot. |
| **ZenML** | Outil d'orchestration : exécute le pipeline et garde l'historique de chaque exécution. |
| **401 / 403 / 409 / 429** | Codes de réponse HTTP : non identifié / pas le droit / déjà fait / trop de requêtes. |

---

## 13. Où trouver quoi dans le projet

| Dossier / fichier | Contenu |
|-------------------|---------|
| `notebooks/` | Partie data science (01–06 texte, 07–11 image, 12 fusion) |
| `src/` | Code réutilisable : nettoyage, modèles, prédiction, monitoring |
| `src/inference/predictor.py` | Chargement des modèles, routage, confiance |
| `src/monitoring/` | Dérive (`drift.py`) et F1 en production (`performance.py`) |
| `src/state.py` | Mémoire partagée Redis (limites, journal, retours) |
| `src/preprocessing/serving.py` | Nettoyage des textes reçus par l'API |
| `scripts/` | Étapes du pipeline : collecte, contrôle qualité, nettoyage, entraînement, promotion, holdout, calibrage |
| `pipelines/` | Pipeline ZenML, scheduler, drift-monitor |
| `api/` | API FastAPI (`main.py`) et sécurité (`security.py`) |
| `app.py` | Application Streamlit |
| `tests/` | Les 73 tests |
| `monitoring/` | Configuration Prometheus, règles d'alerte, tableau de bord Grafana |
| `deploy/nginx.conf` | Répartiteur de charge |
| `docker-compose.yml`, `Dockerfile.*` | Les 14 services Docker |
| `data brut/*.dvc` | Empreintes des données (DVC) |
| `make.ps1` / `Makefile` | Toutes les commandes |
| `ARCHITECTURE.md` | Documentation technique détaillée |
| `docs/STRATEGIE_REENTRAINEMENT.md` | Stratégie de réentraînement, promotion, retour arrière |
| `rendus/` | Rapports data science, résumé MLOps, pitch, script oral, checklist de démo |

---

## 14. Plan de rapport suggéré

1. **Introduction** — contexte Rakuten, problème, objectifs (§ 1, § 2).
2. **Données et modélisation** — exploration, nettoyage, modèles texte / image / fusion, résultats (reprendre `rendus/rapport_partie_*.md`).
3. **Architecture MLOps** — schéma et rôle de chaque service (§ 3).
4. **Phase 1 — Fondations** : scripts, environnements, API, Git, tests (§ 4).
5. **Phase 2 — Microservices et versioning** : MLflow, DVC, registre, Docker, stratégie de réentraînement (§ 5).
6. **Phase 3 — Orchestration et déploiement** : ZenML, sécurité, automatisation, CI, scalabilité (§ 6).
7. **Phase 4 — Surveillance** : monitoring, qualité, dérive, F1 en production, confiance, alertes, Streamlit (§ 7).
8. **Difficultés et solutions** (§ 9).
9. **Limites et perspectives** (§ 10).
10. **Conclusion** — d'un modèle de notebook (F1 0,84) à un service sécurisé, répliqué, versionné, réentraîné automatiquement et surveillé (données et modèle).
11. **Annexes** — glossaire (§ 12), commandes (§ 11), arborescence (§ 13).

> Les chiffres du § 8 sont tous vérifiés sur le projet en fonctionnement. Si vous relancez des entraînements, les numéros de version et les F1 peuvent légèrement changer : vérifiez dans MLflow avant de rendre le rapport.
