# Projet de Classification de Produits E-commerce

## 📋 Contexte du Projet

Ce projet développe un système de **classification automatique** et de **matching multimodale** pour un catalogue e-commerce. Il vise à :

1. **Prédire le code type de produit** (`prdtypecode`, 27 classes) à partir du **texte** (désignation + description).
2. **Associer textes et images** : recherche par similarité (texte → image, image → texte) via un modèle Dual Encoder entraîné sur les paires produit.

### Problématique

Dans un contexte e-commerce, ces outils permettent de :
- **Améliorer l’expérience utilisateur** : navigation, recherche de produits
- **Automatiser le catalogage** : catégorisation et cohérence texte/image
- **Améliorer les recommandations** : produits similaires, recherche multimodale
- **Réduire les erreurs** : standardisation et détection d’incohérences

---

## 📊 Données

Les données sont dans le dossier `data brut/` :

> ⚠️ Les CSV de `data brut/` sont **versionnés avec DVC**, pas avec Git : Git ne contient que les fichiers `*.csv.dvc` (empreintes), les CSV sont stockés dans MinIO. Après clonage : `.\make.ps1 up` puis `.\make.ps1 data-pull`. Les images ne sont pas versionnées (voir `data brut/README.md`).
>
> **Équipe — après avoir récupéré la branche MLOps** : les CSV ont été retirés de Git, `git pull` les supprime de votre dossier `data brut/`. Le stockage DVC (MinIO) étant local à chaque poste, recopiez-les depuis votre sauvegarde ou depuis l'archive DataScientest, puis lancez `python -m dvc status "data brut/X_train_update.csv.dvc"` pour vérifier qu'ils correspondent à la version attendue.

### Fichiers principaux

| Fichier | Description |
|--------|-------------|
| **X_train_update.csv** | ~84 916 produits : `designation`, `description`, `productid`, `imageid` |
| **X_test_update.csv** | ~13 814 produits à classifier (même structure, sans labels) |
| **Y_train.csv** | Labels `prdtypecode` pour l’entraînement |

- **Classification** : objectif = prédire `prdtypecode` (multi-classes).
- **Matching texte-image** : paires (texte, image) construites via `productid` / `imageid` ; images dans `data/processed/image_train/` (convention `image_{imageid}_product_{productid}.jpg`).

---

## 📝 Structure du Projet

```
Datascientest_projets/
├── data brut/              # Données brutes (CSV versionnés par DVC : *.csv.dvc dans Git)
├── data/processed/         # Données nettoyées + images (image_train/)
├── notebooks/              # 01–06 texte · 07–11 image · 12 fusion multimodale
├── src/                    # Code modulaire
│   ├── preprocessing/      # Nettoyage texte, pipeline
│   ├── utils/              # Chargement données
│   ├── modeling/           # Vectorisation, modèles baseline/avancés
│   ├── evaluation/         # Métriques, rapports, matrices de confusion
│   ├── optimization/       # Rééquilibrage, tuning
│   ├── deep_learning/      # MLP, CNN, LSTM (TensorFlow/Keras)
│   ├── interpretability/   # SHAP, feature importance
│   ├── image/              # Chargement images, extraction de features
│   ├── multimodal/         # Matching texte-image (PyTorch)
│   └── inference/          # Prédicteur partagé API / Streamlit / scripts
├── api/                    # API FastAPI (inférence, /metrics, /retrain)
├── pipelines/              # Pipeline ZenML (preprocess_data → train_model → promote_model)
├── scripts/                # Étapes : collecte, process, training, promote, predict
├── docs/                   # Stratégie de réentraînement
├── monitoring/             # Prometheus (+ alertes) + Grafana (provisioning)
├── tests/                  # Tests pytest (API, sécurité, promotion, scheduler)
├── models/                 # Modèles sauvegardés (générés)
├── artifacts/              # Métadonnées des runs (générées)
├── rendus/                 # Rapports et documentation
├── app.py                  # Application Streamlit
├── docker-compose.yml      # Stack MLOps complète
└── make.ps1 / Makefile     # Commandes (up, train, api, ...)
```

---

## 🏗️ Architecture MLOps

Workspace de référence : **`Datascientest_projets`** (données + modèles locaux).

Voir **[ARCHITECTURE.md](ARCHITECTURE.md)** (stack Docker, sécurité, alertes, tests, correspondance avec les 4 phases) et **[docs/STRATEGIE_REENTRAINEMENT.md](docs/STRATEGIE_REENTRAINEMENT.md)** (candidat → promotion → rollback).

```powershell
# 1) Docker Desktop doit être démarré (icône baleine verte)
.\make.ps1 up              # toute la stack
.\make.ps1 train           # pipeline ZenML → promotion si meilleur → rechargé à chaud par l'API
.\make.ps1 data-pull       # récupérer les CSV versionnés par DVC
.\make.ps1 test            # tests pytest
.\make.ps1 down            # stop
```

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000/docs (login `admin` / `rakuten`) |
| Streamlit | http://localhost:8501 |
| ZenML | http://localhost:8888 (`default` / `zenml`) |
| MinIO | http://localhost:9001 (`minioadmin` / `minioadmin`) |
| Prometheus | http://localhost:9092 |
| Grafana | http://localhost:3002 (`admin` / `rakuten`) |
| MLflow | http://localhost:5000 |

Sans Docker (API seule) :

```powershell
.\make.ps1 install-mlops
.\make.ps1 api
.\make.ps1 streamlit
```

---

## 🚀 Installation et Démarrage

### Prérequis

- **Python 3.8+** (recommandé : 3.10 ou 3.11)
- **pip**

### Installation

**Option rapide (tout-en-un) :**

```powershell
# Windows PowerShell - configure tout automatiquement (venv + deps + spacy + nltk + kernel Jupyter)
.\setup_env.ps1
```

**Avec Conda :**

```bash
conda env create -f environment.yml
conda activate datascientest_projet
python -m spacy download fr_core_news_sm
python -m ipykernel install --user --name=datascientest_projet --display-name="Python (Datascientest Projet)"
```

**Installation manuelle :**

```bash
# Environnement virtuel (recommandé)
python -m venv venv
# Windows PowerShell :
.\venv\Scripts\Activate.ps1
# Linux/Mac :
source venv/bin/activate

# Dépendances
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
python -m ipykernel install --user --name=datascientest_projet --display-name="Python (Datascientest Projet)"
```

### Lancer les notebooks

```bash
jupyter notebook
# ou
jupyter lab
```

Ouvrir les notebooks dans l’ordre souhaité depuis le dossier `notebooks/`. Choisir le **kernel** correspondant à l’environnement où `requirements.txt` a été installé.

**Créer un kernel dédié (optionnel) :**

```bash
pip install ipykernel
python -m ipykernel install --user --name=projet_ecole --display-name "Python (projet_ecole)"
```

Puis dans Jupyter : *Kernel → Change kernel → Python (projet_ecole)*.

---

## 📓 Parcours des Notebooks

| Notebook | Contenu |
|----------|--------|
| **01–03** | Texte : exploration, preprocessing (HTML, normalisation), validation |
| **04–06** | Texte : baseline, optimisation, modèles avancés + interprétabilité |
| **07–08** | Image : exploration, preprocessing |
| **09–11** | Image : baseline (ResNet50), optimisation, modèles avancés (CLIP, EfficientNet) |
| **12** | Fusion multimodale texte + image (TF-IDF + CLIP + SVM, F1-macro ≈ 0,84) |

---

## 🛠️ Dépendances principales

- **Données & ML classique** : pandas, numpy, scikit-learn, xgboost, lightgbm, catboost
- **Visualisation** : matplotlib, seaborn, plotly, wordcloud
- **NLP** : nltk, spacy, beautifulsoup4
- **Deep Learning (classification)** : TensorFlow
- **Matching texte-image** : PyTorch, torchvision, sentence-transformers, Pillow

Liste complète : `requirements.txt`.

---

## 📄 Documentation

- **README.md** (ce fichier) : vue d’ensemble et démarrage.
- **ARCHITECTURE.md** : stack MLOps (services Docker, ports, commandes).
- **rendus/** : rapports et référentiels (voir `rendus/README.md`)
  - `reference_classes.md` : référentiel des 27 classes produits (prdtypecode)
  - `rapport_partie_texte.md` / `rapport_partie_image.md` / `rapport_partie_fusion.md`
  - `resume_mlops.md` : synthèse de la phase MLOps

---

## 🔧 Dépannage

- **ModuleNotFoundError** (ex. `torch`, `pandas`) : vérifier l’environnement activé et le kernel Jupyter, puis `pip install -r requirements.txt`.
- **Données manquantes** : `.\make.ps1 data-pull` (MinIO doit tourner), ou placer les CSV à la main et, pour le matching, les images selon `data brut/README.md` et la convention des noms dans `data/processed/image_train/`.
- **Encodage** : le projet suppose UTF-8 ; sous Windows, privilégier PowerShell.

---

## ✅ Livrables et critères

- Exploration et préprocessing documentés (notebooks 01–02, rapports dans `rendus/`).
- Pipeline de classification texte (baseline → optimisation → modèles avancés) avec métriques et interprétabilité.
- Fusion multimodale texte + image (notebook 12, rapport `rendus/rapport_partie_fusion.md`).
- Industrialisation MLOps : API, pipeline, monitoring, tracking (voir **ARCHITECTURE.md**).

---

**Bonne exploration et bon entraînement ! 🎉**
