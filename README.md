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

> ⚠️ Les fichiers de données (`*.csv`, images) ne sont **pas versionnés** dans Git. Après clonage, placer les fichiers dans `data brut/` selon les instructions de `data brut/README.md`.

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
├── data brut/              # Données brutes (CSV, non versionnées)
├── data/processed/         # Données nettoyées + images (image_train/)
├── notebooks/              # Analyse et entraînement
│   ├── 01_texte_exploration_donnees.ipynb   # Section texte : exploration
│   ├── 02_texte_traitement_donnees.ipynb    # Section texte : preprocessing
│   ├── 03_texte_resultats_traitement.ipynb  # Section texte : validation
│   ├── 04_texte_modelisation_baseline.ipynb
│   ├── 05_texte_optimisation_modeles.ipynb
│   ├── 06_texte_modelisation_avancee.ipynb
│   ├── 07_exploration_images_classic.ipynb
│   └── 08_matching_texte_image.ipynb
├── src/                    # Code modulaire
│   ├── preprocessing/     # Nettoyage texte, pipeline
│   ├── utils/              # Chargement données
│   ├── modeling/           # Vectorisation, modèles baseline/avancés
│   ├── evaluation/        # Métriques, rapports, matrices de confusion
│   ├── optimization/      # Rééquilibrage, tuning
│   ├── deep_learning/     # MLP, CNN, LSTM (TensorFlow/Keras)
│   ├── interpretability/  # SHAP, feature importance
│   └── multimodal/        # Matching texte-image (PyTorch)
├── models/                 # Modèles sauvegardés, résultats (générés)
├── rendu textuel/          # Rapports (exploration, préprocessing)
├── class_identification.md
├── PLAN_MATCHING_TEXTE_IMAGE.md   # Descriptif technique matching
├── PROJET_STRUCTURE.md     # Vue détaillée de l’arborescence
├── requirements.txt
└── README.md
```

Pour le détail des fichiers et des rôles par étape, voir **PROJET_STRUCTURE.md**.

---

## 🚀 Installation et Démarrage

### Prérequis

- **Python 3.8+** (recommandé : 3.10 ou 3.11)
- **pip**

### Installation

```bash
# Cloner le projet
git clone <url-du-repo>
cd Datascientest_projets

# Environnement virtuel (recommandé)
python -m venv venv
# Windows PowerShell :
.\venv\Scripts\Activate.ps1
# Linux/Mac :
source venv/bin/activate

# Dépendances
pip install -r requirements.txt
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
| **01** | Exploration des données texte, visualisations |
| **02** | Préprocessing (HTML, normalisation), génération des jeux nettoyés |
| **03** | Modélisation baseline (sklearn, CatBoost, etc.) |
| **04** | Optimisation (poids de classes, SMOTE, etc.) |
| **05** | Modèles avancés (ensembles, MLP/CNN/LSTM), interprétabilité (SHAP) |
| **06** | Exploration et classification par images (CNN) |
| **07** | **Matching texte-image** : Dual Encoder, loss contrastive, Recall@K |
| **08** | Suivi des tâches du projet matching |

- **Classification (texte)** : enchaînement logique 01 → 02 → 03 → 04 → 05 (données dans `data/processed/`).
- **Matching texte-image** : 07 (et 08 pour le suivi). Données : CSV dans `data brut/`, images dans `data/processed/image_train/`. Détails techniques dans **PLAN_MATCHING_TEXTE_IMAGE.md**.

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
- **PROJET_STRUCTURE.md** : arborescence détaillée et fichiers par étape.
- **PLAN_MATCHING_TEXTE_IMAGE.md** : objectifs, architecture Dual Encoder, choix des librairies, implémentation, métriques (Recall@K), pour le rendu matching.
- **class_identification.md** : description des 27 classes (prdtypecode).
- **rendu textuel/** : rapport exploration + préprocessing.

---

## 🔧 Dépannage

- **ModuleNotFoundError** (ex. `torch`, `pandas`) : vérifier l’environnement activé et le kernel Jupyter, puis `pip install -r requirements.txt`.
- **Données manquantes** : placer les CSV et, pour le matching, les images selon `data brut/README.md` et la convention des noms dans `data/processed/image_train/`.
- **Encodage** : le projet suppose UTF-8 ; sous Windows, privilégier PowerShell.

---

## ✅ Livrables et critères

- Exploration et préprocessing documentés (notebooks 01–02, rapport dans `rendu textuel/`).
- Pipeline de classification texte (baseline → optimisation → modèles avancés) avec métriques et interprétabilité.
- Module de matching texte-image (Dual Encoder, entraînement, évaluation Recall@K, sauvegarde du meilleur modèle) avec descriptif technique dans **PLAN_MATCHING_TEXTE_IMAGE.md**.

---

**Bonne exploration et bon entraînement ! 🎉**
