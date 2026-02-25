# Structure du Projet - Vue d'Ensemble

## 📁 Organisation des Fichiers

```
Datascientest_projets/
│
├── 📊 data brut/                    # Données brutes (NE PAS MODIFIER)
│   ├── X_train_update.csv           # 84 916 produits d'entraînement
│   ├── X_test_update.csv            # 13 814 produits de test
│   ├── Y_train.csv                  # Labels d'entraînement
│   └── README.md                    # Instructions pour les données
│
├── 📦 data/                         # Données traitées
│   └── processed/
│       ├── X_train_clean.csv        # Dataset d'entraînement nettoyé
│       ├── X_test_clean.csv        # Dataset de test nettoyé
│       ├── y_train.csv              # Labels
│       └── image_train/             # Images d'entraînement (pour matching texte-image)
│           └── image_{imageid}_product_{productid}.jpg
│
├── 📓 notebooks/                    # Notebooks d'analyse
│   ├── 01_texte_exploration_donnees.ipynb   # Section texte : exploration des données
│   ├── 02_texte_traitement_donnees.ipynb    # Section texte : preprocessing pipeline
│   ├── 03_texte_resultats_traitement.ipynb  # Section texte : validation et statistiques
│   ├── 04_texte_modelisation_baseline.ipynb       # Texte : Modélisation baseline (sklearn, CatBoost, etc.)
│   ├── 05_texte_optimisation_modeles.ipynb        # Texte : Optimisation (poids, SMOTE, etc.)
│   ├── 06_texte_modelisation_avancee.ipynb        # Texte : Ensembles, Deep Learning (MLP, CNN, LSTM)
│   ├── 07_exploration_images_classic.ipynb  # Exploration et classification par images (CNN)
│   ├── 08_matching_texte_image.ipynb        # Matching texte-image (Dual Encoder, contrastive)
│   └── catboost_info/                       # Logs CatBoost (générés)
│
├── 💻 src/                          # Code source modulaire
│   ├── preprocessing/               # Nettoyage et préparation du texte
│   │   ├── html_cleaner.py
│   │   ├── text_normalizer.py
│   │   ├── feature_engineering.py
│   │   ├── pipeline.py
│   │   └── __init__.py
│   ├── utils/                       # Utilitaires
│   │   ├── data_loader.py           # Chargement/sauvegarde données
│   │   └── __init__.py
│   ├── modeling/                    # Modèles et vectorisation
│   │   ├── vectorization.py         # TF-IDF, create_vectorizer
│   │   ├── baseline_models.py       # Modèles baseline, train_all_models
│   │   ├── advanced_models.py       # Modèles avancés
│   │   └── __init__.py
│   ├── evaluation/                  # Métriques et visualisations
│   │   ├── metrics.py               # evaluate_model, classification_report, confusion_matrix
│   │   └── __init__.py
│   ├── optimization/                # Optimisation et rééquilibrage
│   │   ├── class_balancing.py
│   │   ├── hyperparameter_tuning.py
│   │   └── __init__.py
│   ├── deep_learning/               # Réseaux de neurones (TensorFlow/Keras)
│   │   ├── neural_networks.py      # MLP, CNN, LSTM, train_neural_network
│   │   └── __init__.py
│   ├── interpretability/            # Interprétabilité (SHAP, feature importance)
│   │   ├── feature_importance.py
│   │   ├── shap_analysis.py
│   │   └── __init__.py
│   ├── multimodal/                 # Matching texte-image (PyTorch)
│   │   ├── data_loader.py           # Paires (texte, image), load_image
│   │   ├── encoders.py              # TextEncoder, ImageEncoder
│   │   ├── matching_model.py        # DualEncoderModel, loss contrastive
│   │   ├── utils.py                 # recall_at_k, find_matching_images
│   │   └── __init__.py
│   └── __init__.py
│
├── 📂 models/                       # Modèles sauvegardés et résultats (générés)
│   ├── *.pkl                        # Modèles sklearn / baseline
│   ├── final_results_step3.csv
│   ├── conclusions_step3.json
│   └── model_comparison_results.csv
│
├── 📄 rendu textuel/                # Rapports et documentation
│   ├── RAPPORT_COMPLET_ETAPE_1_ET_2.md   # ⭐ Rapport principal (exploration + préprocessing)
│   ├── Template - Rapport exploration des données.xlsx
│   └── README.md
│
├── 📋 class_identification.md       # Description des 27 classes (prdtypecode)
├── 📋 PLAN_MATCHING_TEXTE_IMAGE.md   # Plan technique du matching texte-image (Dual Encoder)
├── ⚙️ requirements.txt              # Dépendances Python
├── 🔒 .gitignore
└── 📖 README.md                     # Documentation principale
```

---

## 🎯 Fichiers par étape

### Section Texte : Exploration (Étape 1)
- `notebooks/01_texte_exploration_donnees.ipynb`
- `rendu textuel/RAPPORT_COMPLET_ETAPE_1_ET_2.md` (sections 1–4)

### Section Texte : Traitement (Étape 2)
- `notebooks/02_texte_traitement_donnees.ipynb`
- `src/preprocessing/` (pipeline, nettoyage HTML, normalisation)
- `rendu textuel/RAPPORT_COMPLET_ETAPE_1_ET_2.md` (section 5)

### Section Texte : Résultats du traitement (Étape 2b)
- `notebooks/03_texte_resultats_traitement.ipynb`
- `data/processed/` (X_train_clean, X_test_clean, y_train)

### Modélisation baseline (Étape 3)
- `notebooks/04_texte_modelisation_baseline.ipynb`
- `src/modeling/` (vectorization, baseline_models)
- `data/processed/*.csv`

### Optimisation
- `notebooks/05_texte_optimisation_modeles.ipynb`
- `src/optimization/` (class_balancing, hyperparameter_tuning)

### Modélisation avancée
- `notebooks/06_texte_modelisation_avancee.ipynb`
- `src/deep_learning/` (MLP, CNN, LSTM)
- `src/interpretability/` (SHAP, feature importance)
- `src/evaluation/` (métriques, rapports, matrices de confusion)
- `models/` (résultats et modèles sauvegardés)

### Exploration images
- `notebooks/07_exploration_images_classic.ipynb` (classification par CNN sur images)

### Matching texte-image
- `notebooks/08_matching_texte_image.ipynb` (entraînement Dual Encoder)
- `src/multimodal/` (data_loader, encoders, matching_model, utils)
- `PLAN_MATCHING_TEXTE_IMAGE.md` (choix techniques et architecture)

---

## 📝 Livrables

### Rapport principal
**Fichier :** `rendu textuel/RAPPORT_COMPLET_ETAPE_1_ET_2.md`

- Exploration des données et visualisations
- Pre-processing détaillé
- Statistiques avant/après et recommandations

### Résultats et modèles
- `models/` : modèles sauvegardés (`.pkl`), comparaisons et conclusions (Step 3)

---

## ✅ Résumé

- **Données** : `data brut/` (brutes), `data/processed/` (nettoyées + images pour matching)
- **Code** : `notebooks/` (analyse) et `src/` (preprocessing, modeling, evaluation, optimization, deep_learning, interpretability, multimodal)
- **Documentation** : `rendu textuel/`, `README.md`, `PLAN_MATCHING_TEXTE_IMAGE.md`, `class_identification.md`
- **Configuration** : `requirements.txt`
