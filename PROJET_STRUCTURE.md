# Structure du Projet - Vue d'Ensemble

## 📁 Organisation des Fichiers

```
Datascientest_projets/
│
├── 📊 data brut/              # Données brutes (NE PAS MODIFIER)
│   ├── X_train_update.csv    # 84 916 produits d'entraînement
│   ├── X_test_update.csv     # 13 814 produits de test
│   ├── Y_train.csv           # Labels d'entraînement
│   └── README.md             # Instructions pour les données
│
├── 📦 data/                   # Données traitées
│   └── processed/            # Datasets nettoyés (générés)
│       ├── X_train_clean.csv # Dataset d'entraînement nettoyé
│       ├── X_test_clean.csv  # Dataset de test nettoyé
│       └── y_train.csv       # Labels
│
├── 📓 notebooks/             # Notebooks d'analyse
│   ├── 01_exploration_data.ipynb      # Étape 1 : Exploration
│   └── 02_preprocessing.ipynb         # Étape 2 : Pre-processing
│
├── 💻 src/                    # Code source modulaire
│   ├── preprocessing/        # Modules de preprocessing
│   │   ├── html_cleaner.py           # Nettoyage HTML
│   │   ├── text_normalizer.py        # Normalisation texte
│   │   ├── feature_engineering.py    # Création de features
│   │   └── pipeline.py               # Pipeline complet
│   └── utils/                 # Utilitaires
│       └── data_loader.py     # Chargement/sauvegarde données
│
├── 📄 rendu textuel/          # Rapports et documentation
│   ├── RAPPORT_COMPLET_ETAPE_1_ET_2.md  # ⭐ RAPPORT PRINCIPAL
│   ├── Template - Rapport exploration des données.xlsx
│   └── README.md             # Documentation des rapports
│
├── 📋 class_identification.md  # Description des 27 classes
├── ⚙️ requirements.txt         # Dépendances Python
├── 🔒 .gitignore             # Fichiers ignorés par Git
└── 📖 README.md              # Documentation principale
```

## 🎯 Fichiers Essentiels

### Pour l'Exploration (Étape 1)
- `notebooks/01_exploration_data.ipynb`
- `rendu textuel/RAPPORT_COMPLET_ETAPE_1_ET_2.md` (sections 1-4)

### Pour le Pre-processing (Étape 2)
- `notebooks/02_preprocessing.ipynb`
- `src/preprocessing/` (modules Python)
- `data/processed/` (datasets générés)
- `rendu textuel/RAPPORT_COMPLET_ETAPE_1_ET_2.md` (section 5)

### Pour la Modélisation (Étape 3+)
- `data/processed/X_train_clean.csv` (dataset nettoyé)
- `data/processed/X_test_clean.csv`
- `data/processed/y_train.csv`

## 📝 Livrables

### Rapport Principal
**Fichier :** `rendu textuel/RAPPORT_COMPLET_ETAPE_1_ET_2.md`

Ce rapport contient :
- ✅ Exploration complète des données
- ✅ 5 visualisations avec commentaires métier
- ✅ Pre-processing détaillé (5 phases)
- ✅ Statistiques avant/après
- ✅ Validation et recommandations

## ✅ Structure Finale

Le projet est maintenant organisé de manière claire et minimaliste :
- **Données** : `data brut/` (brutes) et `data/processed/` (nettoyées)
- **Code** : `notebooks/` (analyse) et `src/` (modules réutilisables)
- **Documentation** : `rendu textuel/` (rapports) et `README.md`
- **Configuration** : `requirements.txt`

