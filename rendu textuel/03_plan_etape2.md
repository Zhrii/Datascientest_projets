# Plan d'Action - Étape 2 : Pre-processing et Feature Engineering

**Deadline : Vendredi 3 octobre**

## 📋 Vue d'Ensemble

L'objectif de cette étape est de transformer les données brutes en un dataset prêt pour la modélisation, en résolvant les problèmes identifiés lors de l'exploration.

---

## 🎯 Objectifs Principaux

1. **Nettoyer les données** : Supprimer le HTML, normaliser les textes
2. **Gérer les valeurs manquantes** : Stratégie adaptée pour les 35% de descriptions vides
3. **Feature Engineering** : Créer des features pertinentes pour la classification
4. **Préparer pour la modélisation** : Dataset final prêt pour ML/DL

---

## 📊 Constats de l'Étape 1 à Résoudre

### Problèmes Identifiés

1. **HTML dans les descriptions** (18.42%)
   - Tags fréquents : `<br/>`, `<li>`, `<p>`, `<strong>`
   - Dépendant de la classe (certaines classes >90% HTML)

2. **Valeurs manquantes** (35.09% de descriptions)
   - Forte variabilité par classe (2403: 97%, 2280: 93% vs 2582: 2.4%)
   - Impact sur la qualité des données disponibles

3. **Variabilité des longueurs**
   - Désignations : 11-250 caractères (médiane 64)
   - Descriptions : 0-12 451 caractères (médiane 231)
   - Classes avec textes très longs (2280) ou très courts (1160, 2403)

4. **Textes multilingues**
   - Français et anglais principalement
   - Nécessité d'un preprocessing adapté

5. **Déséquilibre de classes**
   - Ratio 13.36 (classe 2583: 10 209 vs classe 1180: 764)
   - Nécessite des techniques de rééquilibrage

---

## 🔧 Plan de Pre-processing

### Phase 1 : Nettoyage HTML (Priorité Haute)

**Objectif :** Extraire le texte utile des descriptions contenant du HTML

**Actions :**
1. **Détection du HTML**
   - Utiliser BeautifulSoup ou regex pour identifier les descriptions avec HTML
   - Créer une feature `has_html` (déjà fait en étape 1)

2. **Nettoyage HTML**
   - Extraire le texte des balises HTML
   - Gérer les entités HTML (`&eacute;`, `&#39;`, etc.)
   - Conserver la structure sémantique si possible (paragraphes → sauts de ligne)

3. **Validation**
   - Vérifier que le texte extrait est cohérent
   - Comparer les longueurs avant/après nettoyage
   - Statistiques par classe

**Fonctions à créer :**
```python
def clean_html(text):
    """Nettoie le HTML d'un texte"""
    # Utiliser BeautifulSoup ou regex
    pass

def decode_html_entities(text):
    """Décode les entités HTML"""
    pass
```

---

### Phase 2 : Gestion des Valeurs Manquantes (Priorité Haute)

**Objectif :** Traiter les 35% de descriptions manquantes de manière optimale

**Stratégies à tester :**

1. **Option A : Utilisation de la designation seule**
   - Pour les produits sans description, utiliser uniquement la designation
   - Créer une feature `has_description` pour indiquer au modèle

2. **Option B : Marqueur spécial**
   - Remplacer les descriptions vides par `[DESCRIPTION_VIDE]`
   - Permet au modèle de distinguer les cas

3. **Option C : Imputation par classe**
   - Imputer avec une description générique par classe
   - Risque : peut introduire du biais

**Recommandation :** Tester Option A et Option B, comparer les performances

**Fonctions à créer :**
```python
def handle_missing_descriptions(df, strategy='designation_only'):
    """Gère les descriptions manquantes selon la stratégie choisie"""
    pass
```

---

### Phase 3 : Normalisation des Textes (Priorité Moyenne)

**Objectif :** Standardiser les textes pour améliorer la classification

**Actions :**

1. **Normalisation de base**
   - Conversion en minuscules (ou garder la casse si discriminante)
   - Suppression des espaces multiples
   - Normalisation des caractères spéciaux

2. **Gestion multilingue**
   - Détection de la langue (optionnel)
   - Tokenisation adaptée (français/anglais)
   - Conservation des accents (importants pour le français)

3. **Normalisation des longueurs**
   - Padding/truncation pour les modèles nécessitant des longueurs fixes
   - Ou conservation de la longueur variable selon le modèle choisi

**Fonctions à créer :**
```python
def normalize_text(text, lowercase=True, remove_extra_spaces=True):
    """Normalise un texte"""
    pass

def detect_language(text):
    """Détecte la langue d'un texte (optionnel)"""
    pass
```

---

### Phase 4 : Feature Engineering (Priorité Moyenne)

**Objectif :** Créer des features supplémentaires pour améliorer la classification

**Features à créer :**

1. **Features de longueur**
   - `designation_length` : Longueur de la designation (caractères, mots)
   - `description_length` : Longueur de la description (caractères, mots)
   - `total_text_length` : Somme des deux

2. **Features binaires**
   - `has_description` : Présence d'une description (booléen)
   - `has_html` : Présence de HTML (booléen, avant nettoyage)
   - `is_description_empty` : Description vide ou non

3. **Features de qualité**
   - `description_quality_score` : Score basé sur longueur, présence HTML, etc.
   - `text_completeness` : Ratio designation_length / (designation_length + description_length)

4. **Features textuelles (optionnel)**
   - Nombre de mots uniques
   - Densité de mots-clés par classe
   - Présence de mots techniques spécifiques

**Fonctions à créer :**
```python
def create_length_features(df):
    """Crée les features de longueur"""
    pass

def create_binary_features(df):
    """Crée les features binaires"""
    pass

def create_quality_features(df):
    """Crée les features de qualité"""
    pass
```

---

### Phase 5 : Préparation pour la Modélisation (Priorité Haute)

**Objectif :** Créer le dataset final prêt pour ML/DL

**Actions :**

1. **Combinaison des textes**
   - Fusionner designation + description en un seul texte
   - Ou garder séparés selon le modèle choisi
   - Séparateur approprié (`[SEP]` ou espace)

2. **Encodage des labels**
   - Vérifier que les labels sont bien encodés (numériques)
   - Créer un mapping label → nom de classe (optionnel)

3. **Sauvegarde des datasets**
   - Dataset nettoyé (train/test)
   - Features supplémentaires
   - Métadonnées (mapping classes, statistiques)

4. **Validation finale**
   - Vérifier la cohérence train/test
   - Statistiques descriptives du dataset final
   - Vérifier qu'il n'y a pas de fuite de données

**Fonctions à créer :**
```python
def combine_texts(df, separator=' '):
    """Combine designation et description"""
    pass

def prepare_final_dataset(X_train, X_test, y_train):
    """Prépare le dataset final pour la modélisation"""
    pass
```

---

## 📁 Structure Proposée

```
Projet ecole/
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── html_cleaner.py      # Nettoyage HTML
│   │   ├── text_normalizer.py  # Normalisation texte
│   │   ├── feature_engineering.py  # Création de features
│   │   └── pipeline.py          # Pipeline complet
│   │
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py       # Chargement des données
│
├── notebooks/
│   ├── 01_exploration_data.ipynb  # (déjà fait)
│   └── 02_preprocessing.ipynb     # NOUVEAU : Preprocessing complet
│
├── data/
│   ├── raw/                      # Données brutes (data brut/)
│   ├── processed/                # Données nettoyées
│   │   ├── X_train_clean.csv
│   │   ├── X_test_clean.csv
│   │   └── y_train.csv
│   └── features/                 # Features supplémentaires
│       └── features_train.csv
│
└── reports/
    └── preprocessing_report.md   # Rapport de preprocessing
```

---

## 📝 Notebook de Pre-processing

**Structure proposée pour `02_preprocessing.ipynb` :**

1. **Introduction et Configuration**
   - Objectifs du preprocessing
   - Chargement des données brutes

2. **Phase 1 : Nettoyage HTML**
   - Détection du HTML
   - Nettoyage avec BeautifulSoup
   - Validation et statistiques

3. **Phase 2 : Gestion des Valeurs Manquantes**
   - Analyse des patterns de valeurs manquantes
   - Implémentation des stratégies
   - Comparaison des approches

4. **Phase 3 : Normalisation des Textes**
   - Normalisation de base
   - Gestion multilingue
   - Normalisation des longueurs

5. **Phase 4 : Feature Engineering**
   - Création des features de longueur
   - Création des features binaires
   - Création des features de qualité

6. **Phase 5 : Préparation Finale**
   - Combinaison des textes
   - Sauvegarde des datasets
   - Validation finale

7. **Synthèse et Statistiques**
   - Comparaison avant/après preprocessing
   - Statistiques du dataset final
   - Recommandations pour la modélisation

---

## 📊 Métriques de Succès

### Qualité du Nettoyage

- **HTML :** 100% des descriptions HTML nettoyées
- **Entités HTML :** Toutes décodées correctement
- **Valeurs manquantes :** Stratégie claire et documentée

### Qualité des Features

- **Features créées :** Au moins 5-10 features pertinentes
- **Features validées :** Pas de corrélations parfaites, pas de fuite de données
- **Documentation :** Chaque feature documentée

### Préparation pour la Modélisation

- **Dataset final :** Cohérent, sans erreurs
- **Train/Test :** Distributions similaires
- **Format :** Prêt pour scikit-learn, transformers, etc.

---

## 🎯 Livrables Attendus

### 1. Notebook de Pre-processing (`02_preprocessing.ipynb`)
- Code complet et commenté
- Visualisations avant/après preprocessing
- Statistiques et validations

### 2. Scripts Python modulaires (`src/preprocessing/`)
- Code réutilisable et testable
- Fonctions bien documentées

### 3. Datasets nettoyés (`data/processed/`)
- X_train_clean.csv
- X_test_clean.csv
- Features supplémentaires

### 4. Rapport de Pre-processing (PDF)
- Description des transformations effectuées
- Justification des choix
- Statistiques avant/après
- Visualisations clés

---

## ⏱️ Planning Suggéré

### Semaine 1
- **Jour 1-2 :** Phase 1 (Nettoyage HTML)
- **Jour 3-4 :** Phase 2 (Valeurs manquantes)
- **Jour 5 :** Phase 3 (Normalisation)

### Semaine 2
- **Jour 1-2 :** Phase 4 (Feature Engineering)
- **Jour 3 :** Phase 5 (Préparation finale)
- **Jour 4-5 :** Rédaction du rapport et finalisation

---

## 🔍 Points d'Attention

1. **Cohérence Train/Test**
   - Appliquer les mêmes transformations aux deux datasets
   - Vérifier que les distributions restent similaires

2. **Pas de Fuite de Données**
   - Ne pas utiliser les labels pour le preprocessing
   - Ne pas utiliser les données de test pour entraîner le preprocessing

3. **Reproductibilité**
   - Sauvegarder les paramètres de preprocessing
   - Versionner le code
   - Documenter toutes les transformations

4. **Performance**
   - Le preprocessing doit être rapide (quelques minutes max)
   - Optimiser pour les gros volumes de données

---

## 🚀 Prochaines Étapes (Après Étape 2)

- **Étape 3 :** Modélisation (baselines puis optimisation)
- **Étape 4 :** Évaluation et métriques
- **Étape 5 :** Optimisation et fine-tuning

---

**Prêt à commencer ? 🎉**

