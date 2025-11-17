# Projet de Classification de Produits E-commerce

## 📋 Contexte du Projet

Ce projet consiste à développer un système de classification automatique de produits e-commerce. L'objectif est de prédire le **code de type de produit** (`prdtypecode`) à partir des informations textuelles disponibles (désignation et description).

### Problématique

Dans un contexte e-commerce, la classification automatique des produits est essentielle pour :
- **Améliorer l'expérience utilisateur** : faciliter la navigation et la recherche de produits
- **Optimiser la gestion du catalogue** : automatiser le catalogage et la catégorisation
- **Améliorer les recommandations** : suggérer des produits similaires
- **Réduire les erreurs humaines** : standardiser la classification

### Périmètre du Projet

Le projet se décompose en plusieurs étapes :
1. **Exploration des données + DataViz** (Étape actuelle - Deadline : Vendredi 5 septembre)
2. Préprocessing et nettoyage des données
3. Feature engineering
4. Modélisation et sélection de modèle
5. Évaluation et optimisation
6. Déploiement (si applicable)

---

## 📊 Données Disponibles

Les données sont situées dans le dossier `data brut/` :

> ⚠️ **Note importante** : Les fichiers de données (`*.csv`) ne sont **pas versionnés** dans Git pour des raisons de taille. Si vous clonez ce repository, vous devrez ajouter manuellement les fichiers de données dans le dossier `data brut/`. Voir `data brut/README.md` pour plus d'informations.

### Fichiers

1. **`X_train_update.csv`** - Données d'entraînement
   - **84 916 produits** avec leurs caractéristiques
   - Colonnes :
     - `designation` : Nom/titre du produit
     - `description` : Description détaillée du produit (peut contenir du HTML)
     - `productid` : Identifiant unique du produit
     - `imageid` : Identifiant de l'image associée

2. **`X_test_update.csv`** - Données de test
   - **13 814 produits** à classifier
   - Même structure que `X_train_update.csv`
   - **Pas de labels** (à prédire)

3. **`Y_train.csv`** - Labels d'entraînement
   - **84 916 labels** correspondant aux produits d'entraînement
   - Colonne : `prdtypecode` (code numérique du type de produit)
   - **Variable cible à prédire**

### Caractéristiques des Données

- **Type de problème** : Classification multi-classes (classification supervisée)
- **Variables textuelles** : `designation` et `description` (NLP)
- **Variables numériques** : `productid`, `imageid` (identifiants)
- **Variable cible** : `prdtypecode` (catégorie de produit)

---

## 🎯 Objectifs de l'Étape 1

**Deadline : Vendredi 5 septembre**

Prendre en main et découvrir votre jeu de données et faire une analyse presque exhaustive de celui-ci afin de mettre en lumière :
- La **structure** du dataset
- Les **difficultés** rencontrées
- Les **biais** éventuels

### Livrables attendus

- Exploration complète des données
- Visualisations pertinentes (DataViz)
- Identification des problèmes et biais
- Rapport d'exploration avec insights clés
- Recommandations pour les étapes suivantes

---

## 🛠️ Outils Recommandés

### Python
- **Pandas** : manipulation et analyse des données
- **NumPy** : calculs numériques
- **Matplotlib / Seaborn** : visualisations
- **WordCloud** : nuages de mots
- **NLTK / spaCy** : traitement du langage naturel (pour analyses avancées)

### Environnement
- **Jupyter Notebook** ou **JupyterLab** : environnement interactif recommandé

---

## 📝 Structure du Projet

```
Projet ecole/
│
├── data brut/              # Données brutes (ne pas modifier)
│   ├── X_train_update.csv
│   ├── X_test_update.csv
│   └── Y_train.csv
│
├── developpement/          # Fichiers de développement
│   ├── requirements.txt
│   ├── env.example
│   └── .gitignore
│
├── template/               # Template Excel original
│   └── Template - Rapport exploration des données.xlsx
│
├── rendu textuel/          # Rendu et rapports
│   ├── 01_contexte_et_perimetre.md
│   ├── 02_visualisations_et_analyses.md
│   └── Template - Rapport exploration des données.xlsx
│
├── notebooks/              # Notebooks d'exploration
│   └── 01_exploration_data.ipynb
│
├── src/                    # Code source (pour étapes suivantes)
│   └── (à créer)
│
├── reports/                # Rapports et visualisations (à créer)
│   └── (à créer)
│
└── README.md              # Ce fichier
```

---

## 🚀 Installation et Démarrage

### Prérequis

- **Python 3.8+** (recommandé : Python 3.10 ou 3.11)
- **pip** (gestionnaire de paquets Python)
- **Git** (optionnel, pour le versioning)

### Installation des Dépendances

#### 1. Cloner ou télécharger le projet

Si vous utilisez Git :
```bash
git clone <url-du-repo>
cd "Projet ecole"
```

#### 2. Créer un environnement virtuel (recommandé)

**Sur Windows (PowerShell) :**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Sur Windows (CMD) :**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Sur Linux/Mac :**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Installer les dépendances

```bash
pip install -r developpement/requirements.txt
```

**Alternative :** Si vous préférez installer manuellement les packages essentiels :
```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn jupyter openpyxl
```

#### 4. Configurer l'environnement (optionnel)

Copier le fichier d'exemple et le personnaliser :
```bash
# Sur Windows (PowerShell)
Copy-Item developpement\env.example developpement\.env

# Sur Linux/Mac
cp developpement/env.example developpement/.env
```

Puis éditer `developpement/.env` avec vos paramètres si nécessaire.

#### 5. Vérifier l'installation

```bash
python -c "import pandas; import numpy; import matplotlib; print('✅ Installation réussie !')"
```

### Lancer le Projet

#### Option 1 : Jupyter Notebook (recommandé pour l'exploration)

1. **Démarrer Jupyter Notebook :**
```bash
jupyter notebook
```

2. **Ou démarrer JupyterLab :**
```bash
jupyter lab
```

3. **Ouvrir le notebook d'exploration :**
   - Naviguer vers le dossier `notebooks/`
   - Ouvrir `01_exploration_data.ipynb`
   - **Important :** Sélectionner le kernel Python correct (celui de votre environnement virtuel)

#### Option 2 : Créer un kernel Jupyter dédié (recommandé)

Si vous avez des problèmes avec le kernel, créez un kernel dédié :

```bash
# Installer ipykernel si ce n'est pas déjà fait
pip install ipykernel

# Créer un kernel nommé "projet_ecole"
python -m ipykernel install --user --name=projet_ecole --display-name "Python (projet_ecole)"
```

Puis dans Jupyter, sélectionner le kernel "Python (projet_ecole)".

#### Option 3 : Exécuter des scripts Python directement

```bash
# Exemple : exécuter un script d'analyse
python scripts/analyse.py
```

### Commandes Utiles

#### Mettre à jour les dépendances

```bash
pip install --upgrade -r developpement/requirements.txt
```

#### Vérifier les versions installées

```bash
pip list
```

#### Désactiver l'environnement virtuel

```bash
deactivate
```

#### Réinstaller depuis zéro

```bash
# Supprimer l'environnement virtuel
# Sur Windows
Remove-Item -Recurse -Force venv

# Sur Linux/Mac
rm -rf venv

# Recréer et réinstaller
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\Activate.ps1 sur Windows
pip install -r developpement/requirements.txt
```

### Structure des Commandes par Étape

#### Étape 1 : Exploration des données

```bash
# 1. Activer l'environnement virtuel
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# ou
source venv/bin/activate  # Linux/Mac

# 2. Lancer Jupyter
jupyter notebook

# 3. Ouvrir notebooks/01_exploration_data.ipynb
# 4. Exécuter toutes les cellules
```

#### Étape 2+ : Scripts de preprocessing (à venir)

```bash
# Exécuter un script de preprocessing
python src/preprocessing.py

# Exécuter un script de modélisation
python src/train_model.py
```

### Dépannage

#### Problème : `ModuleNotFoundError: No module named 'pandas'`

**Solution :**
1. Vérifier que l'environnement virtuel est activé
2. Réinstaller les dépendances : `pip install -r developpement/requirements.txt`
3. Vérifier que vous utilisez le bon kernel dans Jupyter

#### Problème : Le kernel Jupyter ne trouve pas les packages

**Solution :**
```bash
# Installer ipykernel dans l'environnement virtuel
pip install ipykernel

# Créer un nouveau kernel
python -m ipykernel install --user --name=projet_ecole

# Dans Jupyter, changer le kernel vers "projet_ecole"
```

#### Problème : Erreurs d'encodage (caractères spéciaux)

**Solution :** Le projet utilise UTF-8. Si vous avez des problèmes :
- Windows : Utiliser PowerShell plutôt que CMD
- Vérifier que vos fichiers sont en UTF-8

### Liste Complète des Dépendances

Les dépendances principales sont listées dans `developpement/requirements.txt` :

- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Matplotlib / Seaborn** : Visualisations
- **Plotly** : Visualisations interactives
- **WordCloud** : Nuages de mots
- **Scikit-learn** : Machine Learning
- **Jupyter** : Environnement interactif
- **Openpyxl** : Lecture/écriture Excel
- Et autres dépendances (voir `requirements.txt`)

---

## 🔍 Notions Clés

### Machine Learning
- **Classification multi-classes** : comprendre les enjeux et métriques
- **Déséquilibre de classes** : identifier si certaines catégories sont sous-représentées
- **Validation croisée** : stratégies pour évaluer les modèles

### Natural Language Processing (NLP)
- **Préprocessing de texte** : nettoyage, normalisation, tokenisation
- **Feature extraction** : TF-IDF, word embeddings, etc.
- **Classification textuelle** : approches traditionnelles vs deep learning

### DataViz
- **Visualisations adaptées** : choisir les bons graphiques selon les données
- **Storytelling avec les données** : présenter les insights de manière claire

---

## 📚 Ressources Utiles

### Classification de texte
- Scikit-learn : [Text classification guide](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- NLP avec Python : [Natural Language Processing with Python](https://www.nltk.org/book/)

### DataViz
- Matplotlib : [Gallery](https://matplotlib.org/stable/gallery/index.html)
- Seaborn : [Tutorial](https://seaborn.pydata.org/tutorial.html)

### E-commerce & Classification
- Rechercher des cas d'usage similaires dans l'industrie
- Étudier les défis spécifiques au e-commerce (multilingue, HTML, etc.)

---

## ✅ Critères de Réussite - Étape 1

- [ ] **Compréhension du contexte** : documentation claire du problème métier
- [ ] **Exploration complète** : toutes les dimensions des données analysées
- [ ] **Visualisations pertinentes** : au moins 5-7 visualisations significatives
- [ ] **Insights identifiés** : défis et opportunités documentés
- [ ] **Code propre** : notebook bien structuré et commenté
- [ ] **Documentation** : rapport d'exploration avec conclusions

---

## 🚀 Prochaines Étapes

- **Étape 2** : Préprocessing et nettoyage des données textuelles
- **Étape 3** : Feature engineering et vectorisation
- **Étape 4** : Modélisation (modèles baselines puis optimisation)
- **Étape 5** : Évaluation et métriques
- **Étape 6** : Optimisation et fine-tuning

---

**Bonne exploration ! 🎉**
