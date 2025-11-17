# Rapport d'Exploration, Data Visualisation et Pre-processing des Données

**Projet :** Classification de Produits E-commerce  
**Auteur :** Tristan  
**Date :** Novembre 2025  
**Étapes couvertes :** Étape 1 (Exploration) + Étape 2 (Pre-processing)

---

## 📊 Résumé Exécutif

Ce rapport présente l'exploration complète, les visualisations et le preprocessing effectués sur un dataset de **84 916 produits e-commerce** pour la classification automatique en **27 catégories**.

### Principaux Résultats

- ✅ **Dataset bien structuré** : 84 916 produits d'entraînement, 13 814 produits de test
- ⚠️ **35% de descriptions manquantes** : Gérées via stratégie `designation_only`
- ⚠️ **18% de descriptions avec HTML** : Nettoyé à 99.88% (103 cas marginaux restants)
- ⚠️ **Déséquilibre de classes** : Ratio 13.36 (classe majoritaire : 10 209, minoritaire : 764)
- ✅ **15 features créées** : Longueur, binaires, qualité
- ✅ **Dataset prêt pour modélisation** : Textes nettoyés, normalisés et combinés

### Impact Business

Le preprocessing a transformé les données brutes en un dataset exploitable pour la modélisation, avec une qualité suffisante pour entraîner des modèles de classification performants.

---

## 🎯 Contexte et Périmètre du Projet

### Objectif

Développer un système de classification automatique de produits e-commerce pour prédire le **code de type de produit** (`prdtypecode`) à partir des informations textuelles (désignation et description).

### Enjeux Métier

- **Amélioration de l'expérience utilisateur** : Faciliter la navigation et la recherche
- **Optimisation opérationnelle** : Automatisation du catalogage
- **Recommandations personnalisées** : Suggérer des produits similaires
- **Standardisation** : Réduction des erreurs humaines

### Type de Problème

- **Classification supervisée multi-classes** (27 classes)
- **Données textuelles** : Désignation + Description (NLP)
- **Volume** : 84 916 produits d'entraînement, 13 814 produits de test

---

## 🔍 Exploration des Données

### Structure du Dataset

#### Fichiers Disponibles

1. **`X_train_update.csv`** : 84 916 produits avec 4 colonnes
   - `designation` : Nom/titre du produit
   - `description` : Description détaillée (peut contenir du HTML)
   - `productid` : Identifiant unique
   - `imageid` : Identifiant de l'image

2. **`X_test_update.csv`** : 13 814 produits (même structure, sans labels)

3. **`Y_train.csv`** : 84 916 labels (`prdtypecode`)

#### Qualité des Données

**Valeurs manquantes :**
- `designation` : 0% (toutes présentes)
- `description` : 35.09% (29 800 descriptions manquantes)
- `productid` / `imageid` : 0%

**Cohérence :**
- ✅ Alignement parfait : X_train et y_train ont le même nombre de lignes
- ✅ Pas de doublons détectés
- ✅ Identifiants uniques (pas de data leakage entre train/test)

### Variable Cible : Distribution des Classes

**Nombre de classes :** 27 catégories distinctes

**Top 5 classes les plus représentées :**
1. **2583** (Piscine/Spa) : 10 209 produits (12.0%)
2. **1560** (Mobilier/Maison) : 5 073 produits (6.0%)
3. **1300** (Drones/Modélisme) : 5 045 produits (5.9%)
4. **2060** (Décoration/Éclairage) : 4 993 produits (5.9%)
5. **2522** (Papeterie) : 4 989 produits (5.9%)

**Top 5 classes les moins représentées :**
1. **1180** (Warhammer) : 764 produits (0.9%)
2. **1940** (Alimentation) : 803 produits (0.9%)
3. **1301** (Accessoires Bébé) : 807 produits (1.0%)
4. **2220** (Accessoires Animaux) : 824 produits (1.0%)
5. **60** (Consoles Portables) : 832 produits (1.0%)

**Déséquilibre :**
- Ratio max/min : **13.36** (classe 2583 : 10 209 vs classe 1180 : 764)
- Test chi-carré : Chi² = 36 570.33, p-value < 0.001 → **Déséquilibre statistiquement significatif**

### Analyse des Variables Textuelles

#### Désignations

- **Longueur moyenne** : 70.2 caractères (train) / 69.9 caractères (test)
- **Longueur médiane** : 64 caractères
- **Plage** : 11 à 250 caractères
- **Qualité** : Toutes les désignations sont présentes (0% de valeurs manquantes)

#### Descriptions

- **Longueur moyenne** (non vides) : 524.6 caractères (train) / 525.7 caractères (test)
- **Longueur médiane** : 231 caractères (train) / 221 caractères (test)
- **Plage** : 0 à 12 451 caractères (train) / 0 à 22 299 caractères (test)
- **Valeurs manquantes** : 35.09% (29 800 descriptions)
- **HTML présent** : 18.42% (15 645 descriptions avec HTML)

**Tags HTML les plus fréquents :**
- `<br/>` : 109 065 occurrences
- `<li>` : 54 255 occurrences
- `<p>` : 29 579 occurrences
- `<strong>` : 25 029 occurrences

### Biais Identifiés

1. **Biais de déséquilibre de classes** : Ratio 13.36 nécessitant des techniques de rééquilibrage
2. **Biais de qualité des données** : 35% de descriptions manquantes, variabilité par classe
3. **Biais linguistique** : Textes multilingues (français/anglais)
4. **Biais de longueur** : Classes avec textes très longs (2280) ou très courts (1160, 2403)

---

## 📈 Data Visualisation et Analyses

### Visualisation 1 : Distribution des Classes (prdtypecode)

#### Figure
**Type :** Barplot horizontal des 20 classes les plus représentées + graphique en échelle logarithmique

#### Commentaire Métier

L'analyse révèle un **déséquilibre modéré à fort** dans le catalogue. La classe **2583 (Piscine/Spa)** domine avec 10 209 occurrences (12.0%), tandis que la classe **1180 (Warhammer)** est la moins représentée avec 764 occurrences (0.9%). Le ratio de 13.36 indique que le modèle risque de développer un biais vers les catégories les plus fréquentes.

**Impact business :**
- Les catégories dominantes (2583, 1560, 1300, 2060, 2522) représentent ensemble près de 40% du catalogue
- Les classes minoritaires (1180, 50, 1281, 1302) nécessiteront une attention particulière
- Le déséquilibre est typique d'un catalogue e-commerce réel

**Recommandations :**
- Mettre en place des techniques de rééquilibrage (SMOTE, class weights)
- Utiliser des métriques adaptées au déséquilibre (F1-score par classe)
- Considérer un seuil de confiance différent selon la classe

#### Validation Statistique

**Test de chi-carré pour le déséquilibre des classes :**
- **Hypothèse nulle (H0) :** La distribution des classes est uniforme
- **Résultat :** Chi² = 36 570.33, p-value < 0.001
- **Conclusion :** Le déséquilibre est statistiquement significatif. On rejette H0 avec un niveau de confiance > 99.9%

**Métriques descriptives :**
- **Ratio max/min :** 13.36 (classe 2583 : 10 209 occurrences / classe 1180 : 764 occurrences)
- **Classes avec < 10 exemples :** 0 (toutes les classes ont suffisamment d'exemples)
- **Distribution :** Long tail typique d'un catalogue e-commerce

---

### Visualisation 2 : Distribution de la Longueur des Textes par Classe

#### Figure
**Type :** Boxplots comparatifs de la longueur (en caractères) des désignations et descriptions pour les 10 classes principales

#### Commentaire Métier

L'analyse révèle des **patterns distincts selon les catégories**. Les désignations de la classe **2280** sont exceptionnellement longues (médiane ~170 caractères), tandis que les classes **1160** et **2403** ont des descriptions très courtes (médiane ~50 caractères). Cette variabilité suggère que certaines catégories nécessitent plus de détails pour être décrites.

**Impact business :**
- Les produits de la classe 2280 nécessitent des descriptions détaillées
- Les classes avec des descriptions courtes peuvent être plus difficiles à classifier
- Le modèle pourrait apprendre à utiliser la longueur du texte comme proxy de la catégorie (biais non souhaitable)

**Recommandations :**
- Normaliser les longueurs de texte (padding/truncation)
- Créer des features explicites de longueur pour permettre au modèle de les utiliser consciemment
- Adapter le preprocessing selon la classe

#### Validation Statistique

**Test de Kruskal-Wallis pour les longueurs de descriptions (top 10 classes) :**
- **Hypothèse nulle (H0) :** Les longueurs de descriptions sont identiques entre les classes
- **Résultat :** H = 26 348.68, p-value < 0.001
- **Conclusion :** Les longueurs sont significativement différentes entre les classes. On rejette H0 avec un niveau de confiance > 99.9%

**Statistiques descriptives par classe (top 10) :**
- **Classe 2280 :** Désignations très longues (médiane ~170 caractères, IQR 100-250)
- **Classes 1160, 2403 :** Descriptions très courtes (médiane ~50 caractères)
- **Classes 1280, 1300, 1560, 1920, 2060, 2583 :** Descriptions moyennes à longues (médiane 500-900 caractères)
- **Variabilité :** Distribution très asymétrique avec de nombreux outliers (descriptions jusqu'à 12 451 caractères)

---

### Visualisation 3 : Analyse des Valeurs Manquantes

#### Figure
**Type :** Heatmap montrant le pourcentage de valeurs manquantes par colonne et par classe (top 15 classes)

#### Commentaire Métier

L'analyse révèle un **problème majeur de qualité des données** : **35.09% des descriptions sont absentes**. Plus préoccupant encore, cette proportion varie considérablement selon la catégorie. Les classes **2403 (97.4%)**, **2280 (93.3%)**, **1160 (91.2%)** et **10 (89.2%)** ont plus de 90% de descriptions manquantes, tandis que les classes **2582 (2.4%)**, **1560 (3.5%)** et **1920 (4.8%)** sont bien documentées.

**Impact business :**
- Les catégories avec peu de descriptions devront s'appuyer principalement sur les désignations
- Cette inégalité crée un biais de qualité : certaines catégories sont mieux documentées
- Le modèle risque de sous-performer sur les catégories mal documentées

**Recommandations :**
- **Stratégie hybride :** Utiliser la designation seule pour les produits sans description
- **Feature engineering :** Créer une feature binaire `has_description`
- **Évaluation adaptée :** Évaluer séparément les performances sur les produits avec/sans description
- **Amélioration future :** Prioriser l'enrichissement des descriptions pour les catégories mal documentées

#### Validation Statistique

**Analyse des valeurs manquantes :**
- **X_train :** 29 800 descriptions manquantes sur 84 916 (35.09%)
- **X_test :** 4 886 descriptions manquantes sur 13 812 (35.38%)
- **Cohérence train/test :** Les proportions sont similaires, ce qui est rassurant pour la généralisation

**Variabilité par classe (exemples) :**
- **Classe 2403 :** 97.4% de descriptions manquantes
- **Classe 2280 :** 93.3% de descriptions manquantes
- **Classe 1160 :** 91.2% de descriptions manquantes
- **Classe 2582 :** 2.4% de descriptions manquantes (meilleure documentation)
- **Classe 1560 :** 3.5% de descriptions manquantes

**Interprétation :** Les valeurs manquantes ne sont pas aléatoires mais dépendent de la classe. Ce pattern systématique nécessite une stratégie de traitement adaptée.

---

### Visualisation 4 : Nuages de Mots par Catégorie

#### Figure
**Type :** Word clouds pour les 5 classes les plus représentées (2583, 1560, 1300, 2060, 2522), basés sur les désignations et descriptions nettoyées

#### Commentaire Métier

Les nuages de mots révèlent des **vocabulaires distincts et caractéristiques** pour chaque catégorie, ce qui est un signal positif pour la classification. La classe **2583** se distingue par des termes liés aux piscines ("piscine", "eau", "filtration", "hors sol"), la classe **1300** par le vocabulaire des drones ("dji", "mavic", "drone", "rc", "quadcopter"), et la classe **2522** par les produits de papeterie ("papier", "note", "bloc", "carnet").

**Impact business :**
- La séparabilité lexicale entre les classes est bonne, ce qui suggère que les modèles de classification textuelle devraient bien performer
- Les mots-clés identifiés peuvent être utilisés pour améliorer le feature engineering
- Certaines catégories ont un vocabulaire très technique (1300 - drones), tandis que d'autres sont plus génériques (2060 - décoration)

**Recommandations :**
- **Feature engineering :** Créer des features binaires pour la présence de mots-clés discriminants
- **Preprocessing adaptatif :** Conserver les termes techniques spécifiques qui sont très discriminants
- **Analyse de similarité :** Mesurer la similarité lexicale entre classes pour identifier les catégories potentiellement confondues

#### Validation Statistique

**Analyse des mots les plus fréquents par classe :**

**Classe 2583 (Piscine/Spa, n=10 209) :**
- Mots caractéristiques : "piscine", "eau", "filtration", "hors sol", "cm", "volume", "pompe"
- Vocabulaire technique lié aux dimensions et équipements de piscine

**Classe 1560 (Mobilier/Maison, n=5 073) :**
- Mots caractéristiques : "bois", "qualité", "rangement", "salle", "bain", "cuisine", "métal"
- Vocabulaire générique lié au mobilier et à l'aménagement

**Classe 1300 (Drones, n=5 045) :**
- Mots caractéristiques : "dji", "mavic", "drone", "rc", "quadcopter", "batterie", "caméra", "fpv"
- Vocabulaire très technique et spécifique au domaine des drones

**Classe 2060 (Décoration/Cadeaux, n=4 993) :**
- Mots caractéristiques : "noël", "diamant", "résine", "lumière", "décoration", "qualité"
- Vocabulaire lié à la décoration et aux cadeaux

**Classe 2522 (Papeterie, n=4 989) :**
- Mots caractéristiques : "papier", "note", "bloc", "carnet", "format a5", "cahier"
- Vocabulaire spécifique aux produits de papeterie

**Interprétation :** Les word clouds confirment que chaque classe possède un vocabulaire distinct, ce qui est favorable pour la classification. Les termes techniques (ex: "mavic", "quadcopter") sont particulièrement discriminants.

---

### Visualisation 5 : Présence de HTML et Qualité des Descriptions

#### Figure
**Type :** Graphique en barres empilées montrant la proportion de descriptions avec HTML, sans HTML, et vides, par classe (top 10)

#### Commentaire Métier

L'analyse révèle une **hétérogénéité importante de la qualité des données** selon les catégories. Les classes **1560 (96%)**, **1920 (94%)**, **2060 (92%)** et **2583 (92%)** ont une très forte proportion de descriptions avec HTML, tandis que les classes **2403 (97%)**, **2280 (93%)** et **1160 (92%)** ont principalement des descriptions vides. Cette observation révèle deux problèmes distincts : certaines catégories nécessitent un nettoyage HTML intensif, tandis que d'autres manquent simplement de contenu.

**Impact business :**
- Les catégories avec beaucoup de HTML nécessitent un preprocessing robuste mais disposent d'un contenu riche une fois nettoyé
- Les catégories avec beaucoup de descriptions vides auront des performances limitées car elles ne peuvent s'appuyer que sur les désignations
- Cette inégalité de qualité crée un défi supplémentaire pour le modèle

**Recommandations :**
- **Nettoyage HTML prioritaire :** Mettre en place un pipeline de nettoyage robuste utilisant BeautifulSoup
- **Gestion des entités HTML :** Décoder les entités HTML pour restaurer les caractères originaux
- **Stratégie hybride :** Pour les catégories avec beaucoup de HTML, nettoyer et utiliser le contenu. Pour les catégories avec beaucoup de vides, s'appuyer sur les désignations
- **Feature de qualité :** Créer des features indiquant la présence de HTML et la présence de description

#### Validation Statistique

**Test de chi-carré pour l'indépendance HTML/classe :**
- **Hypothèse nulle (H0) :** La présence de HTML est indépendante de la classe
- **Résultat :** Chi² = 15 181.95, p-value < 0.001
- **Conclusion :** La présence de HTML est dépendante de la classe. On rejette H0 avec un niveau de confiance > 99.9%

**Proportions par classe (top 10) :**

**Classes avec beaucoup de HTML :**
- **Classe 1560 :** 96% avec HTML, 4% vides
- **Classe 1920 :** 94% avec HTML, 6% vides
- **Classe 2060 :** 92% avec HTML, 8% vides
- **Classe 2583 :** 92% avec HTML, 8% vides

**Classes avec beaucoup de descriptions vides :**
- **Classe 2403 :** 97% vides, 3% avec HTML
- **Classe 2280 :** 93% vides, 8% avec HTML
- **Classe 1160 :** 92% vides, 8% avec HTML

**Tags HTML les plus fréquents :**
- `<br/>` : 109 065 occurrences
- `<li>` : 54 255 occurrences
- `</li>` : 53 199 occurrences
- `<p>` : 29 579 occurrences
- `<strong>` : 25 029 occurrences

**Interprétation :** Le test confirme que la présence de HTML n'est pas aléatoire mais dépend fortement de la classe. Ce pattern systématique suggère que certaines catégories sont mieux formatées (avec HTML) que d'autres.

---

## 🔧 Pre-processing et Feature Engineering

### Vue d'Ensemble

Le preprocessing a transformé les données brutes en un dataset prêt pour la modélisation, en résolvant tous les problèmes identifiés lors de l'exploration.

### Phase 1 : Nettoyage HTML

#### Méthodologie

- **Outil utilisé :** BeautifulSoup pour parser le HTML
- **Décodage des entités HTML :** `&eacute;` → `é`, `&#39;` → `'`, etc.
- **Conservation de la structure :** Paragraphes → sauts de ligne
- **Fallback :** Regex en cas d'erreur de parsing

#### Résultats

**Avant preprocessing :**
- Descriptions avec HTML : 15 645 (18.42%)

**Après preprocessing :**
- Descriptions avec HTML restant : 103 (0.12%)
- **Taux de nettoyage : 99.88%**

**Investigation du HTML restant :**
- **Faux positifs (majorité) :** Textes stylisés avec `<< ... >` (marques de marque)
- **Tags malformés :** Balises avec espaces (`< br / >`) ou non fermées (`</li`)
- **Caractères isolés :** Caractères `>` isolés (103 occurrences)
- **Impact :** Négligeable (< 0.2% du dataset)

**Exemples de transformation :**
```
AVANT : <p>Produit de <strong>qualité</strong> avec <br/>caractéristiques...</p>
APRÈS : Produit de qualité avec caractéristiques...
```

### Phase 2 : Gestion des Valeurs Manquantes

#### Stratégie Choisie : `designation_only`

Pour les produits sans description, utiliser uniquement la designation. Les descriptions vides sont remplacées par une chaîne vide, et une feature `has_description` indique au modèle la présence d'une description.

#### Résultats

**Avant preprocessing :**
- Descriptions manquantes : 29 800 (35.09%)

**Après preprocessing :**
- Descriptions vides : 29 888 (35.20%) - gérées via `has_description`
- **Impact :** Toutes les valeurs manquantes sont gérées de manière appropriée

**Justification :**
- Les désignations sont toujours présentes (0% de valeurs manquantes)
- La feature `has_description` permet au modèle de distinguer les cas
- Évite l'introduction de biais par imputation

### Phase 3 : Normalisation des Textes

#### Transformations Appliquées

- **Minuscules :** Conversion en minuscules (accents conservés)
- **Espaces multiples :** Suppression des espaces multiples
- **Accents :** Conservés (important pour le français)

#### Résultats

- Textes normalisés de manière cohérente
- Pas de perte d'information linguistique (accents conservés)

**Exemple :**
```
AVANT : "Produit  DE   Qualité  Élevée"
APRÈS : "produit de qualité élevée"
```

### Phase 4 : Feature Engineering

#### Features Créées (15 au total)

**Features de Longueur (6 features) :**
1. `designation_length` : Longueur de la designation en caractères
2. `description_length` : Longueur de la description en caractères
3. `total_text_length` : Somme des deux longueurs
4. `designation_word_count` : Nombre de mots dans la designation
5. `description_word_count` : Nombre de mots dans la description
6. `total_word_count` : Nombre total de mots

**Features Binaires (3 features) :**
7. `has_description` : Présence d'une description (0/1)
8. `has_html` : Présence de HTML avant nettoyage (0/1)
9. `is_description_empty` : Description vide (0/1)

**Features de Qualité (3 features) :**
10. `text_completeness` : Ratio designation_length / total_text_length
11. `description_quality_score` : Score basé sur présence + longueur (0-1)
12. `word_density` : Ratio mots/caractères (densité d'information)

**Features Calculées (3 features) :**
13. `designation_avg_word_length` : Longueur moyenne des mots (designation)
14. `description_avg_word_length` : Longueur moyenne des mots (description)
15. `text_combined` : Texte combiné (designation + description)

### Phase 5 : Préparation Finale

#### Structure du Dataset Final

**Colonnes (17 au total) :**
- `productid`, `imageid` : Identifiants
- `designation_clean`, `description_clean`, `text_combined` : Textes nettoyés
- 12 features supplémentaires

**Dimensions :**
- `X_train_final` : 84 916 lignes × 17 colonnes
- `X_test_final` : 13 812 lignes × 17 colonnes

#### Sauvegarde

Les datasets ont été sauvegardés dans `data/processed/` :
- `X_train_clean.csv` : Dataset d'entraînement nettoyé
- `X_test_clean.csv` : Dataset de test nettoyé
- `y_train.csv` : Labels d'entraînement

---

## 📊 Statistiques Avant/Après Preprocessing

### Comparaison Globale

| Métrique | Avant | Après | Amélioration |
|---------|-------|-------|--------------|
| **Descriptions avec HTML** | 18.42% | 0.12% | ✅ 99.88% nettoyé |
| **Descriptions manquantes** | 35.09% | 0% (gérées) | ✅ Gérées via features |
| **Longueur moyenne description** | 524.6 caractères | Variable | ✅ Normalisée |
| **Features disponibles** | 4 colonnes | 17 colonnes | ✅ +13 features |

### Distribution des Longueurs

**Avant preprocessing :**
- Désignations : moyenne = 70.2, médiane = 64 caractères
- Descriptions : moyenne = 524.6, médiane = 231 caractères

**Après preprocessing :**
- Textes combinés : moyenne = 559.2, médiane = 292 caractères
- Distribution plus homogène

### Qualité des Données

**Avant :**
- HTML présent dans 18.42% des descriptions
- 35.09% de descriptions manquantes
- Textes non normalisés

**Après :**
- ✅ HTML complètement nettoyé (99.88%)
- ✅ Valeurs manquantes gérées
- ✅ Textes normalisés et prêts pour NLP

---

## ✅ Validation et Vérifications

### Tests de Validation Effectués

1. ✅ **Pas de NaN dans text_combined** : Tous les textes combinés sont valides
2. ✅ **Pas de NaN dans designation_clean** : Toutes les désignations sont présentes
3. ✅ **Pas de textes vides** : Tous les produits ont au moins une designation
4. ✅ **Cohérence train/test** : Même structure et colonnes
5. ⚠️ **HTML restant** : 103 descriptions (0.12%) - Acceptable (faux positifs majoritaires)

### Cohérence des Données

- **Train/Test :** Distributions similaires, pas de data leakage
- **Alignement :** X_train et y_train parfaitement alignés (84 916 lignes)
- **Complétude :** Tous les produits ont au moins une designation

---

## 🎯 Conclusion et Recommandations

### Résumé des Accomplissements

Le preprocessing a été effectué avec succès :

- ✅ **HTML nettoyé** : 99.88% des descriptions HTML nettoyées (103 cas marginaux restants)
- ✅ **Valeurs manquantes gérées** : Stratégie `designation_only` appliquée avec feature `has_description`
- ✅ **Textes normalisés** : Conversion en minuscules, suppression des espaces multiples
- ✅ **Features créées** : 15 features supplémentaires (longueur, binaires, qualité)
- ✅ **Dataset prêt** : Datasets nettoyés sauvegardés et prêts pour la modélisation

### Recommandations pour la Modélisation

#### 1. Vectorisation des Textes

- **TF-IDF** : Approche classique, rapide et efficace
- **Word2Vec / FastText** : Embeddings de mots, capture les similarités sémantiques
- **BERT / Transformers** : Approche state-of-the-art, meilleures performances mais plus coûteuse

#### 2. Gestion du Déséquilibre

- **Class weights** : Pondérer les classes minoritaires dans la fonction de coût
- **SMOTE / ADASYN** : Oversampling des classes minoritaires
- **Focal Loss** : Fonction de perte adaptée au déséquilibre

#### 3. Sélection de Modèles

**Modèles baselines :**
- Naive Bayes : Rapide, bon pour les textes courts
- SVM : Performant avec TF-IDF
- Logistic Regression : Interprétable, bon point de départ

**Modèles avancés :**
- Random Forest : Gère bien les features numériques supplémentaires
- XGBoost : Performant, gère bien le déséquilibre
- Neural Networks : LSTM, BERT fine-tuned

#### 4. Métriques d'Évaluation

- **F1-score par classe** : Évaluer les performances sur chaque catégorie
- **Matrice de confusion** : Identifier les classes confondues
- **Validation croisée** : Stratified K-Fold pour gérer le déséquilibre

### Prochaines Étapes

1. **Étape 3 : Modélisation**
   - Vectorisation des textes (TF-IDF, Word2Vec, BERT)
   - Entraînement de modèles baselines
   - Sélection du meilleur modèle

2. **Étape 4 : Évaluation**
   - Métriques adaptées au déséquilibre
   - Analyse des erreurs de classification
   - Optimisation des hyperparamètres

3. **Étape 5 : Optimisation**
   - Fine-tuning du modèle sélectionné
   - Techniques de rééquilibrage
   - Amélioration des performances

---

## 📁 Fichiers Générés

### Datasets Nettoyés

- `data/processed/X_train_clean.csv` : Dataset d'entraînement nettoyé (84 916 lignes × 17 colonnes)
- `data/processed/X_test_clean.csv` : Dataset de test nettoyé (13 812 lignes × 17 colonnes)
- `data/processed/y_train.csv` : Labels d'entraînement (84 916 lignes)

### Notebooks

- `notebooks/01_exploration_data.ipynb` : Exploration complète des données
- `notebooks/02_preprocessing.ipynb` : Preprocessing et feature engineering

### Rapports

- `rendu textuel/01_contexte_et_perimetre.md` : Contexte et périmètre
- `rendu textuel/02_visualisations_et_analyses.md` : Visualisations détaillées
- `rendu textuel/04_rapport_preprocessing.md` : Rapport de preprocessing
- `rendu textuel/RAPPORT_COMPLET_ETAPE_1_ET_2.md` : Ce rapport complet

---

**Fin du Rapport**

*Auteur : Tristan*  
*Date : Novembre 2025*  
*Version : 1.0*

