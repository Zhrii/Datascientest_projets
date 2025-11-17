# Contexte et Périmètre du Projet

## Question

Votre première tâche consistera à définir le contexte et le périmètre du projet : j'attends que vous preniez vraiment le temps de bien comprendre le projet et de vous renseigner au mieux sur les notions que celui-ci va introduire.

## Réponse

### Contexte du Projet

Ce projet s'inscrit dans le domaine de l'e-commerce et vise à automatiser la classification de produits à partir de leurs descriptions textuelles. L'objectif principal est de prédire le code de type de produit (`prdtypecode`) en analysant les informations textuelles disponibles : la désignation (nom/titre du produit) et la description détaillée.

**Enjeux métier :**
- **Amélioration de l'expérience utilisateur** : Une classification précise facilite la navigation et la recherche de produits sur une plateforme e-commerce
- **Optimisation opérationnelle** : Automatisation du catalogage, réduisant le temps et les coûts de gestion manuelle
- **Recommandations personnalisées** : Classification correcte permettant de suggérer des produits similaires de manière plus pertinente
- **Standardisation** : Réduction des erreurs humaines et harmonisation de la catégorisation

### Périmètre du Projet

**Type de problème :** Classification supervisée multi-classes
- **Données d'entrée** : Textes (désignation + description) avec présence possible de HTML
- **Variable cible** : `prdtypecode` (code numérique représentant la catégorie de produit)
- **Volume** : 84 916 produits d'entraînement, 13 814 produits de test

**Défis techniques identifiés :**
1. **Traitement du langage naturel (NLP)** : Analyse de textes multilingues (français, anglais)
2. **Nettoyage de données** : Présence de HTML, entités HTML, caractères spéciaux dans les descriptions
3. **Gestion des valeurs manquantes** : Certaines descriptions peuvent être vides
4. **Déséquilibre potentiel des classes** : Certaines catégories peuvent être sous-représentées
5. **Feature engineering** : Extraction de caractéristiques pertinentes à partir de textes

**Notions clés à maîtriser :**
- **Machine Learning** : Classification multi-classes, métriques d'évaluation (précision, rappel, F1-score), gestion du déséquilibre de classes
- **Natural Language Processing** : Préprocessing de texte (tokenisation, normalisation), extraction de features (TF-IDF, word embeddings), modèles de classification textuelle
- **Data Science** : Exploration de données, visualisation, validation croisée, optimisation de modèles

**Démarche méthodologique :**
1. Exploration exhaustive des données et visualisations
2. Préprocessing et nettoyage des données textuelles
3. Feature engineering et vectorisation
4. Modélisation avec sélection de modèles adaptés
5. Évaluation et optimisation des performances
6. Déploiement (si applicable)

Ce projet nécessite une compréhension approfondie des techniques de NLP et de classification, ainsi qu'une approche rigoureuse pour gérer les spécificités des données e-commerce (multilingue, HTML, qualité variable des descriptions).

---

## Question

Il faudra ensuite prendre en main et découvrir votre jeu de données et faire une analyse presque exhaustive de celui-ci afin de mettre en lumière la structure, les difficultés et éventuels biais du dataset.

## Réponse

### Structure du Dataset

**Organisation des données :**
- **X_train_update.csv** : 84 916 produits avec 4 colonnes (designation, description, productid, imageid)
- **X_test_update.csv** : 13 814 produits avec la même structure (sans labels)
- **Y_train.csv** : 84 916 labels correspondants (prdtypecode)
- **Alignement** : X_train et Y_train sont parfaitement alignés (même nombre de lignes)

**Types de variables :**
- **Textuelles** : `designation` (nom court du produit) et `description` (texte détaillé, peut contenir du HTML)
- **Numériques** : `productid` et `imageid` (identifiants uniques)
- **Cible** : `prdtypecode` (classification multi-classes)

### Difficultés Identifiées

1. **Qualité des données textuelles :**
   - Présence de HTML dans les descriptions (tags `<br>`, `<ul>`, `<li>`, `<strong>`, etc.)
   - Entités HTML non décodées (`&eacute;`, `&#39;`, etc.)
   - Valeurs manquantes dans les descriptions (certains produits n'ont pas de description)
   - Textes multilingues (français et anglais principalement)

2. **Complexité du preprocessing :**
   - Nettoyage du HTML nécessaire avant analyse
   - Normalisation des textes (majuscules, accents, ponctuation)
   - Gestion des valeurs manquantes (imputation ou stratégie spécifique)

3. **Défi de classification :**
   - Nombre de classes à déterminer (analyse de la distribution de `prdtypecode`)
   - Déséquilibre potentiel des classes (certaines catégories plus représentées que d'autres)
   - Classes rares pouvant être difficiles à prédire

### Biais Potentiels

1. **Biais linguistique :**
   - Certaines catégories peuvent être majoritairement en français, d'autres en anglais
   - Risque de confusion du modèle entre langue et catégorie

2. **Biais de longueur :**
   - Les descriptions peuvent avoir des longueurs très variables selon les catégories
   - Certaines catégories peuvent avoir des textes plus détaillés que d'autres

3. **Biais de représentation :**
   - Déséquilibre des classes (classes majoritaires vs minoritaires)
   - Classes sous-représentées risquant d'être mal prédites
   - Nécessité potentielle de techniques de rééquilibrage (oversampling, undersampling, SMOTE)

4. **Biais temporel (à vérifier) :**
   - Les `productid` pourraient suivre un ordre chronologique
   - Risque de data leakage si les identifiants révèlent des informations sur la classe

5. **Biais de qualité :**
   - Incohérence dans la qualité des descriptions (certaines très détaillées, d'autres vides)
   - Présence de HTML variable selon les catégories

### Analyse à Effectuer

**Exploration structurelle :**
- Dimensions, types de données, valeurs manquantes par colonne
- Unicité des identifiants, détection de doublons
- Alignement et cohérence entre train et test

**Analyse de la variable cible :**
- Distribution des classes, identification du déséquilibre
- Classes les plus/moins représentées
- Ratio entre classe majoritaire et minoritaire

**Analyse textuelle :**
- Statistiques de longueur (moyenne, médiane, distribution)
- Fréquence des mots, vocabulaire par classe
- Présence et types de HTML
- Détection des langues présentes

**Identification des patterns :**
- Corrélations entre longueur de texte et classe
- Patterns spécifiques par catégorie
- Comparaison train/test pour détecter les différences

Cette analyse exhaustive permettra de définir une stratégie de preprocessing adaptée et d'anticiper les défis de modélisation.

---

## Question

Vous pourrez utiliser ce template : Template - Rapport exploration des données

J'attendrai également au moins 5 représentations graphiques construites à partir de votre jeu de données, visuelles et surtout pertinentes. Pour chacune d'elle j'attendrai :

Un commentaire précis, qui analyse la figure et apporte un avis "métier".

Une validation du constat par des manipulations de données, ou un test statistique.

## Réponse

### Utilisation du Template Excel

Le template "Template - Rapport exploration des données.xlsx" servira de structure de base pour organiser l'analyse. Il permettra de :
- Structurer les résultats de l'exploration de manière standardisée
- Documenter les découvertes de façon claire et professionnelle
- Faciliter la présentation des insights aux parties prenantes

**Approche proposée :**
- Compléter chaque section du template avec les résultats de l'analyse
- Utiliser le template comme guide pour s'assurer de couvrir tous les aspects importants
- Adapter les sections selon les spécificités du dataset

### Proposition de 5 Visualisations Pertinentes

#### 1. Distribution des Classes (prdtypecode)

**Visualisation :** Barplot horizontal des 20 classes les plus représentées + graphique log-scale pour visualiser le déséquilibre

**Commentaire métier :**
- Identifier les catégories dominantes du catalogue
- Quantifier le déséquilibre entre classes (ratio majoritaire/minoritaire)
- Évaluer l'impact sur la stratégie de modélisation (besoin de rééquilibrage ?)
- Identifier les catégories rares qui nécessiteront une attention particulière

**Validation statistique :**
- Calcul du ratio entre la classe la plus fréquente et la moins fréquente
- Test de chi-carré pour vérifier si la distribution est significativement déséquilibrée
- Calcul de l'entropie de Shannon pour mesurer la diversité des classes
- Identification des classes avec moins de 10 exemples (seuil critique)

#### 2. Distribution de la Longueur des Textes par Classe

**Visualisation :** Boxplots comparatifs de la longueur (en caractères/mots) des désignations et descriptions pour les 10 classes principales

**Commentaire métier :**
- Détecter si certaines catégories ont des descriptions systématiquement plus longues/courtes
- Identifier un biais potentiel où le modèle pourrait utiliser la longueur comme proxy de la catégorie
- Comprendre la variabilité de la qualité des descriptions selon les catégories
- Évaluer si le preprocessing devra être adapté par classe

**Validation statistique :**
- Test de Kruskal-Wallis pour comparer les longueurs entre classes
- Calcul des moyennes et écarts-types par classe
- Test de corrélation entre longueur moyenne et fréquence de la classe
- Identification des outliers (textes anormalement longs/courts)

#### 3. Analyse des Valeurs Manquantes

**Visualisation :** Heatmap montrant le pourcentage de valeurs manquantes par colonne et par classe (top 15 classes)

**Commentaire métier :**
- Évaluer l'impact des données manquantes sur la qualité du dataset
- Identifier si certaines catégories sont plus affectées par les valeurs manquantes
- Déterminer si les valeurs manquantes sont aléatoires ou systématiques
- Guider la stratégie d'imputation (suppression, imputation, traitement spécifique)

**Validation statistique :**
- Test du chi-carré pour vérifier l'indépendance entre présence de valeurs manquantes et classe
- Calcul du pourcentage de valeurs manquantes par colonne et par classe
- Analyse des patterns de co-occurrence (si description manque, designation aussi ?)
- Test de corrélation entre taux de valeurs manquantes et performance attendue

#### 4. Nuage de Mots par Catégorie (Top 5 Classes)

**Visualisation :** Word clouds pour les 5 classes les plus représentées, basés sur les désignations et descriptions

**Commentaire métier :**
- Identifier les mots-clés caractéristiques de chaque catégorie
- Détecter les opportunités de feature engineering (mots discriminants)
- Comprendre le vocabulaire spécifique à chaque domaine de produits
- Évaluer la séparabilité des classes (vocabulaire distinct ou chevauchement ?)

**Validation statistique :**
- Calcul de la fréquence des mots par classe
- Calcul du TF-IDF pour identifier les mots les plus discriminants
- Test de spécificité (quels mots sont significativement plus fréquents dans une classe ?)
- Mesure de la similarité lexicale entre classes (cosine similarity sur les vecteurs de mots)

#### 5. Présence de HTML et Qualité des Descriptions

**Visualisation :** Graphique en barres empilées montrant la proportion de descriptions avec HTML, sans HTML, et vides, par classe (top 10)

**Commentaire métier :**
- Quantifier l'ampleur du problème HTML dans le dataset
- Identifier si certaines catégories sont plus affectées (biais de qualité)
- Évaluer l'effort de preprocessing nécessaire
- Comprendre l'impact sur la qualité des données disponibles pour chaque catégorie

**Validation statistique :**
- Test du chi-carré pour vérifier si la présence de HTML est indépendante de la classe
- Calcul des proportions de HTML par classe
- Analyse de corrélation entre présence de HTML et longueur des descriptions
- Identification des tags HTML les plus fréquents (analyse de patterns)

### Structure Proposée pour Chaque Visualisation

Pour chaque graphique, le document contiendra :

1. **Figure** : Graphique clair, titré, avec légendes et axes annotés
2. **Commentaire métier** (3-5 phrases) :
   - Description de ce que montre la visualisation
   - Interprétation des patterns observés
   - Implications pour le projet (preprocessing, modélisation, métriques)
   - Avis sur l'impact business/métier
3. **Validation statistique** :
   - Manipulations de données (calculs, agrégations, filtres)
   - Tests statistiques appropriés (chi-carré, ANOVA, corrélations, etc.)
   - Résultats numériques (p-values, statistiques, métriques)
   - Interprétation des résultats

### Outils et Méthodes

**Visualisations :**
- Python : Matplotlib, Seaborn, Plotly pour des graphiques interactifs
- WordCloud pour les nuages de mots
- Heatmaps pour les corrélations et patterns

**Tests statistiques :**
- Tests d'hypothèses : chi-carré, ANOVA, Kruskal-Wallis, tests de corrélation
- Métriques descriptives : moyennes, médianes, écarts-types, quartiles
- Analyses de similarité : cosine similarity, distance de Jaccard

Cette approche garantit que chaque visualisation apporte une valeur ajoutée à la compréhension du dataset et que les constats sont validés scientifiquement.