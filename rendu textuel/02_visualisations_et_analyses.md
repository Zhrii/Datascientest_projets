# Visualisations et Analyses - Étape 1

## Introduction

Ce document présente les 5 visualisations clés de l'exploration des données, accompagnées de commentaires métier et de validations statistiques basées sur les résultats réels de l'analyse.

--------------------------------------------------------------------------------------------------------------------------------------------

## Visualisation 1 : Distribution des Classes (prdtypecode)

### Figure
**Type :** Barplot horizontal des 20 classes les plus représentées + graphique en échelle logarithmique pour toutes les classes

### Commentaire Métier

L'analyse de la distribution des classes révèle un **déséquilibre modéré à fort** dans le catalogue de produits. La classe **2583 (Piscine/Spa)** domine avec 10 209 occurrences (12.0% du dataset), tandis que la classe **1180 (Warhammer)** est la moins représentée avec seulement 764 occurrences (0.9%). Le ratio de 13.36 entre la classe majoritaire et minoritaire indique que le modèle risque de développer un biais vers les catégories les plus fréquentes.

**Impact business :**
- Les catégories dominantes (2583, 1560, 1300, 2060, 2522) représentent ensemble près de 40% du catalogue, ce qui suggère une concentration sur certains types de produits
- Les classes minoritaires (1180, 50, 1281, 1302) nécessiteront une attention particulière lors de la modélisation pour éviter une sous-performance
- Le déséquilibre observé est typique d'un catalogue e-commerce réel, où certaines catégories sont naturellement plus populaires

**Recommandations :**
- Mettre en place des techniques de rééquilibrage (SMOTE, class weights) pour améliorer les performances sur les classes rares
- Utiliser des métriques adaptées au déséquilibre (F1-score par classe, matrice de confusion) plutôt que la simple précision globale
- Considérer un seuil de confiance différent selon la classe pour optimiser le taux de classification correcte

### Validation Statistique

**Test de chi-carré pour le déséquilibre des classes :**
- **Hypothèse nulle (H0) :** La distribution des classes est uniforme
- **Résultat :** Chi² = 36 570.33, p-value < 0.001
- **Conclusion :** Le déséquilibre est statistiquement significatif. On rejette H0 avec un niveau de confiance > 99.9%

**Métriques descriptives :**
- **Ratio max/min :** 13.36 (classe 2583 : 10 209 occurrences / classe 1180 : 764 occurrences)
- **Classes avec < 10 exemples :** 0 (toutes les classes ont suffisamment d'exemples pour l'apprentissage)
- **Distribution :** Long tail typique d'un catalogue e-commerce (quelques classes dominantes, nombreuses classes avec peu d'exemples)

**Interprétation :** Le déséquilibre est confirmé statistiquement. Bien qu'aucune classe ne soit critique (< 10 exemples), le ratio de 13.36 nécessite des techniques de rééquilibrage pour garantir des performances équitables sur toutes les catégories.

--------------------------------------------------------------------------------------------------------------------------------------------

## Visualisation 2 : Distribution de la Longueur des Textes par Classe

### Figure
**Type :** Boxplots comparatifs de la longueur (en caractères) des désignations et descriptions pour les 10 classes principales

### Commentaire Métier

L'analyse des longueurs de texte révèle des **patterns distincts selon les catégories de produits**, ce qui peut constituer un biais potentiel pour le modèle. Les désignations de la classe **2280** sont exceptionnellement longues (médiane ~170 caractères), tandis que les classes **1160** et **2403** ont des descriptions très courtes (médiane ~50 caractères). Cette variabilité suggère que certaines catégories nécessitent plus de détails pour être décrites, tandis que d'autres peuvent être identifiées par des termes courts.

**Impact business :**
- Les produits de la classe 2280 (probablement des produits techniques ou complexes) nécessitent des descriptions détaillées, ce qui peut être un indicateur de la complexité du produit
- Les classes avec des descriptions courtes (1160, 2403) peuvent être plus difficiles à classifier car elles contiennent moins d'information textuelle
- Le modèle pourrait apprendre à utiliser la longueur du texte comme proxy de la catégorie, ce qui serait un biais non souhaitable

**Recommandations :**
- Normaliser les longueurs de texte (padding/truncation) pour éviter que le modèle utilise la longueur comme feature discriminante
- Créer des features explicites de longueur (designation_length, description_length) pour permettre au modèle de les utiliser consciemment si nécessaire
- Adapter le preprocessing selon la classe : les classes avec des textes courts pourraient bénéficier d'un traitement différent

### Validation Statistique

**Test de Kruskal-Wallis pour les longueurs de descriptions (top 10 classes) :**
- **Hypothèse nulle (H0) :** Les longueurs de descriptions sont identiques entre les classes
- **Résultat :** H = 26 348.68, p-value < 0.001
- **Conclusion :** Les longueurs sont significativement différentes entre les classes. On rejette H0 avec un niveau de confiance > 99.9%

**Statistiques descriptives par classe (top 10) :**
- **Classe 2280 :** Désignations très longues (médiane ~170 caractères, IQR 100-250)
- **Classes 1160, 2403 :** Descriptions très courtes (médiane ~50 caractères)
- **Classes 1280, 1300, 1560, 1920, 2060, 2583 :** Descriptions moyennes à longues (médiane 500-900 caractères)
- **Variabilité :** Distribution très asymétrique avec de nombreux outliers (descriptions jusqu'à 12 451 caractères)

**Interprétation :** Le test confirme que les longueurs varient significativement selon la classe. Cette variabilité peut être exploitée comme feature, mais nécessite une normalisation pour éviter les biais.

--------------------------------------------------------------------------------------------------------------------------------------------

## Visualisation 3 : Analyse des Valeurs Manquantes

### Figure
**Type :** Heatmap montrant le pourcentage de valeurs manquantes par colonne et par classe (top 15 classes)

### Commentaire Métier

L'analyse des valeurs manquantes révèle un **problème majeur de qualité des données** : **35.09% des descriptions sont absentes** dans le dataset d'entraînement. Plus préoccupant encore, cette proportion varie considérablement selon la catégorie. Les classes **2403 (97.4%)**, **2280 (93.3%)**, **1160 (91.2%)** et **10 (89.2%)** ont plus de 90% de descriptions manquantes, tandis que les classes **2582 (2.4%)**, **1560 (3.5%)** et **1920 (4.8%)** sont bien documentées.

**Impact business :**
- Les catégories avec peu de descriptions (2403, 2280, 1160) devront s'appuyer principalement sur les désignations pour la classification, ce qui peut réduire la précision
- Cette inégalité de documentation crée un biais de qualité : certaines catégories sont mieux documentées que d'autres
- Le modèle risque de sous-performer sur les catégories mal documentées, ce qui peut impacter l'expérience utilisateur pour ces produits

**Recommandations :**
- **Stratégie hybride :** Utiliser la designation seule pour les produits sans description, ou créer un marqueur spécial "[DESCRIPTION_VIDE]"
- **Feature engineering :** Créer une feature binaire `has_description` pour indiquer au modèle la disponibilité de la description
- **Évaluation adaptée :** Évaluer séparément les performances sur les produits avec/sans description pour identifier les catégories problématiques
- **Amélioration future :** Prioriser l'enrichissement des descriptions pour les catégories mal documentées (2403, 2280, 1160)

### Validation Statistique

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

**Test d'indépendance (à effectuer) :**
- Un test chi-carré pourrait vérifier si la présence de descriptions manquantes est indépendante de la classe
- Les résultats suggèrent une dépendance forte (certaines classes sont systématiquement moins documentées)

**Interprétation :** Les valeurs manquantes ne sont pas aléatoires mais dépendent de la classe. Ce pattern systématique nécessite une stratégie de traitement adaptée pour chaque catégorie.

--------------------------------------------------------------------------------------------------------------------------------------------

## Visualisation 4 : Nuages de Mots par Catégorie

### Figure
**Type :** Word clouds pour les 5 classes les plus représentées (2583, 1560, 1300, 2060, 2522), basés sur les désignations et descriptions nettoyées

### Commentaire Métier

Les nuages de mots révèlent des **vocabulaires distincts et caractéristiques** pour chaque catégorie, ce qui est un signal positif pour la classification. La classe **2583** se distingue par des termes liés aux piscines ("piscine", "eau", "filtration", "hors sol", "cm"), la classe **1300** par le vocabulaire des drones ("dji", "mavic", "drone", "rc", "quadcopter"), et la classe **2522** par les produits de papeterie ("papier", "note", "bloc", "carnet", "format a5").

**Impact business :**
- La séparabilité lexicale entre les classes est bonne, ce qui suggère que les modèles de classification textuelle devraient bien performer
- Les mots-clés identifiés peuvent être utilisés pour améliorer le feature engineering (création de features basées sur la présence de mots spécifiques)
- Certaines catégories ont un vocabulaire très technique (1300 - drones), tandis que d'autres sont plus génériques (2060 - décoration)

**Recommandations :**
- **Feature engineering :** Créer des features binaires pour la présence de mots-clés discriminants identifiés dans les word clouds
- **Preprocessing adaptatif :** Conserver les termes techniques spécifiques (ex: "mavic", "quadcopter") qui sont très discriminants
- **Analyse de similarité :** Mesurer la similarité lexicale entre classes pour identifier les catégories potentiellement confondues
- **Validation :** Utiliser les mots-clés identifiés pour valider manuellement un échantillon de classifications

### Validation Statistique

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

**Mesure de séparabilité (à calculer) :**
- Un calcul de TF-IDF pourrait quantifier la spécificité de chaque mot à chaque classe
- La similarité cosinus entre les vecteurs de mots de différentes classes pourrait mesurer le chevauchement lexical

**Interprétation :** Les word clouds confirment que chaque classe possède un vocabulaire distinct, ce qui est favorable pour la classification. Les termes techniques (ex: "mavic", "quadcopter") sont particulièrement discriminants.

--------------------------------------------------------------------------------------------------------------------------------------------

## Visualisation 5 : Présence de HTML et Qualité des Descriptions

### Figure
**Type :** Graphique en barres empilées montrant la proportion de descriptions avec HTML, sans HTML, et vides, par classe (top 10)

### Commentaire Métier

L'analyse de la présence de HTML révèle une **hétérogénéité importante de la qualité des données** selon les catégories. Les classes **1560 (96%)**, **1920 (94%)**, **2060 (92%)** et **2583 (92%)** ont une très forte proportion de descriptions avec HTML, tandis que les classes **2403 (97%)**, **2280 (93%)** et **1160 (92%)** ont principalement des descriptions vides. Cette observation révèle deux problèmes distincts : certaines catégories nécessitent un nettoyage HTML intensif, tandis que d'autres manquent simplement de contenu.

**Impact business :**
- Les catégories avec beaucoup de HTML (1560, 1920, 2060, 2583) nécessiteront un preprocessing robuste pour extraire le texte utile, mais disposent d'un contenu riche une fois nettoyé
- Les catégories avec beaucoup de descriptions vides (2403, 2280, 1160) auront des performances limitées car elles ne peuvent s'appuyer que sur les désignations
- Cette inégalité de qualité crée un défi supplémentaire : le modèle devra gérer à la fois des textes avec HTML à nettoyer et des textes absents

**Recommandations :**
- **Nettoyage HTML prioritaire :** Mettre en place un pipeline de nettoyage robuste utilisant BeautifulSoup ou des expressions régulières pour extraire le texte des descriptions HTML
- **Gestion des entités HTML :** Décoder les entités HTML (&eacute;, &#39;, etc.) pour restaurer les caractères originaux
- **Stratégie hybride :** Pour les catégories avec beaucoup de HTML, nettoyer et utiliser le contenu. Pour les catégories avec beaucoup de vides, s'appuyer sur les désignations
- **Feature de qualité :** Créer des features indiquant la présence de HTML et la présence de description pour guider le modèle

### Validation Statistique

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

**Interprétation :** Le test confirme que la présence de HTML n'est pas aléatoire mais dépend fortement de la classe. Ce pattern systématique suggère que certaines catégories sont mieux formatées (avec HTML) que d'autres, ce qui peut être un indicateur de la qualité de la source de données pour chaque catégorie.

--------------------------------------------------------------------------------------------------------------------------------------------

## Synthèse des Validations Statistiques

### Tests Effectués

1. **Test de chi-carré (déséquilibre classes) :** Chi² = 36 570.33, p < 0.001 → Déséquilibre significatif confirmé
2. **Test de Kruskal-Wallis (longueurs descriptions) :** H = 26 348.68, p < 0.001 → Longueurs significativement différentes entre classes
3. **Test de chi-carré (HTML/classe) :** Chi² = 15 181.95, p < 0.001 → Présence de HTML dépendante de la classe

### Constats Principaux

- **Déséquilibre de classes :** Ratio 13.36 nécessitant des techniques de rééquilibrage
- **Variabilité des longueurs :** Nécessité de normalisation pour éviter les biais
- **Valeurs manquantes :** 35% de descriptions absentes, avec forte variabilité par classe
- **Présence de HTML :** 18% des descriptions contiennent du HTML, nécessitant un nettoyage
- **Séparabilité lexicale :** Vocabulaires distincts par classe, favorable pour la classification

### Recommandations Globales

1. **Preprocessing prioritaire :** Nettoyage HTML, gestion des valeurs manquantes, normalisation des longueurs
2. **Rééquilibrage des classes :** Utilisation de SMOTE ou class weights pour améliorer les performances sur les classes minoritaires
3. **Feature engineering :** Création de features de longueur, présence de description, présence de HTML
4. **Évaluation adaptée :** Utilisation de métriques adaptées au déséquilibre (F1-score par classe)

