# Rapport détaillé — Partie Classification Textuelle

> **À l’attention des rédacteurs du rapport final** : ce document décrit en détail la partie « classification par le texte » du projet. Les termes techniques sont expliqués en fin de document (glossaire).

---

## 1. Contexte et objectifs

### 1.1 Enjeu métier
L’objectif est d’**assigner automatiquement une catégorie de produit** (par ex. « Piscine / Spa », « Jouets », « Fournitures Bureau ») à partir des **textes** associés : titre du produit (désignation) et description.

### 1.2 Périmètre
- **Données** : ~85 000 produits avec texte (désignation + description).
- **Classes** : 27 catégories (`prdtypecode`), décrites dans `reference_classes.md`.
- **Modalité** : uniquement le texte — aucune image utilisée dans cette partie.

### 1.3 Pourquoi cette approche
- Valider que la chaîne de traitement texte fonctionne correctement.
- Obtenir une **référence de performance** fiable avant d’optimiser.
- Comparer ensuite avec la modalité image et la fusion multimodale.

---

## 2. Données utilisées

### 2.1 Sources
- **Fichier** : `data/processed/` (données nettoyées issues des notebooks 01–03).
- **Champs texte** : `designation` (titre) et `description` (texte enrichi).

### 2.2 Variable texte finale : `text_combined`
Les deux champs sont fusionnés en une seule chaîne de caractères pour chaque produit.  
**Objectif** : enrichir le signal sémantique (ex. un produit avec peu de mots dans le titre mais une description détaillée).

### 2.3 Labels (à prédire)
- **27 classes** : catégories détaillées (ex. 2583 = Piscine / Spa).
- **24 classes (superclasse)** : variante où les 4 catégories « publications » (livres, presse, BD, romans) sont regroupées en une seule.  
  → Permet de tester si une granularité réduite améliore les performances.

---

## 3. Transformation du texte : TF-IDF

### 3.1 Qu’est-ce que le TF-IDF ?
Le **TF-IDF** (Term Frequency – Inverse Document Frequency) transforme le texte en **vecteur numérique** :
- Chaque mot (ou combinaison de mots) devient une « dimension ».
- La valeur associée mesure à la fois la fréquence du mot dans le document et sa rareté dans l’ensemble des documents.

### 3.2 Paramètres utilisés
- `max_features=10000` : on garde au maximum 10 000 termes les plus fréquents.
- `min_df=2` : un terme doit apparaître dans au moins 2 documents.
- `max_df=0.95` : on exclut les termes présents dans plus de 95 % des documents (mots trop génériques).
- `ngram_range=(1, 2)` : on prend les mots seuls (unigrammes) et les paires de mots consécutifs (bigrammes).

### 3.3 Pourquoi le TF-IDF ?
- Méthode standard, rapide et interprétable.
- Adapté aux textes courts ou moyens (titres, descriptions produit).
- Facile à déployer en production.

---

## 4. Partage des données (entraînement / validation)

- **Règle** : 80 % pour l’entraînement, 20 % pour la validation.
- **Stratification** : la proportion de chaque classe est conservée dans les deux ensembles.
- **Graine aléatoire** : `random_state=42` pour assurer la reproductibilité.

---

## 5. Modèles testés (baselines)

Quatre modèles de référence ont été comparés :

| Modèle | Description (pour non-techniciens) |
|--------|-----------------------------------|
| **Naive Bayes** | Modèle probabiliste simple, très rapide. |
| **Logistic Regression** | Modèle linéaire robuste, souvent performant en classification texte. |
| **SVM (Linear)** | Séparateur à marge maximale ; excellente réputation en NLP. |
| **Random Forest** | Ensemble d’arbres de décision ; moins adapté au texte très dimensionnel. |

---

## 6. Métriques d’évaluation

### 6.1 Pourquoi ne pas se fier uniquement à l’accuracy ?
L’**accuracy** (taux de bonnes réponses) peut être trompeuse quand les classes sont déséquilibrées : un modèle qui prédit toujours la classe la plus fréquente (ex. Piscine) aurait une accuracy élevée tout en ignorant les classes rares.

### 6.2 Métriques utilisées
| Métrique | Signification |
|----------|---------------|
| **F1-macro** | Moyenne du F1 sur toutes les classes, en donnant le même poids à chaque classe. **Métrique principale du projet.** |
| **F1-weighted** | Moyenne pondérée par l’effectif des classes. Complémentaire. |
| **Accuracy** | Pourcentage de prédictions correctes (indicatif). |
| **Precision / Recall** | Précision : parmi les prédictions positives, combien sont correctes. Recall : parmi les vrais positifs, combien sont détectés. |

### 6.3 Validation croisée
- **5 folds** : les données sont découpées en 5 parties ; à chaque fois, on entraîne sur 4 parties et on valide sur la 5e.
- **Objectif** : estimer la stabilité des résultats et limiter le hasard d’un seul découpage train/val.

---

## 7. Résultats obtenus

### 7.1 Baselines (27 classes)

| Modèle | Accuracy | F1-macro | F1-weighted |
|--------|----------|----------|-------------|
| Logistic Regression | 79,1 % | **0,772** | 0,791 |
| Naive Bayes | 72,8 % | 0,695 | 0,718 |
| Random Forest | 52,9 % | 0,460 | 0,511 |
| **SVM (Linear)** | **80,2 %** | **0,785** | **0,801** |

→ **Meilleur modèle** : SVM linéaire, avec F1-macro ≈ 0,78.

### 7.2 Superclasse (24 classes)

| Modèle | Accuracy | F1-macro | F1-weighted |
|--------|----------|----------|-------------|
| **SVM (Linear)** | **84,9 %** | **0,814** | **0,845** |
| Logistic Regression | 83,1 % | 0,791 | 0,826 |
| Naive Bayes | 77,2 % | 0,710 | 0,762 |
| Random Forest | 54,7 % | 0,429 | 0,513 |

→ En regroupant les publications, on gagne environ **3 points de F1-macro**.

### 7.3 Détail par classe (SVM, 27 classes)
Exemples de classes bien ou mal prédites :
- **Très bonnes** : Piscine (2583), Drones (1300), Textile (1920), Cartes (1160).
- **Difficiles** : Jouets (1280), Jeux éducatifs (1281), Jardin (2582), Mobilier (1560).

---

## 8. Optimisation

### 8.1 Optimisation du paramètre C (SVM)
- **C** contrôle le compromis entre simplicité du modèle et ajustement aux données.
- Une recherche (grille de valeurs) a été effectuée pour choisir un C optimal.

### 8.2 Rééquilibrage des classes (class weights)
- Les modèles ont été testés avec `class_weight='balanced'` pour donner plus de poids aux classes rares.
- **Résultat** : le F1-macro **diminue** légèrement (−0,9 % en 27 classes, −1,7 % en 24 classes).
- **Conclusion** : le rééquilibrage n’apporte pas d’amélioration sur ce jeu de données.

---

## 9. Pourquoi certains modèles sont moins performants ?

| Modèle | Limite principale |
|--------|-------------------|
| **Naive Bayes** | Hypothèse d’indépendance des mots trop forte ; sensible aux mots rares. |
| **Random Forest** | Peu adapté aux milliers de dimensions creuses (TF-IDF). |
| **SVM / Logistic Regression** | Très adaptés aux espaces haute dimension et aux données creuses. |

---

## 10. Pipeline technique (résumé)

1. Chargement des données nettoyées.
2. Vérification de l’alignement entre textes et labels.
3. Fusion `designation` + `description` → `text_combined`.
4. Vectorisation TF-IDF.
5. Split stratifié 80/20.
6. Entraînement des 4 modèles baselines.
7. Calcul des métriques (Accuracy, F1-macro, F1-weighted).
8. Validation croisée (5 folds).
9. Optimisation ciblée (SVM, class weights).

---

## 11. Sorties et livrables

- **Modèles** : sauvegardés dans `models/` (SVM, Logistic Regression, etc.).
- **Vectoriseur TF-IDF** : sauvegardé pour réutilisation.
- **Label encoder** : pour garder la correspondance code → nom de classe.
- **Notebooks** : 04 (baselines), 05 (optimisation).

---

## 12. Limites identifiées

- **Déséquilibre des classes** : certaines classes (ex. Piscine) sont très surreprésentées, d’autres (ex. Warhammer) très sous-représentées.
- **Expressivité sémantique** : les modèles linéaires ne capturent pas des nuances complexes (ex. synonymes, reformulations).
- **Confusions lexicales** : les erreurs se concentrent sur des classes proches (publications, bébé, jouets, jardin).

---

## 13. Recommandations pour la suite

- Tester des modèles plus avancés (ex. CamemBERT, BERT) pour améliorer la compréhension sémantique.
- Intégrer éventuellement des champs métier (marques, catégories).
- Comparer avec la modalité image et la fusion multimodale (notebook 12).

---

## 14. Synthèse pour le rapport final

- Le **pipeline texte** est stable, reproductible et performant.
- **Meilleur modèle** : SVM linéaire avec F1-macro ≈ 0,78 (27 classes) et 0,81 (24 classes).
- Le rééquilibrage des classes **n’améliore pas** les résultats.
- Le texte seul constitue une **référence solide** pour la fusion multimodale.

---

## Glossaire (termes techniques)

| Terme | Définition simple |
|-------|-------------------|
| **TF-IDF** | Transformation du texte en vecteur de nombres, basée sur la fréquence des mots et leur rareté. |
| **F1-macro** | Moyenne du F1 de chaque classe, toutes les classes ayant le même poids. |
| **F1-weighted** | Moyenne du F1 pondérée par le nombre d’exemples par classe. |
| **Stratification** | Conservation des proportions de chaque classe dans les ensembles train/val. |
| **Validation croisée** | Découpage en plusieurs parties pour estimer la performance de façon plus robuste. |
| **Class weights** | Pondération des classes pour compenser le déséquilibre. |
| **SVM** | Support Vector Machine — algorithme de classification par séparation linéaire. |
