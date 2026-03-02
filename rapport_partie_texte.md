# Rapport détaillé — Partie Texte

## 1) Objectif et périmètre
- **Objectif principal** : établir une **référence de performance** pour la classification des produits à partir du texte (`prdtypecode`, 27 classes).
- **Pourquoi** : disposer d’un **baseline fiable** avant d’optimiser, et valider que la chaîne de traitement texte est correcte.
- **Périmètre** : uniquement la **modalité texte** (pas d’images), sur les données d’entraînement.

## 2) Données et préparation
- **Source** : données nettoyées et préparées dans `data/processed/`.
- **Champs utilisés** : `designation` + `description`.
- **Variable texte finale** : `text_combined`.
  - **Pourquoi** : regrouper les deux champs augmente le signal sémantique disponible par produit.
- **Labels** : `prdtypecode` (27 classes) et version **superclass** (24 classes) issue du regroupement des classes publications.
  - **Pourquoi la superclass** : test d’un regroupement pour limiter la granularité et stabiliser les performances.

## 3) Vectorisation du texte
- **Méthode** : **TF‑IDF**.
- **Justification** :
  - méthode standard en NLP classique, performante en classification.
  - robuste sur textes courts / descriptions produit.
  - rapide, interprétable, facile à industrialiser.

## 4) Split entraînement / validation
- **Méthode** : split stratifié.
- **Seed** : `random_state=42`.
- **Pourquoi** :
  - conserver la **distribution des classes** dans les deux ensembles.
  - garantir la **reproductibilité** des résultats.

## 5) Baselines testées (Notebook 04)
Modèles choisis pour leur **simplicité**, **robustesse**, et **référence standard** en NLP :

1. **Naive Bayes**  
   - Très rapide, baseline classique.
2. **Logistic Regression**  
   - Linéaire, robuste, souvent performante en TF‑IDF.
3. **SVM linéaire (LinearSVC)**  
   - Modèle très compétitif en classification texte.
4. **Random Forest**  
   - Modèle non linéaire de référence, souvent moins performant en bag‑of‑words mais utile pour comparaison.

## 6) Évaluation et métriques
- **Accuracy** : utile, mais insuffisante en cas de classes déséquilibrées.
- **F1‑macro** : métrique principale, **pondère toutes les classes de manière égale**.
- **F1‑weighted** : complémentaire, donne plus de poids aux classes fréquentes.

**Pourquoi F1‑macro ?**  
Les classes sont déséquilibrées : un modèle peut avoir une bonne accuracy en favorisant les classes majoritaires, ce que F1‑macro pénalise.

## 7) Validation croisée (Notebook 04)
- **CV 5 folds** pour les modèles baselines.
- **Pourquoi** :
  - mesurer la stabilité des résultats,
  - éviter le biais d’un seul split train/val,
  - obtenir une estimation plus robuste.

## 8) Optimisation (Notebook 05)

### 8.1) Optimisation SVM
- **Recherche d’hyperparamètres** (ex : `C`).
- **Pourquoi** : SVM linéaire est souvent le meilleur modèle sur TF‑IDF ; optimiser `C` peut améliorer légèrement la généralisation.

### 8.2) Rééquilibrage des classes (class weights)
- **Test effectué** : SVM avec `class_weight='balanced'`.
- **Résultat** :  
  - F1‑macro **baisse** légèrement par rapport à la baseline.
  - Exemple observé :  
    - 27 classes : F1‑macro ↓ (~‑0.9 %)  
    - 24 classes : F1‑macro ↓ (~‑1.7 %)

**Conclusion** :  
Le rééquilibrage **n’apporte pas d’amélioration** sur ce dataset.  

## 9) Détails du pipeline (pas à pas)
1. **Chargement des données**  
   - Validation de l’alignement `X_train` / `y_train`.  
   - Vérification de la présence de `text_combined`.
2. **Vectorisation TF‑IDF**  
   - Transformation des textes en matrice creuse de features.  
   - Conservation du vectoriseur pour la reproductibilité.
3. **Split stratifié**  
   - Conservation de la distribution des classes.
4. **Entraînement des baselines**  
   - Entraînement des modèles sur le même split pour comparaison équitable.
5. **Évaluation**  
   - Calcul Accuracy / F1‑macro / F1‑weighted.
6. **Validation croisée**  
   - Robustesse des performances sur plusieurs folds.
7. **Optimisation ciblée**  
   - SVM optimisé et test de class weights.

## 10) Pourquoi certains modèles sont moins performants
- **Naive Bayes** :  
  hypothèses d’indépendance trop fortes, sensible aux mots rares.
- **Random Forest** :  
  peu adapté aux matrices TF‑IDF très dimensionnelles et creuses.
- **SVM / LogReg** :  
  adaptés aux espaces haute dimension, souvent les meilleurs en NLP.

## 11) Analyse des résultats
- **F1‑macro** faible sur les classes rares :  
  le modèle a du mal à généraliser sur les classes peu représentées.
- **F1‑weighted** supérieur :  
  reflète la bonne performance sur les classes majoritaires.
- **Superclass (24 classes)** :  
  parfois un léger gain, mais peut **masquer** les difficultés de classes rares.

## 12) Pourquoi le rééquilibrage n’a pas aidé
Plusieurs raisons possibles :
- le dataset est déjà suffisamment riche pour les classes dominantes,
- les poids amplifient le bruit des classes rares,
- le modèle se déstabilise (meilleure couverture mais baisse globale du score).

## 13) Sorties et livrables générés
- **Modèles** : enregistrés dans `models/`  
  - SVM / LogReg / autres baselines selon le meilleur score.
- **Vectoriseur** : TF‑IDF sauvegardé.  
- **Label encoder** : sauvegardé pour la compatibilité train/test.

## 14) Limites identifiées
- **Déséquilibre fort** des classes.  
- Les modèles linéaires restent limités en expressivité sémantique.
- Les erreurs sont concentrées sur des classes proches lexicalement.

## 15) Recommandations pour la suite
- Tester un **modèle textuel plus avancé** (ex. CamemBERT / BERT) si budget temps OK.
- Intégrer éventuellement des **caractéristiques métiers** (marques, catégories).
- Continuer la comparaison avec la **modalité image** et la fusion multimodale.

## 16) Conclusion globale (texte)
- Le pipeline texte est **stable**, reproductible et performant en baseline.  
- Le **SVM linéaire** reste la meilleure référence.  
- Le **rééquilibrage** n’apporte pas de gain mesurable sur ce dataset.  
- Les optimisations futures doivent viser un meilleur modèle sémantique (transformers) et une fusion multimodale.
