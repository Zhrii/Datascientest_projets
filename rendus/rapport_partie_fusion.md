# Rapport détaillé — Partie Fusion Multimodale

> **À l’attention des rédacteurs du rapport final** : ce document décrit en détail la partie « fusion texte + image » du projet. Les termes techniques sont expliqués en fin de document (glossaire).

---

## 1. Contexte et objectifs

### 1.1 Enjeu métier
L’objectif est de **combiner les informations texte et image** pour :
- **Améliorer la classification** : le texte et l’image apportent des informations complémentaires.
- **Permettre le matching** : rechercher des images à partir d’une requête texte (et inversement), via des représentations alignées.

### 1.2 Périmètre
- **Données** : produits ayant **à la fois** un texte et une image valide (~44 969 paires).
- **Classes** : mêmes 27 catégories (`prdtypecode`) que les parties texte et image.

### 1.3 Intérêt de la fusion
- Le **texte** est souvent plus riche pour certains produits (descriptions détaillées).
- L’**image** peut compenser quand le texte est absent ou peu informatif.
- La **combinaison** vise à dépasser les performances de chaque modalité prise séparément.

---

## 2. Données et alignement

### 2.1 Construction des paires
- **Fonction** : `load_text_image_pairs()`.
- **Méthode** : jointure (merge) sur `productid` et `imageid` pour associer chaque texte à son image.

### 2.2 Problème d’alignement (corrigé)
Un **bug d’index** a été identifié et corrigé :
- Lors du merge entre textes et images, les indices (index pandas) étaient réinitialisés.
- Les labels (`prdtypecode`) restaient associés aux anciens indices, ce qui provoquait un **désalignement** : un label pouvait être associé à la mauvaise paire (texte, image).
- **Conséquence** : les scores F1 chutaient à des valeurs aberrantes (~0,02, pire qu’aléatoire).
- **Correction** : jointure des labels avec les données avant le merge avec les images, puis utilisation directe de `prdtypecode` dans le dataframe fusionné.

### 2.3 Partage des données
- Split 80/20, stratifié, même graine (`random_state=42`) que les autres parties.

---

## 3. Représentations extraites par modalité

| Modalité | Extracteur | Dimensions | Remarque |
|----------|------------|------------|----------|
| Texte | TF-IDF + TruncatedSVD | 300 | Réduction de dimension pour limiter la taille du vecteur fusionné. |
| Image | ResNet50 | 2048 | Features visuelles classiques. |
| Image + Texte | CLIP ViT-B/32 | 512 chacune | Espace partagé texte–image, aligné sémantiquement. |

Toutes les features sont **normalisées L2** avant concaténation ou combinaison.

---

## 4. Approches de fusion testées

### 4.1 Late Fusion (concaténation)
On **concatène** les vecteurs texte et image en un seul vecteur, puis on entraîne un classificateur dessus.

| Combinaison | Dimensions | Description |
|-------------|------------|-------------|
| TF-IDF + ResNet50 | 300 + 2048 = **2348** | Texte classique + image classique. |
| TF-IDF + CLIP image | 300 + 512 = **812** | Texte classique + image CLIP. |
| TF-IDF + CLIP image + CLIP texte | 300 + 512 + 512 = **1324** | Texte classique + texte CLIP + image CLIP. |

**Classificateurs** : Logistic Regression, LinearSVC, avec `class_weight='balanced'`.

### 4.2 CLIP multimodal
CLIP encode **texte et image** dans un même espace 512d. On combine les deux vecteurs :
- **Concaténation** : [clip_text ; clip_image] → 1024 dimensions.
- **Moyenne** : (clip_text + clip_image) / 2 → 512 dimensions.

**Avantage** : alignement sémantique naturel entre texte et image.

### 4.3 Matching texte ↔ image
- **Similarité cosinus** : on mesure la proximité entre l’embedding CLIP du texte et celui de l’image.
- **Recall@K** : pour une requête (texte ou image), on regarde si la paire correcte est dans les K plus proches voisins.
- **Application** : recherche d’images par requête textuelle (ex. « canapé gris ») ou d’images similaires à une image donnée.

---

## 5. Résultats obtenus (classification)

### 5.1 Classement complet des approches (F1-macro)

| Rang | Approche | F1-macro |
|------|----------|----------|
| 1 | **Fusion : TF-IDF + CLIP img + CLIP txt (SVM Linear)** | **0,841** |
| 2 | Fusion : TF-IDF + CLIP img + CLIP txt (Logistic Regression) | 0,832 |
| 3 | Fusion : CLIP concat 1024d (SVM Linear) | 0,823 |
| 4 | Fusion : CLIP moyenne 512d (SVM Linear) | 0,803 |
| 5 | Fusion : TF-IDF + CLIP img (SVM Linear) | 0,800 |
| 6 | Fusion : TF-IDF + CLIP img (Logistic Regression) | 0,797 |
| 7 | CLIP texte seul | 0,796 |
| 8 | Fusion : TF-IDF + ResNet50 (SVM Linear) | 0,764 |
| 9 | Fusion : TF-IDF + ResNet50 (Logistic Regression) | 0,751 |
| 10 | Texte seul (TF-IDF) | 0,695 |
| 11 | CLIP image seule | 0,671 |
| 12 | Image seule (ResNet50) | 0,514 |

### 5.2 Principaux enseignements

1. **La meilleure fusion** (TF-IDF + CLIP image + CLIP texte + SVM) atteint **F1-macro 0,84**, soit environ **+5 points** par rapport au texte seul TF-IDF (0,70) et **+17 points** par rapport à l’image seule ResNet50 (0,51).

2. **CLIP multimodal** (concat ou moyenne) performe bien : 0,80–0,82, grâce à l’alignement sémantique.

3. **Texte seul TF-IDF** (0,70) dans ce contexte multimodal est moins performant que le texte seul sur le dataset complet (où on avait ~0,78). La différence s’explique par le **sous-ensemble** utilisé : seuls les produits avec image sont pris en compte, ce qui change légèrement la distribution.

4. **Image seule** reste la modalité la plus faible (0,51 avec ResNet50, 0,67 avec CLIP).

---

## 6. Résultats du matching (Recall@K)

Sur un échantillon de paires (texte, image) :
- **Recall@1** : 40,5 % — la paire correcte est en 1re position dans 40,5 % des cas.
- **Recall@5** : 60,6 % — la paire correcte est dans le top 5 dans 60,6 % des cas.
- **Recall@10** : 69,1 %.
- **Recall@20** : 77,1 %.

**Similarité moyenne** :
- Paires correctes (diagonale) : 0,31.
- Paires incorrectes : 0,08.

→ L’écart entre paires correctes et incorrectes montre que CLIP distingue bien les bonnes associations, mais le Recall@1 reste modéré (domaine e-commerce non optimisé pour CLIP).

---

## 7. Pipeline technique (résumé)

1. Chargement des paires (texte, image) via `load_text_image_pairs()` avec jointure correcte des labels.
2. Extraction des features : TF-IDF (sur `text_combined`), ResNet50, CLIP (image + texte).
3. Mise en cache : `multimodal_resnet_cache.npz`, `multimodal_clip_cache.npz`.
4. Fusion : concaténation ou moyenne selon la configuration.
5. Classification : LR, SVM sur les vecteurs fusionnés.
6. Matching : matrice de similarité cosinus, calcul du Recall@K.

---

## 8. Points d’attention

### 8.1 Produits sans image
- **~45 % des produits** n’ont pas d’image exploitable.
- La fusion ne s’applique qu’aux produits avec image.
- **En production** : prévoir un **fallback texte-seul** pour les produits sans image.

### 8.2 Limitation de CLIP
- **Troncature à 77 tokens** : les descriptions longues perdent de l’information.
- **Pas de fine-tuning** : CLIP est utilisé tel quel, non adapté au domaine e-commerce.

### 8.3 Bug d’alignement
- Un bug d’alignement des labels a conduit à des F1 ~0,02 avant correction.
- Après correction, les résultats sont cohérents avec les attentes.

---

## 9. Sorties et livrables

- **Caches** : ResNet et CLIP pour les features multimodales.
- **Comparaison** : `multimodal_comparison.csv`, graphique des approches.
- **Démo matching** : visualisation de la similarité texte–image, exemples de recherche.
- **Notebook** : 12 (fusion multimodale).

---

## 10. Limites identifiées

- La fusion peut ne pas toujours surpasser le texte seul si les images apportent peu de signal (ex. produits très similaires visuellement).
- Classes **visuellement ambiguës** : publications, DLC, textiles restent difficiles.
- Pas de **Dual Encoder** entraîné de bout en bout ; utilisation de CLIP pré-entraîné uniquement.

---

## 11. Recommandations

- **Fine-tuning CLIP** ou Dual Encoder sur les paires produit pour améliorer l’alignement au domaine e-commerce.
- **Ensemble** : combiner les prédictions texte-seul, image-seul et fusion (vote ou moyenne).
- **Recall@K** : exploiter le matching pour la recherche produits et les recommandations.

---

## 12. Synthèse pour le rapport final

- La **fusion multimodale** permet de dépasser nettement le texte seul : F1-macro jusqu’à **0,84**.
- **Meilleure configuration** : TF-IDF + CLIP image + CLIP texte, avec SVM.
- Le **matching CLIP** permet la recherche par similarité (texte→image, image→texte), avec Recall@1 ≈ 40 %.
- Un **système de fallback** (texte seul) est indispensable pour les produits sans image.

---

## Glossaire (termes techniques)

| Terme | Définition simple |
|-------|-------------------|
| **Late fusion** | Combinaison des représentations texte et image par concaténation, suivie d’un classificateur. |
| **TruncatedSVD** | Réduction de dimension pour compresser les vecteurs TF-IDF. |
| **Recall@K** | Pour une requête, proportion de cas où la bonne réponse est dans les K premières. |
| **Similarité cosinus** | Mesure de similarité entre deux vecteurs (entre 0 et 1). |
| **Embedding** | Représentation vectorielle d’un texte ou d’une image. |
| **Normalisation L2** | Mise à l’échelle des vecteurs pour que leur norme soit 1. |
