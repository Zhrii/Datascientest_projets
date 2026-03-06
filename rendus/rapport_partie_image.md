# Rapport détaillé — Partie Classification par Images

> **À l’attention des rédacteurs du rapport final** : ce document décrit en détail la partie « classification par les images » du projet. Les termes techniques sont expliqués en fin de document (glossaire).

---

## 1. Contexte et objectifs

### 1.1 Enjeu métier
L’objectif est de **prédire la catégorie produit** à partir uniquement de l’**image** associée au produit (sans utiliser le texte). Cela permet :
- De vérifier si le visuel suffit à classer un produit.
- De comparer avec la modalité texte et la fusion multimodale.
- D’anticiper les cas où le texte est absent ou peu fiable.

### 1.2 Périmètre
- **Données** : ~45 000 images nettoyées (produits ayant une image valide et conforme).
- **Classes** : mêmes 27 catégories (`prdtypecode`) que la partie texte.
- **Modalité** : uniquement l’image — aucun texte utilisé.

### 1.3 Différence avec la partie texte
- Environ **45 % des produits** n’ont pas d’image exploitable (données manquantes ou images invalides).
- Le nombre d’images est donc inférieur au nombre total de produits texte (~45 k vs ~85 k).

---

## 2. Données et préparation

### 2.1 Source des images
- **Dossier** : `data/processed/image_clean/`
- **Convention de nommage** : `image_{imageid}_product_{productid}.jpg`

### 2.2 Prétraitement (notebooks 07–08)
1. **Validation** : détection et suppression des images corrompues ou illisibles.
2. **Redimensionnement** : toutes les images sont mises au format 224×224 pixels (standard des modèles utilisés).
3. **Normalisation** : valeurs des pixels centrées et réduites selon les statistiques ImageNet (mean/std).
4. **Dataset final** : tableau avec chemins d’images et labels alignés.

### 2.3 Partage des données
- Split **80 % entraînement / 20 % validation**, stratifié par classe.
- Graine aléatoire `random_state=42` pour reproductibilité.

---

## 3. Approches d’extraction de représentations visuelles

Trois familles d’approches ont été testées :

### 3.1 ResNet50 (baseline)

**Qu’est-ce que c’est ?**  
ResNet50 est un réseau de neurones profond pré-entraîné pour reconnaître des objets (ImageNet). On utilise la sortie de la couche juste avant le classificateur final.

- **Source** : modèle pré-entraîné ImageNet (torchvision).
- **Dimensions** : 2048 valeurs par image (vecteur de « features »).
- **Rôle** : baseline classique en vision par ordinateur, rapide à extraire.

### 3.2 CLIP (ViT-B/32)

**Qu’est-ce que c’est ?**  
CLIP est un modèle **vision–langage** : entraîné sur des paires (image, texte), il produit des représentations alignées entre images et textes. Les features sont plus sémantiques que celles d’un modèle purement visuel.

- **Source** : OpenCLIP (ViT-B/32) pré-entraîné sur LAION-2B.
- **Dimensions** : 512 valeurs par image, normalisées L2.
- **Atout** : meilleure compréhension du « sens » de l’image (concepts, contexte).

### 3.3 EfficientNet-B0 (fine-tuning)

**Qu’est-ce que c’est ?**  
EfficientNet-B0 est un réseau de classification d’images. On a « affiné » (fine-tuned) le modèle sur nos données produits : les dernières couches du réseau sont réentraînées pour s’adapter au domaine e-commerce.

- **Source** : EfficientNet-B0 pré-entraîné ImageNet.
- **Stratégie** : gel des couches basses, dégel des 2 derniers blocs + classificateur.
- **Limite** : entraînement sur CPU, peu d’epochs — gain modéré.

---

## 4. Classificateurs sur les features extraites

Sur les vecteurs de features (ResNet, CLIP), on entraîne des classificateurs « classiques » :

| Classificateur | Description |
|----------------|-------------|
| **Logistic Regression** | Modèle linéaire simple, avec `class_weight='balanced'` pour tenir compte du déséquilibre des classes. |
| **LinearSVC** | SVM linéaire, avec `class_weight='balanced'`. |
| **LightGBM, XGBoost, CatBoost** | Algorithmes de gradient boosting, avec pondération des classes. |
| **MLP** | Réseau de neurones (512, 256 neurones) sur les features CLIP, avec StandardScaler et early stopping. |

Pour CLIP ViT-L/14, une recherche par grille (GridSearchCV) sur le paramètre `C` a été effectuée pour optimiser le F1-macro.

---

## 5. Résultats observés

### 5.1 Vue d’ensemble (F1-macro en validation)

| Approche | F1-macro | Remarque |
|----------|----------|----------|
| ResNet50 + Logistic Regression | **0,507** | Baseline image |
| ResNet50 + LightGBM | **0,524** | Légère amélioration |
| EfficientNet-B0 (fine-tuning) | **0,554** | Gain par adaptation au domaine |
| CLIP ViT-B/32 + Logistic Regression | **0,658** | Net progrès grâce à CLIP |
| CLIP ViT-B/32 + SVM Linear | **0,663** | Meilleur résultat image seul |
| Texte seul (TF-IDF + SVM) | ~0,78 | Référence pour comparaison |

→ **Conclusion** : CLIP apporte une nette amélioration par rapport à ResNet50 et EfficientNet. L’image seule reste toutefois moins performante que le texte seul.

### 5.2 Détail par extracteur

**ResNet50 (baseline)** :
- Logistic Regression : F1-macro 0,5066 (avec class weights) / 0,5171 (sans).
- Naive Bayes, Random Forest, SVM : moins performants.
- LightGBM : meilleur modèle ResNet50 avec F1-macro 0,524.

**CLIP ViT-B/32** :
- SVM : F1-macro 0,6634 (meilleur résultat image).
- Logistic Regression : F1-macro 0,6580.

**EfficientNet-B0** :
- Fine-tuning sur 10 epochs : F1-macro final 0,5538.
- Gain modéré par rapport à ResNet50, mais inférieur à CLIP.

### 5.3 Détail par classe (CLIP + SVM)

Exemples de performances par classe (précision, rappel, F1) :
- **Très bonnes** : Cartes (1160) 0,97, Piscine (2583) 0,86, Textile (1920) 0,85, Consoles portables (60) 0,86.
- **Moyennes** : Drones (1300) 0,79, Décoration (2060) 0,66, Mobilier (1560) 0,60.
- **Difficiles** : Jouets (1280) 0,41, Jeux éducatifs (1281) 0,35, Jardin (2582) 0,48, Accessoires gaming (50) 0,50.

---

## 6. Difficultés spécifiques à la modalité image

### 6.1 Classes visuellement proches
- **Mobilier (1560) vs Décoration (2060)** : meubles et luminaires partagent des formes et couleurs similaires.
- **Jardin (2582) vs Piscine (2583) vs Outils jardin (2585)** : univers extérieur commun.
- **Figurines (1140) vs Warhammer (1180)** : figurines de collection vs figurines de jeu.

### 6.2 Classe 2905 (DLC / codes de téléchargement)
- Images souvent : texte ou logo sur fond blanc.
- Peu discriminantes visuellement.
- ~90 images seulement après nettoyage — très petite classe.

### 6.3 Classe 1920 (Textile)
- Beaucoup d’images très similaires (draps, couettes sur fond blanc).
- Difficile de distinguer les produits entre eux.

### 6.4 Déséquilibre des classes
- Ratio max/min > 40× entre certaines classes (ex. Piscine ~3 400 images vs DLC ~90).

---

## 7. Pipeline technique (résumé)

1. Chargement du dataset images avec labels (notebook 09).
2. Extraction des features (ResNet50, CLIP) avec mise en cache pour éviter les re-calculs.
3. Split stratifié 80/20.
4. Entraînement des classificateurs (LR, SVM, LightGBM, etc.) sur les features.
5. Pour EfficientNet : fine-tuning du réseau entier (ou des dernières couches) sur les images.
6. Évaluation : F1-macro, F1-weighted, accuracy.

---

## 8. Sorties et livrables

- **Caches** : `clip_features_cache_vit_l14.npz`, `image_features_cache.npz`.
- **Modèles** : `efficientnet_best.pth`, classifieurs sklearn dans `models/`.
- **Label encoder** : `label_encoder_image.pkl`.
- **Comparaisons** : `image_advanced_comparison.csv`, `image_all_models_comparison.png`.
- **Notebooks** : 07 (exploration), 08 (traitement), 09 (baseline), 10 (optimisation), 11 (avancé).

---

## 9. Limites identifiées

- Performances **inférieures au texte seul** : image ~0,66 F1-macro vs texte ~0,78.
- Fine-tuning EfficientNet **limité** (CPU, peu d’epochs).
- Certaines classes **peu discriminables visuellement** : publications, DLC, textiles.
- **Déséquilibre fort** entre classes (ratio > 40×).

---

## 10. Recommandations

- Utiliser **CLIP ViT-L/14** comme extracteur principal si ressources suffisantes.
- Tester un **ensemble** CLIP + EfficientNet pour combiner forces visuelles et sémantiques.
- En production : privilégier la **fusion multimodale** (texte + image) pour maximiser la performance.

---

## 11. Synthèse pour le rapport final

- Le pipeline image est **opérationnel** avec CLIP comme meilleure approche.
- **Meilleur résultat image seul** : CLIP + SVM, F1-macro ≈ 0,66.
- Les **features CLIP** sont nettement plus discriminantes que ResNet50 et EfficientNet.
- L’image seule **ne dépasse pas le texte** ; la fusion multimodale est la voie pour améliorer la classification globale.

---

## Glossaire (termes techniques)

| Terme | Définition simple |
|-------|-------------------|
| **Features** | Vecteur de nombres représentant une image (ou un texte) pour le modèle. |
| **ResNet50** | Réseau de neurones profond pré-entraîné pour la reconnaissance d’objets. |
| **CLIP** | Modèle vision–langage qui aligne images et textes dans un même espace. |
| **ViT-B/32** | Variante de CLIP basée sur un Transformer pour les images (ViT = Vision Transformer). |
| **Fine-tuning** | Réentraînement partiel d’un modèle pré-entraîné sur de nouvelles données. |
| **Early stopping** | Arrêt de l’entraînement quand la performance sur la validation ne s’améliore plus. |
| **StandardScaler** | Normalisation des features (moyenne 0, écart-type 1). |
