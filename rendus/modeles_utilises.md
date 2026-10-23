# Modèles utilisés dans le projet

> Guide oral : explications claires, simples et concises. Paramètres utilisés dans le projet.

---

## 1. Vectorisation et réduction de dimension

### TF-IDF (Term Frequency – Inverse Document Frequency)

**Fonctionnement** : transforme le texte en vecteur de nombres. Chaque mot reçoit un poids : fort s’il est fréquent dans le document et rare dans le corpus, faible s’il est trop commun. On obtient une représentation numérique du contenu sémantique.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `max_features` | 10 000 | Nombre max de termes (dimensions) |
| `min_df` | 2 | Un terme doit apparaître dans au moins 2 documents |
| `max_df` | 0,95 | Exclut les termes présents dans plus de 95 % des docs |
| `ngram_range` | (1, 2) | Unigrammes et bigrammes |
| `stop_words` | french | Suppression des mots courants (le, de, et…) |

---

### TruncatedSVD

**Fonctionnement** : réduit le nombre de dimensions d’un vecteur en gardant l’essentiel de l’information. Utile pour fusionner TF-IDF (10 000 dim) avec des features image (512 ou 2048 dim) sans que le texte domine.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `n_components` | 300 | Nombre de dimensions finales |

---

## 2. Extracteurs de features images

### ResNet50

**Fonctionnement** : réseau de neurones pré-entraîné sur ImageNet. On utilise la couche avant la classification pour obtenir un vecteur de 2048 valeurs par image. Pas d’entraînement — on extrait uniquement des représentations visuelles de haut niveau.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `weights` | ImageNet1K_V1 | Pré-entraînement |
| `output_dim` | 2048 | Dimension des features |
| Taille image | 224×224 | Format attendu par le réseau |
| Normalisation | ImageNet (mean/std) | Cohérence avec l’entraînement |

---

### CLIP (ViT-L/14)

**Fonctionnement** : modèle vision–langage entraîné sur des paires image–texte. Place images et textes dans le même espace. Les features sont plus sémantiques que ResNet : elles captent mieux le sens que l’aspect purement visuel.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| Modèle | ViT-L-14 (OpenCLIP) | Variante grande taille |
| Dataset pré-entraînement | LAION-2B | Paires image–texte web |
| Dimension sortie | 768 | Par image ou texte |

---

### EfficientNet-B0 (fine-tuning)

**Fonctionnement** : CNN pré-entraîné ImageNet, adapté au domaine e-commerce. On gèle les couches basses et on réentraîne les dernières couches sur nos données produits.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `weights` | ImageNet1K_V1 | Point de départ |
| Stratégie | Gel couches basses, dégel 2 derniers blocs | Adaptation partielle |
| Epochs | ~10 | Entraînement limité (CPU) |

---

## 3. Classificateurs Machine Learning

### Naive Bayes (MultinomialNB)

**Fonctionnement** : modèle probabiliste simple. Suppose que les features sont indépendantes. Rapide, peu de paramètres, souvent correct sur le texte.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `alpha` | 1,0 | Lissage pour éviter les probabilités nulles |

---

### Régression logistique

**Fonctionnement** : modèle linéaire qui associe une combinaison pondérée des features à une probabilité par classe. Robuste, interprétable, adapté aux espaces haute dimension (TF-IDF).

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `max_iter` | 1000 | Nombre max d’itérations |
| `random_state` | 42 | Reproductibilité |
| `class_weight` | balanced ou dict | Pondération des classes déséquilibrées |
| `n_jobs` | -1 | Parallélisation |

---

### SVM linéaire (LinearSVC)

**Fonctionnement** : cherche l’hyperplan qui sépare au mieux les classes en maximisant la marge. Très performant sur le texte vectorisé. On utilise la version linéaire, plus rapide que le noyau RBF sur les grandes matrices creuses.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `max_iter` | 1000 à 3000 | Convergence |
| `dual` | False | Formulation primale (adaptée aux grandes matrices) |
| `C` | 0,1 à 10 (optimisé) | Régularisation : petit C = plus de marge, grand C = meilleur ajustement |
| `class_weight` | balanced | Déséquilibre des classes |

---

### Random Forest

**Fonctionnement** : ensemble d’arbres de décision entraînés sur des sous-échantillons différents. La prédiction est le vote majoritaire. Peu adapté aux milliers de dimensions creuses (TF-IDF), plus adapté aux features images.

**Paramètres projet (baseline)** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `n_estimators` | 100 | Nombre d’arbres |
| `max_depth` | 20 | Profondeur max par arbre |

**Paramètres projet (optimisé)** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `n_estimators` | 200 | Plus d’arbres |
| `max_depth` | 30 | Arbres plus profonds |
| `min_samples_split` | 5 | Min d’échantillons pour couper un nœud |
| `min_samples_leaf` | 2 | Min d’échantillons par feuille |
| `class_weight` | balanced | Déséquilibre |

---

### XGBoost

**Fonctionnement** : gradient boosting. Construit une séquence d’arbres où chacun corrige les erreurs du précédent. Rapide, souvent performant, gère bien le déséquilibre.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `n_estimators` | 100 | Nombre d’arbres (rounds) |
| `max_depth` | 6 | Profondeur par arbre |
| `learning_rate` | 0,1 | Vitesse d’apprentissage |
| `scale_pos_weight` | ratio max/min des classes | Pondération déséquilibre |
| `eval_metric` | mlogloss | Métrique multiclasse |

---

### LightGBM

**Fonctionnement** : gradient boosting par feuille (leaf-wise) au lieu de niveau par niveau. Souvent plus rapide que XGBoost, efficace sur les features structurées ou les embeddings.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `n_estimators` | 100 | Nombre d’arbres |
| `max_depth` | 6 | Profondeur |
| `learning_rate` | 0,1 | Vitesse d’apprentissage |
| `class_weight` | balanced ou dict | Déséquilibre |
| `verbose` | -1 | Pas de logs |

---

### CatBoost

**Fonctionnement** : gradient boosting avec gestion native des variables catégorielles. Robustesse au surapprentissage, moins sensible aux hyperparamètres.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `iterations` | 100 | Nombre d’arbres |
| `depth` | 6 | Profondeur |
| `learning_rate` | 0,1 | Vitesse d’apprentissage |
| `class_weights` | dict | Déséquilibre |
| `thread_count` | -1 | Tous les cœurs CPU |

---

### Gradient Boosting (HistGradientBoostingClassifier)

**Fonctionnement** : gradient boosting de scikit-learn, basé sur des histogrammes. Plus rapide que l’ancien GradientBoostingClassifier, performances proches.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `max_iter` | 100 | Nombre d’arbres |
| `learning_rate` | 0,1 | Vitesse d’apprentissage |
| `max_depth` | 5 | Profondeur |

---

### MLP (réseau de neurones)

**Fonctionnement** : perceptron multicouche. Couches de neurones empilées avec activation ReLU. Utilisé sur les features CLIP (768 dim) pour la classification image.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `hidden_layer_sizes` | (512, 256) | Deux couches cachées |
| `activation` | relu | Fonction d’activation |
| `max_iter` | 300 | Epochs max |
| `early_stopping` | True | Arrêt si plus de progrès sur la validation |
| `learning_rate_init` | 1e-3 | Taux d’apprentissage initial |
| `learning_rate` | adaptive | Ajustement automatique |
| `batch_size` | 256 | Taille des lots |

---

### CNN 1D (texte, notebook 06)

**Fonctionnement** : convolution sur le vecteur TF-IDF traité comme séquence. Les filtres détectent des motifs locaux ; le pooling réduit la dimension. Classificateur dense en sortie.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| Filtres | [64, 128] | Nombre de cartes de features par couche conv |
| Kernel size | 3 | Taille du filtre convolutionnel |
| Dropout | 0,3 | Régularisation |
| Optimizer | Adam | Descente de gradient adaptative |
| Epochs | 15–20 | Itérations d’entraînement |

---

### LSTM (texte, notebook 06)

**Fonctionnement** : couche récurrente qui mémorise des dépendances à longue portée. Adaptée aux séquences (texte). Les unités LSTM traitent l’entrée pas à pas et transmettent un état caché.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| Unités | [64, 32] | Taille des couches LSTM |
| Dropout | 0,3 | Régularisation |
| Optimizer | Adam | Descente de gradient |
| Epochs | 15–20 | Itérations d’entraînement |

---

### VotingClassifier et StackingClassifier (texte, notebook 06)

**VotingClassifier** : plusieurs modèles (LR, SVM, RF, Naive Bayes) votent. La prédiction finale est le vote majoritaire (ou pondéré par probabilités).

**StackingClassifier** : un méta-modèle (régression logistique) apprend à combiner les prédictions de plusieurs estimateurs de base. Utilise une validation croisée interne pour éviter le surapprentissage.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| Estimateurs base | LR, SVM, RF, Naive Bayes | Modèles de premier niveau |
| Méta-estimateur | Logistic Regression | Combinaison des prédictions |
| `cv` (Stacking) | 5 | Folds pour générer les méta-features |

---

## 4. Optimisation et rééquilibrage

### RandomizedSearch (texte)

**Fonctionnement** : teste un nombre fixe de combinaisons de paramètres tirées au hasard. Plus rapide que GridSearch sur de grandes grilles.

**Paramètres projet (SVM texte)** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `C` | [0,1 ; 0,5 ; 1 ; 2 ; 5 ; 10] | Régularisation |
| `max_iter` | [1000 ; 2000 ; 3000] | Convergence |
| `n_iter` | 20 | Nombre de combinaisons testées |
| `cv` | 3 | Folds de validation croisée |

---

### GridSearch (image CLIP)

**Fonctionnement** : teste toutes les combinaisons d’une grille de paramètres. Plus exhaustif mais plus lent.

**Paramètres projet (LR et SVM sur CLIP)** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `C` | [0,01 ; 0,1 ; 0,5 ; 1 ; 5 ; 10] | Régularisation |
| `cv` | 3 | Folds |
| `scoring` | f1_macro | Métrique d’évaluation |

---

### Poids de classes (class_weight)

**Fonctionnement** : donne plus de poids aux classes rares dans la fonction de perte. Le modèle pénalise plus les erreurs sur les classes minoritaires.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `method` | balanced | Poids inversement proportionnel à la fréquence |

---

### SMOTE

**Fonctionnement** : génère des exemples synthétiques pour les classes minoritaires en interpolant entre voisins proches. Rééquilibre le dataset avant entraînement.

**Paramètres projet** :
| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `sampling_strategy` | auto | Rééquilibrage automatique |
| `k_neighbors` | 3 ou 5 | Nombre de voisins pour l’interpolation |
| `random_state` | 42 | Reproductibilité |

---

## Synthèse : où est utilisé chaque modèle ?

| Modèle | Texte | Image | Fusion |
|--------|-------|-------|--------|
| TF-IDF | ✓ | — | ✓ |
| TruncatedSVD | — | — | ✓ |
| ResNet50 | — | ✓ | ✓ |
| CLIP ViT-L/14 | — | ✓ | ✓ |
| EfficientNet-B0 | — | ✓ | — |
| Naive Bayes | ✓ | — | — |
| Logistic Regression | ✓ | ✓ | ✓ |
| SVM (LinearSVC) | ✓ | ✓ | ✓ |
| Random Forest | ✓ | ✓ | — |
| XGBoost | ✓ | ✓ | — |
| LightGBM | ✓ | ✓ | — |
| CatBoost | ✓ | ✓ | — |
| Gradient Boosting | ✓ | ✓ | — |
| MLP | ✓ (TF-IDF) | ✓ (CLIP) | — |
| CNN 1D | ✓ | — | — |
| LSTM | ✓ | — | — |
| Voting / Stacking | ✓ | — | — |
