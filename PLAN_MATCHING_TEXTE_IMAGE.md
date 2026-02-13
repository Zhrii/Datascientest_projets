# Descriptif détaillé : Projet de matching texte-image

**Document de rendu — Justifications techniques et implémentation**

---

## 1. Contexte et objectifs du projet

### 1.1 Contexte métier

Ce projet s’inscrit dans un cadre de **classification de produits e-commerce**. Les données proviennent d’un catalogue contenant environ **84 916 produits d’entraînement** et **13 814 produits de test**, avec pour chaque produit :

- une **désignation** (titre court),
- une **description** (éventuellement avec balises HTML),
- un **identifiant produit** (`productid`) et un **identifiant d’image** (`imageid`),
- un **code type de produit** (`prdtypecode`, 27 classes).

L’objectif global du projet est d’exploiter ces informations (texte et, dans cette partie, image) pour améliorer la catégorisation, la recherche et la cohérence du catalogue.

### 1.2 Objectifs du volet matching texte-image

Le **matching texte-image** vise à apprendre une correspondance entre :

- le **texte produit** (désignation + description),
- et l’**image associée** au même produit.

Une fois ce lien appris, le système doit permettre :

1. **Recherche texte → image** : à partir d’un texte (requête ou fiche produit), retrouver l’image correspondante parmi un ensemble d’images.
2. **Recherche image → texte** : à partir d’une image, retrouver le texte (ou le produit) associé.
3. **Vérification de paires** : juger si un couple (texte, image) correspond ou non au même produit (similarité dans un espace commun).

Ces fonctionnalités sont utiles pour :

- la **recherche multimodale** (texte + image),
- la **détection d’incohérences** (mauvaise image sur une fiche),
- la **recommandation** ou l’enrichissement de fiches à partir de paires (texte, image) bien appariées.

---

## 2. Problématique et contraintes

### 2.1 Données

- **Texte** : multilingue (français, allemand, anglais), parfois court ou bruité (HTML, champs vides).
- **Images** : une image par produit (convention de nommage `image_{imageid}_product_{productid}.jpg`), pas toutes les produits ont une image (merge interne nécessaire).
- **Volume** : ~84k paires possibles après jointure ; entraînement et évaluation doivent rester faisables sur une machine standard (CPU/GPU raisonnable).

### 2.2 Contraintes techniques

- Réutiliser au maximum le **préprocessing texte** déjà en place dans le projet (nettoyage HTML, normalisation).
- Pouvoir faire de la **recherche par similarité** (texte seul ou image seule) sans re-modèle dédié.
- Avoir une solution **réplicable** (code modulaire, dépendances claires) et **documentée** pour un rendu.

---

## 3. Choix de l’architecture : Dual Encoder + apprentissage contrastif

### 3.1 Pourquoi pas d’autres approches ?

- **CLIP (ou autre modèle zero-shot)**  
  Avantage : pas d’entraînement. Inconvénient : modèle généraliste, peu adapté au vocabulaire et aux codes métier du catalogue (e.g. prdtypecode, désignations FR/DE/EN). On souhaite un modèle **adapté au domaine**.

- **Fusion précoce (Early Fusion)**  
  Un seul réseau qui prend (texte, image) en entrée. Inconvénient : on ne peut pas faire de recherche **texte seul** ou **image seule** ; moins flexible en inférence.

- **Fusion tardive (Late Fusion)**  
  Encoders séparés puis combinaison. Souvent utilisée pour une décision unique (ex. classification) et pas pour une recherche asymétrique (requête texte vs. base d’images). Moins naturel pour “recherche par texte” ou “par image”.

### 3.2 Choix retenu : Dual Encoder + contrastive learning

**Principe :**

- **Encodeur texte** : transforme un texte en un vecteur de dimension fixe \(d\) (ex. 256).
- **Encodeur image** : transforme une image en un vecteur de même dimension \(d\).
- Les deux espaces sont **alignés** : une paire (texte du produit \(i\), image du produit \(i\)) doit être proche, les paires (texte \(i\), image \(j\)) pour \(j \neq i\) doivent être plus éloignées.
- L’alignement est appris par une **loss contrastive** (type InfoNCE) sur des batches de paires.

**Intérêts :**

- **Recherche bidirectionnelle** : on peut encoder uniquement du texte ou uniquement des images, puis comparer par similarité (ex. cosinus).
- **Adaptation au catalogue** : le modèle apprend les spécificités du domaine (produits, désignations, styles d’images).
- **Scalabilité** : une fois les embeddings calculés, la recherche se fait par similarité (éventuellement avec un index type FAISS).
- **Réutilisation** : même principe que CLIP, mais entraîné sur nos paires (texte, image) réelles.

---

## 4. Stack technique et justifications des librairies

### 4.1 Framework : PyTorch

- **sentence-transformers** (encodeur de texte) est construit sur **PyTorch** et **Transformers**.
- **torchvision** fournit des modèles d’images pré-entraînés (ex. ResNet50) et des pipelines de chargement simples.
- PyTorch permet d’écrire la **loss contrastive** et la boucle d’entraînement de façon lisible (gradients, device CPU/GPU).
- Le reste du projet peut rester sur TensorFlow pour la classification ; le module matching est autonome en PyTorch.

**Versions utilisées :** `torch>=2.0.0`, `torchvision>=0.15.0` (voir `requirements.txt`).

### 4.2 Encodeur de texte : sentence-transformers

**Librairie :** `sentence-transformers` (repos sur Hugging Face Transformers + PyTorch).

**Modèle :** `paraphrase-multilingual-MiniLM-L12-v2`.

**Justifications :**

- **Multilingue** : le catalogue contient du français, de l’allemand et de l’anglais ; un modèle multilingue évite d’avoir un encoder par langue.
- **Tâche “paraphrase / similarité”** : le modèle est pré-entraîné pour capturer la similarité sémantique (ex. “livre” / “ouvrage” / “tome”), ce qui correspond bien à des désignations ou descriptions de produits.
- **Taille raisonnable** : MiniLM (12 couches) est plus léger qu’un BERT base/large, ce qui permet des batches plus grands et des temps d’entraînement acceptables.
- **Interface simple** : `model.encode(texts)` retourne des vecteurs ; on peut ajouter une couche de **projection** (Linear) pour ramener la dimension native (384) à 256 et l’aligner avec l’encodeur d’images.

**Alternatives écartées :**

- CamemBERT : français uniquement, pas adapté au multilingue.
- BERT multilingue complet : plus lourd, peu de gain attendu pour ce cas.
- Universal Sentence Encoder (TensorFlow) : aurait imposé un second framework et une intégration plus lourde.

**Implémentation dans le projet :**  
La classe `TextEncoder` dans `src/multimodal/encoders.py` encapsule ce modèle, gère la projection optionnelle (384 → `embedding_dim`) et l’appareil (CPU/GPU).

### 4.3 Encodeur d’images : ResNet50 (torchvision)

**Librairie :** `torchvision.models` (PyTorch).

**Modèle :** ResNet50 pré-entraîné sur **ImageNet** (poids `ResNet50_Weights.IMAGENET1K_V1`).

**Justifications :**

- **Transfer learning** : les représentations visuelles (formes, textures, objets) sont réutilisées ; seules les dernières couches (remplacées par une **projection** 2048 → `embedding_dim`) sont entraînées pour l’alignement texte-image.
- **Standard** : architecture bien comprise, bon compromis précision / coût.
- **Sortie fixe** : la couche fully-connected finale est remplacée par `Identity` ; on récupère un vecteur 2048-d, puis une `Linear(2048, embedding_dim)` donne l’embedding final.

**Prétraitement des images :**  
Redimensionnement à 224×224 (attendu par ResNet), normalisation [0,1] en entrée ; la normalisation ImageNet (moyenne/écart-type) peut être appliquée dans le forward (comme dans `ImageEncoder.forward`).

**Implémentation :**  
La classe `ImageEncoder` dans `src/multimodal/encoders.py` contient le backbone ResNet50 (sans la tête de classification) et la couche de projection. En phase 1 d’entraînement, le backbone peut être gelé et seules les projections (texte + image) sont mises à jour.

### 4.4 Autres dépendances

- **Pillow (PIL)** : chargement et redimensionnement des images dans le data loader.
- **pandas, numpy** : chargement des CSV, manipulation des paires (texte, image_path, productid, imageid).
- **scikit-learn** : `train_test_split` stratifié sur `prdtypecode` pour le split train/validation des paires.

Les dépendances sont listées dans `requirements.txt` (notamment `sentence-transformers`, `torch`, `torchvision`, `Pillow`).

---

## 5. Fonction de coût : InfoNCE (loss contrastive)

### 5.1 Formulation

Pour un batch de \(N\) paires \((t_i, v_i)\) (texte \(i\), image \(i\)) :

- On calcule les embeddings \(\mathbf{t}_i\), \(\mathbf{v}_i\) (après normalisation L2).
- La matrice de similarité (logits) est :  
  \(\mathbf{S}_{ij} = \langle \mathbf{t}_i, \mathbf{v}_j \rangle / \tau\),  
  avec \(\tau\) un paramètre de **température**.
- Pour chaque \(i\), la paire positive est \((t_i, v_i)\) ; les autres images du batch jouent le rôle de **négatives**.
- La loss est une **cross-entropy** : les labels sont \(y_i = i\) (la bonne image pour le texte \(i\) est à l’index \(i\) dans le batch).

En pratique :  
`logits = (text_emb @ image_emb.T) / temperature`  
`loss = F.cross_entropy(logits, torch.arange(batch_size))`

### 5.2 Intérêt de cette loss

- **Pas de construction explicite de triplets** : les négatives sont les autres paires du même batch (efficace et stable).
- **Température \(\tau\)** : un \(\tau\) petit (ex. 0,07) rend la distribution plus “peakée” et pousse le modèle à bien séparer la paire positive des négatives.
- **Symétrie** : on peut définir la loss dans les deux sens (texte→image ou image→texte) ; ici une seule direction (texte comme requête, image comme cible) suffit pour l’apprentissage, la similarité restant symétrique en inférence.

### 5.3 Implémentation

La fonction `contrastive_loss` et la méthode `DualEncoderModel.compute_loss` sont dans `src/multimodal/matching_model.py`. Les embeddings sont normalisés avant le produit scalaire ; la température est un argument du modèle (défaut 0,07).

---

## 6. Hyperparamètres et réglages

| Paramètre        | Valeur utilisée | Justification |
|------------------|------------------|----------------|
| **embedding_dim**| 256              | Compromis expressivité / coût mémoire ; cohérent entre texte et image. |
| **temperature**  | 0,07            | Valeur type CLIP ; renforce la séparation des paires. |
| **batch_size**   | 32 (ou 64)      | Plus de négatives par batch améliore le contrastive ; 32 reste gérable en mémoire. |
| **learning_rate**| 1e-4            | Fine-tuning / transfer learning : pas trop agressif pour ne pas dégrader les représentations pré-entraînées. |
| **epochs**       | 5 à 20          | Phase 1 (projections seules) : quelques epochs ; phase 2 (fine-tuning complet) optionnel. |

La **phase 1** consiste à geler le backbone ResNet50 et (si besoin) le modèle sentence-transformers, et à n’entraîner que les couches de projection (texte 384→256, image 2048→256). Cela limite le risque d’overfitting et accélère l’entraînement.

---

## 7. Implémentation réalisée dans le projet

### 7.1 Module `src/multimodal/`

- **`data_loader.py`**
  - **`load_text_image_pairs(data_dir, image_train_dir, preprocess_text=True, root=None)`**  
    Charge `X_train_update.csv` et `Y_train.csv`, construit la table des images à partir des noms de fichiers `image_{imageid}_product_{productid}.jpg`, fait un **merge interne** (seuls les produits avec image sont gardés). Le texte est soit préprocessé via le pipeline du projet (nettoyage HTML, normalisation), soit une concaténation simple désignation + description. Option `root` pour stocker des chemins d’images **relatifs** (portabilité).
  - **`create_pairs_dataset(pairs_df, train_size=0.8, random_state=42)`**  
    Split train/validation **stratifié sur `prdtypecode`** pour garder une répartition des classes proche entre train et val.
  - **`load_image(image_path, size=(224,224), base_dir=None)`**  
    Charge une image (PIL), redimensionne, convertit en numpy normalisé [0,1] (format H, W, 3). Gère les chemins relatifs via `base_dir`.

- **`encoders.py`**
  - **`TextEncoder`** : wrapper autour de `SentenceTransformer` (paraphrase-multilingual-MiniLM-L12-v2), projection optionnelle vers `embedding_dim`, `encode(texts)` en numpy (ou tenseur pour l’entraînement).
  - **`ImageEncoder`** : ResNet50 (sans tête de classification) + `Linear(2048, embedding_dim)`, normalisation ImageNet dans le forward si entrée en [0,1], méthodes `forward` (PyTorch), `encode` (numpy), `eval()` / `train()`.

- **`matching_model.py`**
  - **`contrastive_loss(text_emb, image_emb, temperature)`** : normalisation L2 puis cross-entropy comme décrit ci-dessus.
  - **`DualEncoderModel`** : agrège les deux encodeurs, expose `get_text_embeddings`, `get_image_embeddings`, `compute_loss`, et une méthode `parameters()` pour ne passer que les paramètres à entraîner (projections, éventuellement backbone).

- **`utils.py`**
  - **`compute_similarity(text_emb, image_emb)`** : similarité cosinus entre vecteur(s) texte et vecteurs image.
  - **`find_matching_images(text, text_encoder, image_embeddings, image_paths, top_k)`** : pour un texte, retourne les `top_k` images les plus proches (chemins + scores).
  - **`find_matching_texts(image_emb, text_embeddings, text_list, top_k)`** : recherche inverse (image → textes).
  - **`recall_at_k(query_embeddings, gallery_embeddings, query_ids, gallery_ids, k_values)`** : calcule Recall@K pour des requêtes (ex. textes) et une galerie (ex. images), en utilisant les identifiants (ex. productid) pour savoir si la “vraie” image est dans le top-K.

### 7.2 Notebook d’entraînement et d’évaluation

**`notebooks/07_matching_texte_image.ipynb`** :

1. Configuration des chemins (racine du projet, `data brut`, `data/processed/image_train`) et imports du module `multimodal`.
2. Chargement des paires avec `load_text_image_pairs(..., root=ROOT)` et split train/val avec `create_pairs_dataset`.
3. Création des encodeurs (`TextEncoder`, `ImageEncoder`) et du `DualEncoderModel` (température 0,07), optimiseur Adam (lr=1e-4).
4. Boucle d’entraînement (epochs, batches) : pour chaque batch, chargement des images via `load_image(..., base_dir=ROOT)`, encodage texte (sans gradient) et image (avec gradient sur la projection), calcul de la loss InfoNCE, backward et step. Affichage type “Keras” (epoch, steps, temps, accuracy batch, loss).
5. Évaluation : calcul des embeddings texte et image sur la validation, puis **Recall@K** (K = 1, 5, 10) via `recall_at_k`.
6. Exemple de recherche : `find_matching_images` pour un texte donné sur un sous-ensemble d’images.

Le **workbook** `08_workbook_matching_texte_image.ipynb` sert au suivi des tâches (données, encoders, entraînement, etc.).

### 7.3 Intégration avec le reste du projet

- Le **préprocessing texte** est réutilisé via `PreprocessingPipeline` (option `preprocess_text=True` dans `load_text_image_pairs`), ce qui assure la cohérence avec les autres notebooks (exploration, classification texte).
- Les **chemins d’images** peuvent être stockés en relatif (`root=ROOT`), ce qui rend le notebook et les données plus portables (autre machine, autre chemin).

---

## 8. Métriques d’évaluation

- **Recall@K** (K = 1, 5, 10) : pour chaque requête (ex. un texte), on regarde si l’image “vraie” (même productid) est dans les K images les plus similaires. On rapporte la proportion de requêtes pour lesquelles c’est le cas. C’est la métrique principale pour un usage “recherche”.
- **Accuracy en batch** (pendant l’entraînement) : pour chaque batch, proportion de paires où l’image correcte est en top-1 dans le batch ; indicateur de progression.
- **Mean Reciprocal Rank (MRR)** : possible à dériver à partir des rangs de la bonne image ; non implémenté dans le code actuel mais facile à ajouter à partir de `recall_at_k` ou des scores de similarité.

Valeurs cibles indicatives : Recall@1 > 0,3, Recall@5 > 0,6, Recall@10 > 0,8 (à ajuster selon la difficulté du catalogue).

---

## 9. Structure des fichiers et livrables

```
src/multimodal/
├── __init__.py
├── data_loader.py    # Paires (texte, image), split train/val, load_image
├── encoders.py       # TextEncoder (sentence-transformers), ImageEncoder (ResNet50)
├── matching_model.py # DualEncoderModel, contrastive_loss (InfoNCE)
└── utils.py          # Similarité, find_matching_images/texts, recall_at_k

notebooks/
├── 07_matching_texte_image.ipynb   # Entraînement + évaluation + exemple de recherche
└── 08_workbook_matching_texte_image.ipynb  # Suivi des tâches
```

**Documentation :**  
Ce fichier (`PLAN_MATCHING_TEXTE_IMAGE.md`) constitue le **descriptif détaillé** du volet matching : objectifs, choix d’architecture, justifications des librairies et de la loss, description de l’implémentation et des métriques, pour un rendu complet et reproductible.

---

## 10. Limites et pistes d’amélioration

- **Données** : seuls les produits ayant une image sont utilisés (merge interne) ; les produits sans image ne participent pas à l’entraînement.
- **Mémoire** : encoder toute la base (ex. 84k images) en une fois peut être coûteux ; en production, précalculer et sauvegarder les embeddings, puis utiliser un index de similarité (ex. FAISS) pour la recherche.
- **Temps d’entraînement** : sur CPU, plusieurs heures possibles ; utilisation du GPU recommandée pour accélérer.
- **Fine-tuning** : une phase 2 (dégeler le backbone image, voire fine-tuner le texte) peut améliorer les métriques au prix d’un entraînement plus long et d’un risque d’overfitting ; à tester sur la validation (Recall@K, loss).
- **Multimodalité** : ce module ne fait pas de classification de type de produit (prdtypecode) ; il sert à l’alignement texte-image. L’utilisation des embeddings pour la classification (fusion avec les modèles texte existants) est une extension possible.

---

## 11. Résumé des choix techniques (tableau de synthèse)

| Composant           | Choix retenu                          | Justification principale |
|---------------------|----------------------------------------|---------------------------|
| Architecture        | Dual Encoder + contrastive learning   | Recherche bidirectionnelle, adaptation au catalogue |
| Encodeur texte      | sentence-transformers (MiniLM multilingue) | Multilingue, similarité sémantique, léger |
| Encodeur image      | ResNet50 (torchvision, ImageNet)      | Transfer learning, standard, projection 2048→256 |
| Framework           | PyTorch                               | Compatible sentence-transformers et torchvision, loss custom simple |
| Loss                | InfoNCE (cross-entropy sur similarités) | Négatives = autres paires du batch, stable, température réglable |
| Dimension embedding | 256                                   | Alignement texte/image, compromis mémoire/performance |
| Température         | 0,07                                  | Standard type CLIP, séparation nette des paires |
| Préprocessing texte  | Pipeline projet (HTML, normalisation) ou concat simple | Cohérence avec le reste du projet |
| Métriques           | Recall@K (1, 5, 10)                   | Adaptées à la recherche par similarité |

---

*Document à jour par rapport à l’implémentation dans `src/multimodal/` et les notebooks 07 et 08.*
