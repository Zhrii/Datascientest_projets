# Pitch Technique — 12 minutes

> **Durée cible** : 12 minutes (~160 mots/min) | **Ordre** : notebooks 01 à 12 | **Ton** : technique, factuel

---

## 1. Exploration des données texte (Notebook 01)

Le projet s’appuie sur les données produits Rakuten : environ quatre-vingt-cinq mille exemples en entraînement et treize mille en test. Chaque produit est décrit par une désignation courte, une description optionnelle, un `productid` et un `imageid`. La variable cible est le `prdtypecode`, soit vingt-sept classes de catégories produits.

L’exploration révèle la structure du jeu de données et ses limites. Le déséquilibre des classes est marqué : la classe Piscine et Spa représente environ dix mille exemples, alors que les plus rares — DLC, Accessoires Animaux — en comptent moins de cent. Le rapport entre la classe la plus fréquente et la plus rare dépasse quarante. Environ trente-cinq pour cent des descriptions sont manquantes ; les désignations, elles, sont toujours présentes. Cette disparité impactera le choix des métriques : on privilégiera le F1-macro. Les variables textuelles présentent une forte hétérogénéité : longueurs très variables, présence de balises HTML et mélange de langues.
---

## 2. Traitement des données texte (Notebook 02)

Le préprocessing s’organise en cinq étapes. 
Première étape : nettoyage HTML avec BeautifulSoup et lxml — détection des balises, extraction du contenu, décodage des entités, suppression des résidus. 
Deuxième étape : gestion des valeurs manquantes. Les descriptions vides sont remplacées par une chaîne vide ou un placeholder pour conserver les désignations. 
Troisième étape : normalisation — minuscules, suppression des caractères spéciaux superflus, uniformisation des espaces. 
Quatrième étape : feature engineering — indicateurs de longueur (nombre de mots en désignation et description), features binaires (présence de chiffres, de majuscules), indicateurs de qualité. 
Cinquième étape : fusion désignation plus description en une colonne `text_combined`, puis sauvegarde dans `data/processed`. Une variante crée une superclasse Publications en regroupant quatre classes, réduisant le nombre de classes à vingt-quatre.

---

## 3. Validation du traitement (Notebook 03)

Ce notebook valide le préprocessing. On compare les statistiques avant et après : longueur moyenne des textes, taux de remplissage, distribution des classes. Des visualisations contrôlent que le nettoyage n’a pas introduit de biais. La validation finale confirme que les fichiers `X_train_clean`, `X_test_clean` et `y_train` sont cohérents et prêts pour la modélisation.

---

## 4. Modélisation baseline texte (Notebook 04)

Les données nettoyées sont vectorisées avec TF-IDF : `max_features` à dix mille, `ngram_range` de un à deux pour capturer unigrammes et bigrammes, `min_df` à deux pour éviter les termes trop rares, `max_df` à zéro virgule quatre-vingt-quinze pour exclure les mots génériques. Les labels sont encodés avec un LabelEncoder. Le split est stratifié : quatre-vingt pour cent entraînement, vingt pour cent validation, graine fixe pour reproductibilité.

Quatre modèles baselines sont entraînés : Naive Bayes, régression logistique, SVM linéaire et Random Forest. La métrique principale est le F1-macro, adaptée au déséquilibre des classes. Le SVM linéaire obtient le meilleur résultat : environ zéro virgule soixante-dix-huit en F1-macro sur vingt-sept classes, suivi de près par la régression logistique. Le Random Forest sous-performe, peu adapté aux espaces TF-IDF de haute dimension. On visualise les performances, la matrice de confusion et la distribution des classes réel versus prédit. Une validation croisée à cinq folds confirme la stabilité. L’analyse des erreurs identifie les classes les plus confondues — publications, jouets, mobilier. On sauvegarde le vectoriseur TF-IDF, le label encoder et le meilleur modèle pour les notebooks suivants.

---

## 5. Optimisation des modèles texte (Notebook 05)

On repart du SVM linéaire. L’analyse par classe met en évidence les top et bottom F1. On teste l’optimisation des hyperparamètres avec RandomizedSearch sur le paramètre C du SVM. Pour le déséquilibre, on applique les poids de classes — `compute_class_weight` — et SMOTE. On compare également Random Forest optimisé, Gradient Boosting, régression logistique pondérée, XGBoost, LightGBM et CatBoost sur les mêmes features TF-IDF. Les poids de classes ne permettent pas d’améliorer le F1-macro ; LightGBM, XGBoost et la régression logistique pondérée rivalisent avec le SVM ; les poids de classes n'améliorent pas systématiquement. On sauvegarde le modèle final ainsi qu’une version sur vingt-quatre classes avec la superclasse Publications, qui atteint environ zéro virgule quatre-vingt-un en F1-macro.

---

## 6. Modèles avancés et interprétabilité texte (Notebook 06)

On teste des ensembles : VotingClassifier et StackingClassifier combinant plusieurs modèles. On explore le deep learning : MLP, CNN et LSTM sur les vecteurs TF-IDF avec TensorFlow et Keras. Pour l’interprétabilité, on utilise SHAP — résumés globaux, waterfall plots — et les feature importances classiques. Les ensembles et le deep learning n’apportent pas de gain systématique par rapport au SVM optimisé sur ce jeu. On conclut sur le modèle final retenu pour la partie texte et on documente les choix et limites. Le texte seul sert de référence pour la fusion multimodale.

---

## 7. Exploration des données images (Notebook 07)

On inventorie les images selon la convention de nommage : `image_{imageid}_product_{productid}.jpg`, présentes dans `data brut` ou `data/processed/image_clean`. Environ quarante-cinq mille produits ont une image exploitable, soit un peu plus de la moitié du train ; les autres nécessiteront un fallback texte en production. On analyse la distribution des classes par image et les caractéristiques techniques : tailles, formats jpg ou png. On détecte les fichiers corrompus ou invalides pour les exclure. Des échantillons visuels par classe permettent de vérifier la cohérence. La synthèse oriente les choix du traitement et de la modélisation image.

---

## 8. Traitement des données images (Notebook 08)

On valide et filtre les images : exclusion des fichiers corrompus ou inexploitables. On définit les paramètres de prétraitement. Le redimensionnement à deux cent vingt-quatre par deux cent vingt-quatre pixels correspond à la taille d'entrée attendue par ResNet50, CLIP et les modèles pré-entraînés ImageNet. La normalisation selon les moyennes et écarts-types ImageNet est nécessaire : ces modèles ont été entraînés avec ces statistiques, et appliquer la même normalisation garantit des features cohérentes. On construit le dataset final avec chemins d’images et labels alignés. On sauvegarde les métadonnées et statistiques pour les notebooks de modélisation et la fusion.

---

## 9. Modélisation baseline image (Notebook 09)

On charge les produits ayant une image et leurs labels — environ quarante-cinq mille. ResNet50 sert d'extracteur de features, sans entraînement : on passe chaque image dans le réseau pré-entraîné ImageNet et on récupère la sortie de la couche avant la classification — un vecteur de deux mille quarante-huit dimensions qui encode des représentations visuelles de haut niveau. On entraîne ensuite des classificateurs classiques sur ces vecteurs — régression logistique, SVM, Random Forest — pour prédire le prdtypecode. Split stratifié 80-20. La baseline atteint environ zéro virgule cinquante-deux en F1-macro, nettement en dessous du texte seul : l'image seule discrimine moins bien certaines catégories, comme les publications ou le DLC dont les visuels sont peu distinctifs. On visualise la matrice de confusion et on sauvegarde l’extracteur, les features en cache et le meilleur modèle pour la suite.

---

## 10. Optimisation des modèles image (Notebook 10)

On repart des features ResNet50 en cache et du modèle baseline du notebook neuf. Pour lutter contre le déséquilibre des classes — ratio quarante fois entre la plus fréquente et la plus rare — on calcule des poids de classes avec la méthode balanced et on les passe aux modèles. On teste Random Forest optimisé, Gradient Boosting, régression logistique pondérée, XGBoost, LightGBM et CatBoost sur les mêmes features. LightGBM obtient le meilleur score : environ zéro virgule cinquante-deux en F1-macro, devant XGBoost et la baseline. Le gain reste faible — moins d'un point par rapport à la baseline — car les features ResNet50, conçues pour ImageNet, ne capturent pas assez bien les spécificités du domaine e-commerce. C'est ce qui motive l'exploration de CLIP et du fine-tuning dans le notebook onze.

---

## 11. Modélisation avancée image (Notebook 11)

On explore CLIP et le fine-tuning. CLIP est un modèle vision-langage entraîné sur des paires image-texte. On a testé ViT-B/32 mais il n'atteignait qu'environ soixante pour cent en F1-macro. On retient donc CLIP ViT-L/14, pré-entraîné sur LAION-2B, qui extrait des features de sept cent soixante-huit dimensions — plus riches et sémantiques que ResNet50. On applique GridSearch sur SVM, régression logistique et MLP pour la classification. CLIP ViT-L/14 plus classifieur optimisé atteint environ zéro virgule soixante-douze en F1-macro, soit un gain d'une vingtaine de points par rapport à ResNet50. On teste aussi le fine-tuning d’EfficientNet-B0 sur le domaine e-commerce : gel des couches basses, dégel des derniers blocs, entraînement sur quelques epochs. EfficientNet atteint environ zéro virgule cinquante-cinq. La comparaison finale montre que CLIP améliore nettement la baseline ResNet50. Les classes difficiles restent les publications, le DLC et le textile. On sauvegarde le meilleur modèle image pour la fusion.

---

## 12. Fusion multimodale (Notebook 12)

**Objectif** : combiner texte et image pour améliorer la classification. On aligne les données via un merge sur `productid` et `imageid`. Seuls les produits ayant les deux modalités sont conservés — environ quarante-cinq mille paires. Un bug d’alignement des labels a été identifié et corrigé : le merge réinitialisait les indices, provoquant un désalignement entre paires et labels ; les scores F1 chutaient à zéro virgule zéro deux, pire qu'aléatoire. La correction consiste à joindre les labels avant le merge pour éviter ce décrochage. On teste ensuite trois approches.

**Approche 1 — Late fusion** : on concatène les vecteurs. Le texte est vectorisé en TF-IDF puis réduit à trois cents dimensions avec TruncatedSVD : le TF-IDF brut dépasse dix mille dimensions, ce qui dominerait le vecteur fusionné, alourdirait l'entraînement et risque le surapprentissage ; trois cents dimensions suffisent à capturer l'essentiel du signal textuel. L'image apporte les features ResNet50 ou CLIP. On colle les deux et on entraîne un classificateur. On teste aussi l'ajout du texte encodé par CLIP pour enrichir le signal sémantique.

**Approche 2 — CLIP multimodal** : encodage du texte et de l’image dans le même espace CLIP, combinaison par concaténation ou moyenne, puis classification. L'alignement sémantique est déjà intégré dans CLIP.

**Approche 3 — Matching** : recherche d’images par requête texte et inversement via similarité cosinus. Inversement : une image permet de retrouver les produits similaires — recherche visuelle. Pour une requête texte type « canapé gris », on encode la requête avec CLIP, on calcule la similarité cosinus avec chaque image du catalogue, on renvoie les K images les plus proches. Le Recall@K mesure : pour une requête donnée, dans quel pourcentage de cas la paire correcte apparaît dans le top K. Par exemple Recall@10 égal soixante-neuf pour cent signifie qu'environ trois fois sur dix le bon produit n'est pas dans le top 10.

**Résultats** : la meilleure fusion combine TF-IDF, CLIP image et CLIP texte avec un SVM. Elle atteint zéro virgule quatre-vingt-quatre en F1-macro, soit un gain d’environ cinq points par rapport au texte seul sur les produits avec image. Le texte seul y plafonne à zéro virgule soixante-dix-neuf. Recall@1 est d’environ quarante pour cent — le bon produit est en tête quatre fois sur dix. Recall@10 à soixante-neuf : le bon produit est dans le top 10 dans sept cas sur dix. En production, près de la moitié des produits n'a pas d'image. On prévoit un fallback texte-seul, par exemple le SVM sur TF-IDF, pour les classifier.

---

## 13. Industrialisation MLOps (après les notebooks)

Une fois le modèle champion identifié — fusion F1-macro 0,84 — on l’industrialise selon le schéma d’architecture ZenML.

**Inférence.** FastAPI sur le port 8000 expose `/predict`, `/health`, `/model/info`, `/retrain` et `/metrics`. Streamlit devient un client de cette API : la démo appelle le même service que la “prod”. Le modèle texte est chargé depuis le Model Registry de MLflow, version marquée « production ».

**Pipeline.** Un pipeline ZenML en trois étapes — `preprocess_data`, `train_model`, `promote_model` — lancé par `make train`, par l’API ou automatiquement par un scheduler quand le modèle a plus de trente jours ou qu’il y a dix pour cent de données en plus. Le nouveau modèle n’entre en production que s’il est au moins aussi bon que l’actuel ; l’ancien est archivé. MLflow trace chaque entraînement, MinIO stocke les artefacts, MySQL les métadonnées de ZenML.

**Sécurité et qualité.** L’API demande un login et un mot de passe, et limite chaque utilisateur à trente prédictions par minute. Trente tests pytest vérifient l’API, la sécurité, la promotion et le scheduler, à chaque push sur GitHub.

**Monitoring.** Prometheus collecte les métriques de l’API — volume, latence, erreurs, refus de sécurité, F1 et âge du modèle — et évalue huit règles d’alerte ; Grafana les affiche.

**Valeur.** On ne refait pas la science dans Docker : on montre qu’un modèle labo devient un service déployable, sécurisé, observable et réentraînable.

Commande de démarrage : `.\make.ps1 up`. Détails dans `ARCHITECTURE.md` et `rendus/resume_mlops.md`.

---

## Synthèse des performances

| Modalité | Meilleur modèle | F1-macro |
|----------|-----------------|----------|
| Texte seul | TF-IDF + SVM | ~0,78 |
| Image seule | CLIP + SVM | ~0,66 |
| Fusion | TF-IDF + CLIP (img+txt) + SVM | **~0,84** |

La fusion multimodale exploite la complémentarité texte-image et atteint les meilleures performances sur le périmètre des produits avec les deux modalités.

---

**Notes pour la lecture** : débit cible ~160 mots/minute ; ordre strict notebooks 01 à 12 ; ton technique et factuel, prêt à être lu à voix haute.

---

**Note — Signification du « + » dans la fusion** : TF-IDF + CLIP (img+txt) + SVM signifie que l’on **concatène** les vecteurs (on les colle bout à bout), puis on entraîne un SVM sur ce vecteur combiné. TF-IDF donne 300 dimensions, CLIP image 512 (ou 768), CLIP texte 512 (ou 768) — le vecteur fusionné fait environ 1 300 à 1 800 dimensions. TF-IDF et CLIP texte sont complémentaires : le premier repose sur les mots, le second sur le sens. Le SVM apprend à classifier les 27 classes à partir de ces features combinées.
