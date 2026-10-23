# Script Oral — Partie 2 : Défis & Preprocessing — 6 minutes

> **Durée cible** : 6 minutes (~160 mots/min) | **Intervenant** : Tristan | **Ton** : technique, pédagogue, factuel

---

## Introduction (≈45 sec)

L’idée que je veux vous transmettre est simple : on n’a pas jeté des données sales dans un algorithme. On a analysé, compris et soigné les biais complexes du catalogue Rakuten. Le dataset présente quatre défis majeurs que nous avons identifiés et résolus méthodiquement. Je vais vous les présenter un par un, avec les chiffres et les solutions concrètes.

---

## Défi 1 : Le Grand Écart — Long Tail (≈1 min)

**Le problème.** La structure du catalogue suit une longue traîne. La classe dominante — Piscine et Spa — compte environ 10209 exemples. La plus rare — Warhammer — en compte 764. Le rapport dépasse 13 pour 1. Avec une métrique naïve comme l’accuracy, un modèle pourrait tricher : il suffirait de prédire systématiquement Piscine pour obtenir un score élevé, au détriment total des classes minoritaires.

**La solution.** Passage au F1-Score macro comme métrique principale. Cette métrique traite chaque classe de façon égale, quelle que soit sa fréquence. Et on applique des poids de classes équilibrés — `class_weight='balanced'` — sur les SVM, régression logistique, Random Forest et les classificateurs de la fusion — pour que le modèle accorde plus d’importance aux exemples rares pendant l’entraînement. Résultat : on évalue et on optimise réellement la performance sur toutes les vingt-sept catégories.

---

## Défi 2 : Les Données Manquantes — NMAR (≈1 min 10)

**Le problème.** Environ 35,1 % des descriptions sont absentes. Mais ce n’est pas du hasard : c’est du NMAR, Missing Not At Random. Le manque dépend de la catégorie. Les Livres ont quasi 97,4% de descriptions vides. Le Mobilier, lui, en a presque zéro. Si on s’appuie trop sur la description seule, le modèle échoue sur environ un tiers du catalogue.

**La solution.** Création d’une feature hybride : `text_combined`, qui concatène la désignation — toujours présente — et la description. Quand la description est vide, on garde uniquement la désignation. On ne se retrouve jamais avec du texte vide. Chaque produit a donc au minimum une information textuelle exploitable, ce qui stabilise la vectorisation TF-IDF et évite les erreurs sur un tiers du catalogue.

---

## Défi 3 : La Pollution HTML (≈50 sec)

**Le problème.** 18,4%  des textes sont parasités par du code web : balises `br`, `li`, `ul` et autres. Si on ne fait rien, le modèle risque d’apprendre des corrélations fallacieuses — par exemple, la présence de certaines balises serait associée à une catégorie. Il classifierait selon la mise en page, pas selon le contenu sémantique.

**La solution.** Nettoyage chirurgical avec BeautifulSoup. Détection des balises, extraction du contenu textuel, décodage des entités HTML, suppression des résidus. Le taux de réussite du nettoyage dépasse 99.98%. Les textes sont ensuite normalisés : minuscules, suppression de la ponctuation superflue, accents conservés, espaces uniformisés. Le modèle ne voit plus que des mots, pas du HTML.

---

## Défi 4 : L’Hétérogénéité (≈1 min)

**Le problème.** Les textes vont de onze à plus de douze mille caractères, et les titres sont multilingues. Une fois vectorisé en TF-IDF — jusqu’à dix mille dimensions — le jeu devient très creux, coûteux à traiter et risqué pour le surapprentissage.

**La solution.** Normalisation stricte des textes, puis réduction avec TruncatedSVD : on projette le TF-IDF sur trois cents dimensions. On conserve l’essentiel du signal sémantique tout en stabilisant les matrices et en accélérant l’entraînement.

---

## Les Trois Étapes de Résolution — Vue d’ensemble (≈1 min)

**Première étape : le preprocessing.** BeautifulSoup pour le nettoyage HTML, NLP standard pour la normalisation, création d’environ quinze variables d’ingénierie, vectorisation TF-IDF réduite à trois cents dimensions. Les données sont prêtes pour la modélisation.

**Deuxième étape : les modèles unimodaux.** Le texte seul — SVM linéaire sur TF-IDF — atteint environ 0.78 en F1-macro. L’image seule — CLIP ViT-B/32 — atteint environ 0.70. Une difficulté : près de la moitié des produits n’ont pas d’image exploitable. Il faudra prévoir un fallback texte en production.

**Troisième étape : la fusion multimodale.** On concatène les vecteurs — TF-IDF réduit, CLIP image et CLIP texte — puis on entraîne un SVM linéaire sur ce vecteur fusionné. Le Recall@10 atteint 79.1%. Le F1-macro monte à 0.84 : c’est le modèle recommandé à Rakuten.

---

## Transition vers Nicolas (≈20 sec)

La donnée est maintenant propre, et nous avons vu comment les défis ont été traités. Pour la suite : comment avons-nous modélisé tout cela, et à quoi ressemble l’application finale — y compris la couche MLOps FastAPI, monitoring Grafana et pipeline ZenML ? Nicolas va vous présenter nos choix techniques et la démo.

---

**Notes pour la lecture** : débit cible ~160 mots/minute ; articuler clairement les chiffres ; marquer une pause avant chaque « solution » pour renforcer la structure problème-solution ; s’appuyer sur la diapo pour les chiffres-clés.

**Comptage approximatif** : ~950 mots pour ~6 minutes à 160 mots/min.
