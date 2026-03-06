# Référentiel des Classes Produits (prdtypecode)

> **À l'attention des rédacteurs du rapport final** : ce document décrit les 27 catégories de produits du catalogue. Il sert de référence pour interpréter les résultats de classification (texte, image, fusion) et comprendre les confusions entre classes.

---

## Vue d'ensemble

Le dataset Rakuten contient **84 916 produits** répartis en **27 classes** identifiées par un code numérique (`prdtypecode`). Ces 27 classes peuvent être regroupées en **24 classes** via une superclasse qui fusionne les 4 catégories de publications.

**Utilisation dans le projet** : chaque modèle (texte, image, fusion) prédit ce code à partir des données produit. La qualité de la prédiction est évaluée en comparant la classe prédite à la classe réelle (ground truth).

---

## Les 27 classes détaillées

*Légende des colonnes :*
- **Code** : identifiant numérique de la classe (`prdtypecode`).
- **Catégorie** : nom métier de la classe.
- **Effectif (texte)** : nombre de produits avec texte dans cette classe.
- **Effectif (images nettoyées)** : nombre de produits avec une image valide dans cette classe (certaines images sont exclues lors du nettoyage).
- **% du dataset** : proportion de la classe par rapport au total (84 916 produits texte).

| Code | Catégorie | Effectif (texte) | Effectif (images nettoyées) | % du dataset |
|------|-----------|------------------:|----------------------------:|-------------:|
| 10 | Livres / Publications | 3 116 | 1 917 | 3,7 % |
| 40 | Jeux Vidéo / Films (Imports) | 2 508 | 1 939 | 3,0 % |
| 50 | Accessoires Gaming / Consoles | 1 681 | 785 | 2,0 % |
| 60 | Consoles de Jeux Portables / Rétro | 832 | 667 | 1,0 % |
| 1140 | Figurines / Collectibles | 2 671 | 1 297 | 3,1 % |
| 1160 | Cartes à Collectionner | 3 953 | 2 057 | 4,7 % |
| 1180 | Jeux de Figurines / Warhammer | 764 | 346 | 0,9 % |
| 1280 | Jouets pour Enfants | 4 870 | 3 058 | 5,7 % |
| 1281 | Jeux Éducatifs / Jeux de Société Enfants | 2 070 | 1 086 | 2,4 % |
| 1300 | Drones / Maquettes / Modélisme | 5 045 | 3 732 | 5,9 % |
| 1301 | Accessoires Bébé / Puériculture | 807 | 526 | 1,0 % |
| 1302 | Jouets Extérieurs / Sports / Pêche | 2 491 | 1 730 | 2,9 % |
| 1320 | Puériculture / Bébé | 3 241 | 1 329 | 3,8 % |
| 1560 | Mobilier / Décoration Intérieure | 5 073 | 3 219 | 6,0 % |
| 1920 | Textile Maison / Literie | 4 303 | 3 425 | 5,1 % |
| 1940 | Alimentation / Boissons | 803 | 325 | 0,9 % |
| 2060 | Décoration / Éclairage / Bricolage | 4 993 | 3 831 | 5,9 % |
| 2220 | Accessoires Animaux | 824 | 512 | 1,0 % |
| 2280 | Presse / Revues / Magazines | 4 760 | 1 714 | 5,6 % |
| 2403 | Livres / BD / Partitions | 4 774 | 1 534 | 5,6 % |
| 2462 | Consoles de Jeux / Lots | 1 421 | 373 | 1,7 % |
| 2522 | Fournitures Bureau / Papeterie | 4 989 | 2 380 | 5,9 % |
| 2582 | Jardin / Extérieur | 2 589 | 1 591 | 3,0 % |
| 2583 | Piscine / Spa | 10 209 | 3 410 | 12,0 % |
| 2585 | Outils Jardin / Bricolage Extérieur | 2 496 | 1 349 | 2,9 % |
| 2705 | Livres / Romans / Littérature | 2 761 | 747 | 3,3 % |
| 2905 | Jeux Vidéo Téléchargeables / DLC | 872 | 90 | 1,0 % |

**Total** : 84 916 produits (texte) / 44 969 images nettoyées

---

## Regroupement en 24 classes (superclasse)

Les 4 classes de publications sont fusionnées en une superclasse unique :

| Classes fusionnées | Code superclasse |
|---|---|
| 10 (Livres / Publications) | **9999** |
| 2280 (Presse / Revues / Magazines) | **9999** |
| 2403 (Livres / BD / Partitions) | **9999** |
| 2705 (Livres / Romans / Littérature) | **9999** |

La superclasse **9999 — Publications** totalise **15 411 produits** (18,1 % du dataset).

Les 23 autres classes restent inchangées.

---

## Description détaillée par classe

### Classe 10 — Livres / Publications
- **Contenu** : Livres généraux, publications variées, ouvrages en plusieurs langues
- **Exemples** : « Olivia: Personalisiertes Notizbuch / 150 Seiten », ouvrages académiques, romans courts
- **Particularités** : Descriptions souvent absentes (~89 %), titres multilingues (allemand, anglais, français)

### Classe 40 — Jeux Vidéo / Films (Imports)
- **Contenu** : Jeux vidéo importés, films en version étrangère, DVDs/Blu-rays
- **Exemples** : Jeux PS4/Xbox en version import, films en VO
- **Particularités** : Titres souvent en anglais

### Classe 50 — Accessoires Gaming / Consoles
- **Contenu** : Manettes, casques gaming, câbles, accessoires pour consoles
- **Exemples** : Manettes PS4, câbles HDMI, supports de console
- **Particularités** : Classe minoritaire (1 681 produits)

### Classe 60 — Consoles de Jeux Portables / Rétro
- **Contenu** : Consoles portables, consoles rétro, émulateurs
- **Exemples** : Game Boy, consoles rétro mini, PSP
- **Particularités** : Très petite classe (832 produits)

### Classe 1140 — Figurines / Collectibles
- **Contenu** : Figurines d'action, statues, objets de collection pop culture
- **Exemples** : Figurines Funko Pop, figurines anime, statues de collection
- **Particularités** : Confusion possible avec 1180 (Warhammer)

### Classe 1160 — Cartes à Collectionner
- **Contenu** : Cartes Pokémon, Magic, Yu-Gi-Oh, cartes sportives
- **Exemples** : Boosters, cartes rares, lots de cartes
- **Particularités** : Descriptions très souvent absentes (~91 %), images petites et difficiles à distinguer

### Classe 1180 — Jeux de Figurines / Warhammer
- **Contenu** : Figurines Warhammer, peintures de figurines, accessoires modélisme
- **Exemples** : Boîtes Warhammer 40K, sets de peinture Citadel
- **Particularités** : Plus petite classe (764 produits), très spécialisée

### Classe 1280 — Jouets pour Enfants
- **Contenu** : Jouets classiques, poupées, peluches, véhicules miniatures
- **Exemples** : Playmobil, LEGO, Barbie, Hot Wheels
- **Particularités** : Classe volumineuse, confusion possible avec 1281

### Classe 1281 — Jeux Éducatifs / Jeux de Société Enfants
- **Contenu** : Jeux de société, puzzles, jeux éducatifs
- **Exemples** : Monopoly Junior, puzzles enfants, jeux d'apprentissage
- **Particularités** : Frontière floue avec 1280 (jouets)

### Classe 1300 — Drones / Maquettes / Modélisme
- **Contenu** : Drones, maquettes, modèles réduits, RC, pièces détachées
- **Exemples** : Drones DJI, maquettes Revell, voitures RC
- **Particularités** : Grande classe (5 045), vocabulaire technique spécifique

### Classe 1301 — Accessoires Bébé / Puériculture
- **Contenu** : Biberons, tétines, accessoires repas bébé
- **Exemples** : Biberons Avent, anneaux de dentition
- **Particularités** : Petite classe (807), confusion possible avec 1320

### Classe 1302 — Jouets Extérieurs / Sports / Pêche
- **Contenu** : Équipements sportifs, matériel de pêche, jeux d'extérieur
- **Exemples** : Cannes à pêche, trampolines, ballons
- **Particularités** : Catégorie hétérogène (sports + pêche + extérieur)

### Classe 1320 — Puériculture / Bébé
- **Contenu** : Poussettes, sièges auto, vêtements bébé, mobilier bébé
- **Exemples** : Poussettes Chicco, gigoteuses, lits parapluie
- **Particularités** : Confusion possible avec 1301 (accessoires bébé)

### Classe 1560 — Mobilier / Décoration Intérieure
- **Contenu** : Meubles, étagères, rangements, objets déco
- **Exemples** : Tables basses, étagères murales, cadres photo
- **Particularités** : HTML très fréquent dans les descriptions (~96 %), confusion avec 2060

### Classe 1920 — Textile Maison / Literie
- **Contenu** : Draps, couettes, coussins, rideaux, serviettes
- **Exemples** : Parures de lit, plaids, nappes
- **Particularités** : Images souvent très similaires (textiles blancs sur fond blanc)

### Classe 1940 — Alimentation / Boissons
- **Contenu** : Produits alimentaires, boissons, confiseries, compléments
- **Exemples** : Thé, café, bonbons, épices
- **Particularités** : Très petite classe (803 produits), atypique dans un marketplace e-commerce

### Classe 2060 — Décoration / Éclairage / Bricolage
- **Contenu** : Luminaires, ampoules, outils, quincaillerie, déco murale
- **Exemples** : Lampes LED, appliques murales, vis et boulons
- **Particularités** : HTML très fréquent (~93 %), confusion avec 1560 (mobilier)

### Classe 2220 — Accessoires Animaux
- **Contenu** : Gamelles, jouets pour animaux, niches, litière
- **Exemples** : Croquettes, colliers, arbres à chat
- **Particularités** : Petite classe (824 produits), bien distincte thématiquement

### Classe 2280 — Presse / Revues / Magazines
- **Contenu** : Journaux, magazines, revues spécialisées, numéros anciens
- **Exemples** : « Journal Des Arts N° 133 », magazines de collection
- **Particularités** : Descriptions quasi absentes (~93 %), confusion avec les autres classes publications

### Classe 2403 — Livres / BD / Partitions
- **Contenu** : Bandes dessinées, mangas, partitions musicales, albums illustrés
- **Exemples** : Astérix, One Piece, partitions piano
- **Particularités** : Descriptions absentes (~97 %), confusion avec 10 et 2705

### Classe 2462 — Consoles de Jeux / Lots
- **Contenu** : Lots de jeux vidéo, packs consoles, bundles
- **Exemples** : Pack PS4 + jeux, lots de jeux Nintendo
- **Particularités** : Classe minoritaire (1 421 produits)

### Classe 2522 — Fournitures Bureau / Papeterie
- **Contenu** : Stylos, cahiers, classeurs, fournitures scolaires, imprimantes
- **Exemples** : Cartouches d'encre, agendas, trombones
- **Particularités** : Grande classe (4 989), vocabulaire distinctif

### Classe 2582 — Jardin / Extérieur
- **Contenu** : Mobilier de jardin, plantes, serres, décoration extérieure
- **Exemples** : Salons de jardin, pots de fleurs, barbecues
- **Particularités** : Confusion possible avec 2583 et 2585

### Classe 2583 — Piscine / Spa
- **Contenu** : Piscines, accessoires piscine, spas, robots nettoyeurs
- **Exemples** : Piscines hors-sol Intex, pompes, produits chimiques
- **Particularités** : **Plus grande classe** (10 209 produits, 12 %), déséquilibre majeur

### Classe 2585 — Outils Jardin / Bricolage Extérieur
- **Contenu** : Tondeuses, taille-haies, arrosage, clôtures
- **Exemples** : Tondeuses Bosch, tuyaux d'arrosage, grillages
- **Particularités** : Confusion possible avec 2582 (jardin) et 2060 (bricolage)

### Classe 2705 — Livres / Romans / Littérature
- **Contenu** : Romans, littérature française et étrangère, essais
- **Exemples** : Romans de poche, classiques littéraires
- **Particularités** : Confusion forte avec 10 et 2403

### Classe 2905 — Jeux Vidéo Téléchargeables / DLC
- **Contenu** : Codes de téléchargement, DLC, cartes prépayées
- **Exemples** : Cartes PSN, codes Xbox Live, DLC Fortnite
- **Particularités** : Seulement 90 images nettoyées (perte massive au nettoyage car images souvent juste du texte/logo sur fond blanc)

---

## Déséquilibre des classes

### Classes les plus représentées (texte)
1. **2583** Piscine / Spa — 10 209 (12,0 %)
2. **1560** Mobilier — 5 073 (6,0 %)
3. **1300** Drones / Modélisme — 5 045 (5,9 %)
4. **2060** Décoration / Bricolage — 4 993 (5,9 %)
5. **2522** Fournitures Bureau — 4 989 (5,9 %)

### Classes les moins représentées (texte)
1. **1180** Warhammer — 764 (0,9 %)
2. **1940** Alimentation — 803 (0,9 %)
3. **1301** Accessoires Bébé — 807 (1,0 %)
4. **2220** Accessoires Animaux — 824 (1,0 %)
5. **60** Consoles Rétro — 832 (1,0 %)

**Ratio max/min** : 13,4× (2583 vs 1180)

---

## Confusions fréquentes entre classes

Ces groupes de classes sont régulièrement confondus par les modèles (texte et image). L’identification de ces confusions aide à interpréter les erreurs et à prioriser les améliorations.

| Groupe de confusion | Classes concernées | Raison | Impact sur les modèles |
|---|---|---|---|
| **Publications** | 10, 2280, 2403, 2705 | Toutes des publications écrites, descriptions souvent absentes | Texte : peu de mots discriminants ; Image : couvertures similaires |
| **Bébé / Puériculture** | 1301, 1320 | Accessoires vs mobilier bébé, vocabulaire proche | Les deux classes partagent des termes courants (bébé, puériculture) |
| **Jouets** | 1280, 1281 | Jouets classiques vs jeux éducatifs, frontière floue | Distinction subjective (puzzle = jouet ou jeu éducatif ?) |
| **Maison / Déco** | 1560, 2060 | Mobilier vs décoration/bricolage, produits similaires | Texte : HTML fréquent ; Image : meubles et luminaires visuellement proches |
| **Jardin** | 2582, 2583, 2585 | Jardin / piscine / outils extérieurs, univers commun | Produits souvent vendus ensemble, contextes similaires |
| **Gaming** | 40, 50, 60, 2462, 2905 | Jeux, consoles, accessoires, DLC — univers gaming | Vocabulaire commun (PS4, Xbox, jeu, etc.) |
| **Figurines** | 1140, 1180 | Figurines collection vs figurines Warhammer | Visuellement et sémantiquement très proches |

---

## Synthèse pour le rapport final

- **27 classes** : granularité fine, certaines classes très minoritaires (ex. 1180 Warhammer avec 764 produits).
- **Déséquilibre** : ratio 13,4× entre la plus grande (Piscine 10 209) et la plus petite (Warhammer 764).
- **Classes difficiles** : publications (descriptions absentes), DLC (images peu informatives), textiles (images similaires).
- **Superclasse 24** : fusion des 4 classes publications en une seule ; améliore parfois légèrement les métriques en réduisant la complexité.
