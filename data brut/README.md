# Données du Projet

## ⚠️ Important

Les fichiers de données (`*.csv`) dans ce dossier **ne sont pas versionnés** dans Git pour des raisons de taille et de confidentialité.

## Fichiers attendus

Pour que le projet fonctionne, vous devez placer les fichiers suivants dans ce dossier :

- `X_train_update.csv` - Données d'entraînement (84 916 produits)
- `X_test_update.csv` - Données de test (13 814 produits)
- `Y_train.csv` - Labels d'entraînement (84 916 labels)

## Structure attendue

```
data brut/
├── X_train_update.csv
├── X_test_update.csv
├── Y_train.csv
└── README.md (ce fichier)
```

## Images (partie classification / matching)

Pour les notebooks images (07, 08, etc.) et le matching texte-image, les images doivent être placées dans :

```
data/processed/image_train/
```

**Format des noms** : `image_{imageid}_product_{productid}.jpg`

Exemple : `image_1263597046_product_3804725264.jpg`

Sources possibles : archive fournie par Datascientest ou Challenge Data ENS.

## Note

Si vous clonez ce repository, vous devrez ajouter manuellement ces fichiers de données dans ce dossier.

