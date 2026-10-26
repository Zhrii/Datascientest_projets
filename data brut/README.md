# Données du Projet

## ⚠️ Important

Les CSV de ce dossier sont **versionnés avec DVC** (pas avec Git, pour des raisons de taille) :

- Git ne contient que `X_train_update.csv.dvc`, `Y_train.csv.dvc`, `X_test_update.csv.dvc` : l'empreinte (md5) de chaque fichier ;
- les CSV eux-mêmes sont stockés dans MinIO (bucket `dvc`, voir `.dvc/config`).

```powershell
.\make.ps1 up          # MinIO doit tourner
.\make.ps1 data-pull   # récupère les CSV correspondant au commit courant
```

Après une mise à jour des données :

```powershell
python -m dvc add "data brut/X_train_update.csv"   # met à jour l'empreinte
git add "data brut/X_train_update.csv.dvc"
.\make.ps1 data-push
```

Chaque entraînement enregistre ces empreintes dans MLflow (`data_X_train_update_md5`) : on sait avec quelles données chaque modèle a été entraîné.

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

Le stockage DVC est le MinIO local de la stack Docker. Sur un nouveau poste (MinIO vide), placez les CSV à la main dans ce dossier, puis `.\make.ps1 data-push` pour les y déposer.

