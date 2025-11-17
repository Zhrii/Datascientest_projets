# Guide de Configuration Git

Ce guide vous aide à mettre en place Git pour ce projet.

## 🚀 Initialisation du Repository

### 1. Initialiser Git (si pas déjà fait)

```bash
git init
```

### 2. Vérifier les fichiers à commiter

```bash
git status
```

Vous devriez voir que les fichiers CSV dans `data brut/` sont ignorés (c'est normal).

### 3. Ajouter les fichiers au staging

```bash
# Ajouter tous les fichiers (sauf ceux dans .gitignore)
git add .

# Ou ajouter fichier par fichier
git add README.md
git add notebooks/
git add rendu\ textuel/
git add developpement/
# etc.
```

### 4. Faire le premier commit

```bash
git commit -m "Initial commit: Projet classification produits e-commerce"
```

### 5. Ajouter le remote (si vous avez un repo GitHub/GitLab)

```bash
# Remplacer <url-du-repo> par l'URL de votre repository
git remote add origin <url-du-repo>
```

### 6. Pousser vers le remote

```bash
# Première fois
git push -u origin main

# Ou si votre branche s'appelle "master"
git push -u origin master
```

## 📋 Fichiers ignorés par Git

Les fichiers suivants sont automatiquement ignorés (voir `.gitignore`) :

- ✅ Fichiers CSV dans `data brut/` (trop volumineux)
- ✅ Environnement virtuel (`venv/`, `.venv/`)
- ✅ Fichiers `.env` (variables d'environnement)
- ✅ Fichiers Python compilés (`__pycache__/`, `*.pyc`)
- ✅ Checkpoints Jupyter (`.ipynb_checkpoints/`)
- ✅ Modèles sauvegardés (`*.pkl`, `*.joblib`)
- ✅ Logs (`*.log`)

## ⚠️ Important : Données non versionnées

Les fichiers de données (`X_train_update.csv`, `X_test_update.csv`, `Y_train.csv`) **ne sont pas versionnés** dans Git.

Si quelqu'un clone votre repository, il devra :
1. Ajouter manuellement les fichiers CSV dans `data brut/`
2. Ou utiliser un système de stockage externe (Google Drive, Dropbox, etc.)

## 🔍 Vérifications utiles

### Voir ce qui sera commité

```bash
git status
```

### Voir les différences

```bash
git diff
```

### Voir l'historique

```bash
git log --oneline
```

## 📝 Bonnes pratiques

1. **Commits réguliers** : Faites des commits fréquents avec des messages clairs
2. **Messages de commit descriptifs** : 
   - ✅ `git commit -m "Ajout visualisation distribution des classes"`
   - ❌ `git commit -m "update"`
3. **Ne pas commiter les données** : Les CSV restent locaux
4. **Utiliser des branches** pour les nouvelles fonctionnalités :
   ```bash
   git checkout -b feature/preprocessing
   ```

## 🆘 Problèmes courants

### "Les fichiers CSV apparaissent dans git status"

Vérifiez que `.gitignore` est bien à la racine du projet et contient :
```
data brut/*.csv
```

### "Erreur: remote origin already exists"

Si vous avez déjà un remote, vous pouvez le modifier :
```bash
git remote set-url origin <nouvelle-url>
```

### "Les données ne sont pas ignorées"

Assurez-vous que les fichiers CSV ne sont pas déjà trackés :
```bash
# Retirer les fichiers CSV du tracking (sans les supprimer)
git rm --cached "data brut/*.csv"
git commit -m "Remove CSV files from tracking"
```

