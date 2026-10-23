# Checklist démo live — Stack MLOps

Durée cible : **3–4 minutes** (après la partie data).

## Avant la démo

```powershell
cd "F:\Projets\Projet ecole\Datascientest_projets"
docker compose ps          # tous Up / healthy
.\make.ps1 health          # text_ready / image_ready
```

Ouvrir dans le navigateur (onglets prêts) :
1. http://localhost:8501 — Streamlit  
2. http://localhost:8000/docs — FastAPI  
3. http://localhost:3002 — Grafana (`admin` / `rakuten`)  
4. http://localhost:5000 — MLflow (optionnel)  
5. http://localhost:8888 — ZenML (optionnel)

## Script (ordre recommandé)

### 1. Streamlit — prédiction (≈1 min)
- Sidebar : vérifier **API FastAPI 🟢 online**
- Page **4. Assistant Texte** : coller  
  `Livre Harry Potter à l'école des sorciers édition poche`  
  → catégorie type Livre / BD, mention **via API**
- (Si fusion OK) Page **6. Moteur Multimodal** : texte + image produit

### 2. FastAPI Swagger (≈45 sec)
- `/docs` → `POST /predict` → Try it out  
- Montrer `/health` et `/model/info`  
- Message clé : *Streamlit n’est qu’un client ; le service métier est l’API*

### 3. Grafana monitoring (≈45 sec)
- Dashboard **Rakuten Inference API**  
- Montrer compteurs / latence après les prédictions de la démo  
- Message clé : *on observe la prod, pas seulement le F1 du notebook*

### 4. Closing optionnel (≈30 sec)
- MLflow : une run d’entraînement avec métrique F1  
- Ou ZenML : “orchestration du pipeline preprocess → train”

## Phrase de conclusion

> On est partis d’un modèle labo à F1 0,84. Aujourd’hui c’est un service FastAPI, une UI, un pipeline rejouable et un monitoring — la même architecture que sur notre schéma ZenML.

## Si un service est down

```powershell
.\make.ps1 up
# attendre ~2 min le healthcheck API (chargement CLIP/ResNet)
```
