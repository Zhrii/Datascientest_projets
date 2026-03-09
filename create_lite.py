import pandas as pd
import numpy as np
import shutil
import os
from pathlib import Path

# --- CONFIGURATION ---
NB_IMAGES = 2000
DOSSIER_SOURCE = r"C:\Users\Nicolas\analyse_images_ecommerce\images_extraites\image_train"
DOSSIER_DESTINATION = r"data\processed\image_clean"

print(f"🚀 Création de la version LITE ({NB_IMAGES} produits)...")

# 1. Création du dossier propre
os.makedirs(DOSSIER_DESTINATION, exist_ok=True)

# 2. Découpage du CSV
df = pd.read_csv("val_df_multimodal.csv")
df_lite = df.head(NB_IMAGES)
df_lite.to_csv("val_df_multimodal_lite.csv", index=False)
print("✅ CSV Lite créé !")

# 3. Découpage du NPZ
cache = np.load("models/clip_features_cache.npz") # Modifie le chemin si ton npz n'est pas dans models/
keys = cache.files
embeds = cache[keys[0]] if 'img_val' not in keys else cache['img_val']
embeds_lite = embeds[:NB_IMAGES]
np.savez("models/clip_features_cache_lite.npz", img_val=embeds_lite)
print("✅ NPZ Lite créé !")

# 4. Copie des 2000 images exactes
print("📸 Copie des images (ça va prendre 10 secondes)...")
for img_path in df_lite['image_path']:
    # On extrait juste le nom (ex: image_123.jpg)
    nom_fichier = img_path.split('\\')[-1].split('/')[-1]
    chemin_source = os.path.join(DOSSIER_SOURCE, nom_fichier)
    chemin_dest = os.path.join(DOSSIER_DESTINATION, nom_fichier)
    
    if os.path.exists(chemin_source):
        shutil.copy(chemin_source, chemin_dest)

print(f"🎉 Terminé ! {NB_IMAGES} images copiées dans le dossier de l'appli.")