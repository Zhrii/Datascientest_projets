import numpy as np
import os

print("🚀 Lancement de la création du fichier NPZ allégé...")

# 1. On s'assure que le dossier de destination existe
os.makedirs("models", exist_ok=True)

try:
    # 2. On charge ton fichier d'origine qui est à la racine
    print("📥 Chargement du gros fichier d'origine...")
    cache = np.load("clip_features_cache.npz")
    
    # 3. On crée un dictionnaire pour stocker les nouvelles données allégées
    data_lite = {}
    
    # 4. On découpe PROPREMENT les 4 compartiments à 2000 lignes
    print("✂️ Découpage des données (2000 lignes max)...")
    for key in cache.files:
        data_lite[key] = cache[key][:2000]
        print(f"   - {key} : nouvelle taille {data_lite[key].shape}")
        
    # 5. On sauvegarde le tout dans le dossier models
    chemin_final = "models/clip_features_cache_lite.npz"
    np.savez(chemin_final, **data_lite)
    
    print(f"\n✅ SUCCÈS ABSOLU ! Le fichier a été créé ici : {chemin_final}")

except FileNotFoundError:
    print("\n❌ ERREUR : Le fichier 'clip_features_cache.npz' est introuvable à cet endroit.")
except Exception as e:
    print(f"\n❌ ERREUR INCONNUE : {e}")