import streamlit as st
import os
import pickle
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import open_clip
import pandas as pd
from huggingface_hub import hf_hub_download

# Dictionnaire de traduction des catégories Rakuten
DICT_CATEGORIES = {
    10: "Livre d'occasion",
    40: "Jeux vidéo",
    50: "Accessoire jeu vidéo",
    60: "Console de jeux",
    1140: "Figurine / Goodies",
    1160: "Cartes à collectionner",
    1180: "Jeu de plateau / Warhammer",
    1280: "Jouet enfant",
    1281: "Jeu de société",
    1300: "Jouet technique / Modélisme",
    1301: "Vêtement enfant / Puériculture",
    1302: "Jeu d'extérieur",
    1320: "Puériculture",
    1560: "Mobilier",
    1920: "Chambre / Literie",
    1940: "Alimentation / Épicerie",
    2060: "Décoration intérieure",
    2220: "Animalerie",
    2280: "Magazine",
    2403: "Livre / BD",
    2462: "Jeu vidéo (Occasion)",
    2522: "Papeterie",
    2582: "Meuble d'extérieur",
    2583: "Piscine / Spa",
    2585: "Bricolage / Jardinage",
    2705: "Livre neuf",
    2905: "Jeu PC"
}

# 1. CONFIGURATION SYSTÈME ET IMPORTS

sys.path.append(os.path.abspath("."))

try:
    from src.modeling.vectorization import TFIDFVectorizer
    from src.modeling.baseline_models import BaselineModels
    # from src.image.feature_extractor import ImageFeatureExtractor
except ImportError as e:
    st.error(f"Erreur d'importation : {e}")
    st.info("Vérifiez la structure de votre dossier 'src/'.")
    st.stop()


# 2. CONFIGURATION DE LA PAGE STREAMLIT

st.set_page_config(
    page_title="Rakuten E-commerce AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2.5 CHARTE GRAPHIQUE RAKUTEN et PERSONNALISATION

st.markdown("""
    <style>
        /* Cacher les menus Streamlit par défaut */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Alertes info et titres au Rouge Rakuten */
        .stAlert {
            border-left-color: #BF0000 !important;
            background-color: #FFF5F5 !important;
            color: #333333 !important;
        }
        
        h1, h2, h3 {
            color: #BF0000 !important;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)



# 3. TOUS LES CHARGEMENTS DE MODÈLES (BRANCHÉS EN LOCAL)

MODELS_DIR = Path("models")

os.makedirs(MODELS_DIR, exist_ok=True)
HF_REPO = "nicolasN2215/ecommerce-resnet-image"

def get_hf_model(filename):
    """Télécharge le fichier depuis Hugging Face s'il n'est pas sur le PC"""
    local_path = MODELS_DIR / filename
    if not local_path.exists():
        st.toast(f"Téléchargement de {filename}...", icon="⏳")
        hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=MODELS_DIR)
    return local_path

# Fichiers TEXTE
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")  # Fichier local (à la racine)
ENCODER_PATH = Path("label_encoder.pkl")        # Fichier local (à la racine)
MODEL_TEXT_PATH = get_hf_model("svm_linear_baseline_27_classes_final.pkl") 

# Fichier IMAGE
MODEL_IMAGE_PATH = get_hf_model("image_logistic_regression_27_classes_baseline.pkl")

# Fichier MULTIMODAL
CLIP_CACHE_PATH = Path("models/clip_features_cache_lite.npz")

@st.cache_resource(show_spinner="Chargement de l'IA Texte...")
def load_text_pipeline():
    if not VECTORIZER_PATH.exists() or not MODEL_TEXT_PATH.exists():
        return None, None, None
    try:
        vectorizer = TFIDFVectorizer.load(VECTORIZER_PATH)
        model = BaselineModels.load_model(MODEL_TEXT_PATH)
        encoder = pickle.load(open(ENCODER_PATH, 'rb')) if ENCODER_PATH.exists() else None
        return vectorizer, model, encoder
    except Exception:
        return None, None, None

@st.cache_resource(show_spinner="Chargement de l'IA Image (ResNet50)...")
def load_image_pipeline():
    if not MODEL_IMAGE_PATH.exists():
        return None, "Fichier modèle image introuvable dans models/."
    try:
        # On importe l'extracteur
        from src.image.feature_extractor import ImageFeatureExtractor
        
        extractor = ImageFeatureExtractor(output_dim=2048)
        model_image = BaselineModels.load_model(MODEL_IMAGE_PATH)
        return extractor, model_image
    except Exception as e:
        return None, f"Erreur d'importation : {e}"


# MOTEUR MULTIMODAL : FONCTIONS DE CHARGEMENT

@st.cache_resource(show_spinner="Chargement du cerveau CLIP...")
def load_clip_engine():
    try:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval() 
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        return model, tokenizer
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner="Chargement et vérification du catalogue...")
def load_gallery_data():
    if not CLIP_CACHE_PATH.exists():
        return None, None
    
    cache = np.load(CLIP_CACHE_PATH)
    keys = cache.files
    gallery_embeddings = cache[keys[0]] if 'img_val' not in keys else cache['img_val']
    
    try:
        val_df = pd.read_csv("val_df_multimodal_lite.csv")
        # Sécurité
        if len(val_df) != len(gallery_embeddings):
            st.warning(f"Attention : L'IA a trouvé {len(gallery_embeddings)} images mais le CSV contient {len(val_df)} produits !")
        return gallery_embeddings, val_df
        
    except Exception as e:
        st.error(f"Erreur de lecture du CSV : {e}")
        return None, None


# 4. MENU LATÉRAL

# Logo 
st.sidebar.image("images/logo_rakuten.png", width=350)
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choisissez un module :",
    [
        "1. Accueil & Contexte",
        "2. Défis & Modélisation",
        "3. Assistant Texte",
        "4. Assistant Image",
        "5. Moteur Multimodal",
        "6. Résultats & Conclusion"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Projet Data Science - Rakuten 2026")
st.sidebar.caption("Équipe : Morgane, Tristan et Nicolas")


# PAGE 1 : ACCUEIL ET CONTEXTE

if page == "1. Accueil & Contexte":
    st.title("Projet Rakuten 2 : Présentation de la solution data")
    
    st.markdown("**Le Contexte Métier**")
    st.info(
        "Rakuten est une marketplace mondiale mettant en relation des milliers d'acheteurs et de vendeurs.\n\n"
        "Dans cet écosystème décentralisé, la plateforme fait face à un défi de standardisation : les vendeurs saisissent leurs produits de manière hétérogène..\n\n"
        "Ce manque de structure dégrade la pertinence des recherches et impacte directement le taux de conversion."
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**La donnée du catalogue : Structure et Volumétrie**")
        st.markdown("""
        Le jeu de données totalise près de **100 000 produits** issus de la marketplace Rakuten. Chaque produit est défini par 4 variables sources :
        - **designation** : Le titre du produit (présent à 100%).
        - **description** : Détails techniques (manquant dans 35% des cas).
        - **productid** : Clé de jointure unique. 
        - **imageid** : Identifiant permettant de lier le texte à l'image stockée.
        
        **Architecture de la Cible (Target) :** La variable à prédire est le **prdtypecode**, qui contient **27 catégories distinctes**.
        """)
        
    with col2:
        st.markdown("**L'Équipe Data - Mars 2026**")
        st.markdown("""
        - **Nicolas NEGUIRAL**
        - **Tristan PRUVOST**
        - **Morgane BERNARD**
        """)

# PAGE 2 : EXPLORATION (EDA)

elif page == "2. Défis & Modélisation":
    st.title("Défis Majeurs & Résolution Multimodale")
    
    st.markdown("### A. Les 4 défis majeurs du Dataset")
    st.info("Le jeu de données Rakuten présente plusieurs anomalies structurelles qu'il a fallu traiter avec précision avant toute modélisation.")
    
    # Première ligne de défis (2 colonnes)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Le 'Grand Écart' des Classes**")
        st.markdown("- **Constat :** Structure en Longue Traîne (Long Tail).")
        st.markdown("- **Chiffres :** La classe dominante possède 10 209 exemples contre 764 pour la plus rare.")
        st.markdown("- **Impact :** Un modèle naïf maximiserait l'Accuracy au détriment des classes minoritaires.")
        st.success("**Stratégie :** F1-Score Macro et utilisation de `class_weight='balanced'`.")
        
    with col2:
        st.markdown("**2. L'Asymétrie des Données Manquantes (NMAR)**")
        st.markdown("- **Constat :** Le manque d'information dépend de la catégorie (Missing Not At Random).")
        st.markdown("- **Chiffres :** 35,1% des descriptions absentes (quasi nul pour le Mobilier, 97,4% pour les Livres).")
        st.markdown("- **Impact :** Échec du modèle sur 1/3 du catalogue si on s'appuie trop sur la description.")
        st.success("**Stratégie :** Création d'une feature hybride `text_combined` (Désignation + Description).")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Deuxième ligne de défis (2 colonnes)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**3. Le Bruit Structurel (Pollution HTML)**")
        st.markdown("- **Constat :** 18,4% des textes sont parasités par des balises (br, li, ul).")
        st.markdown("- **Impact :** Apprentissage de corrélations fallacieuses (le modèle classifie selon la mise en page).")
        st.success("**Stratégie :** Nettoyage 'chirurgical' via BeautifulSoup.")
        
    with col4:
        st.markdown("**4. Hétérogénéité des Séquences**")
        st.markdown("- **Constat :** Longueurs instables (11 à 12 000 caractères) et titres multilingues.")
        st.markdown("- **Impact :** Matrices de données très creuses et instables pour des modèles linéaires.")
        st.success("**Stratégie :** Normalisation NLP stricte et réduction via TruncatedSVD.")

    st.divider()

    st.markdown("### B. Les 3 étapes de résolution")
    
    # Les 3 étapes en 3 colonnes
    step1, step2, step3 = st.columns(3)
    
    with step1:
        st.markdown("**1. Preprocessing**")
        st.markdown("- **Nettoyage :** BeautifulSoup (99,88% de réussite).")
        st.markdown("- **NLP :** Minuscules, sans ponctuation, avec accents conservés.")
        st.markdown("- **Features :** Création de 15 variables.")
        st.markdown("- **Vectorisation :** TF-IDF réduit à 300 dimensions.")
        
    with step2:
        st.markdown("**2. Modèles Unimodaux**")
        st.markdown("- **Texte :** SVM Linéaire atteint un F1-Macro de 0,78.")
        st.markdown("- **Image :** CLIP (ViT-B/32) atteint 0,66.")
        st.warning("**Difficulté :** 45% des produits n'ont pas d'image exploitable.")
        
    with step3:
        st.markdown("**3. Fusion Multimodale**")
        st.markdown("- **Méthode :** Concaténation Late Fusion [TF-IDF + CLIP Image + CLIP Texte].")
        st.markdown("- **Matching :** Recall@10 de 69,1% via CLIP.")
        st.success("**Résultat final :** F1-Macro à 0,84 ! C'est le modèle recommandé à Rakuten.")


# PAGE 3 : ASSISTANT TEXTE

elif page == "3. Assistant Texte":
    st.title("Auto-catégorisation par le Texte")
    st.markdown("L'assistant analyse la désignation et la description pour déduire la catégorie.")
    
    vectorizer, model_text, encoder = load_text_pipeline()
    
    if model_text is None:
        st.warning("Modèles locaux non détectés. Placez vos fichiers .pkl dans le dossier models/.")
    else:
        st.success("Modèle Texte chargé depuis les fichiers locaux.")
        
        user_text = st.text_area("Entrez la fiche produit (Titre + Description) :", height=150)
        
        if st.button("Analyser le texte", type="primary"):
            if not user_text:
                st.error("Veuillez entrer du texte.")
            else:
                with st.spinner("Analyse sémantique en cours..."):
                    # Dictionnaire de traduction des codes Rakuten
                    dict_categories = {
                        10: "Livre d'occasion", 40: "Jeu vidéo", 50: "Accessoire de jeu vidéo",
                        60: "Console de jeu", 1140: "Figurine / Produit dérivé", 
                        1160: "Cartes à collectionner", 1280: "Jouets pour enfants", 
                        1281: "Jeux de société", 1300: "Jouets techniques (RC, Drones)", 
                        1320: "Puériculture", 1560: "Mobilier d'intérieur", 
                        1920: "Literie / Linge de maison", 1940: "Épicerie / Boissons", 
                        2060: "Décoration d'intérieur", 2220: "Animalerie", 
                        2280: "Magazines", 2403: "Livres anciens", 2462: "Jeux PC", 
                        2522: "Papeterie", 2582: "Meubles de jardin", 
                        2583: "Piscine / Accessoires", 2585: "Bricolage / Outillage", 
                        2705: "Livres", 2905: "Jeu PC (Dématérialisé)"
                    }

                    # La vraie prédiction
                    X_vec = vectorizer.transform(pd.Series([user_text]))
                    pred_code_interne = model_text.predict(X_vec)[0]
                    
                    # On retrouve le code Rakuten 
                    code_rakuten = encoder.inverse_transform([pred_code_interne])[0]
                    
                    # On traduit en français 
                    nom_categorie = dict_categories.get(code_rakuten, "Catégorie non répertoriée")
                    
                    st.success(f"Catégorie prédite : **{nom_categorie}**")
                    st.caption(f"Code Rakuten correspondant : {code_rakuten}")

# PAGE 4 : ASSISTANT IMAGE

elif page == "4. Assistant Image":
    st.title("Auto-catégorisation par l'Image")
    st.markdown("Importez une photo du produit. L'assistant va analyser les pixels pour déduire la catégorie.")
    
    extractor, model_image = load_image_pipeline()
    _, _, encoder = load_text_pipeline() # On récupère l'encodeur de la page 3
    
    if extractor is None:
        st.warning(f"Modèle Image non disponible. Détail : {model_image}")
    else:
        uploaded_file = st.file_uploader("Uploadez l'image d'un produit (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # 1. Affichage de l'image
            image = Image.open(uploaded_file)
            st.image(image, width=300)
            
            if st.button("Analyser l'image", type="primary"):
                with st.spinner("Extraction des caractéristiques visuelles en cours..."):
                    try:
                        # 2. Sauvegarde temporaire
                        temp_path = "temp_image.jpg"
                        image.convert('RGB').save(temp_path)
                        
                        # 3. Extraction (ResNet50) et Prédiction
                        features = extractor.extract_from_paths([temp_path])
                        pred_code_interne = model_image.predict(features)[0]
                        code_rakuten = encoder.inverse_transform([pred_code_interne])[0]
                        
                        # 4. Traduction en français
                        dict_categories = {
                            10: "Livre d'occasion", 40: "Jeu vidéo", 50: "Accessoire de jeu vidéo",
                            60: "Console de jeu", 1140: "Figurine / Produit dérivé", 
                            1160: "Cartes à collectionner", 1280: "Jouets pour enfants", 
                            1281: "Jeux de société", 1300: "Jouets techniques (RC, Drones)", 
                            1320: "Puériculture", 1560: "Mobilier d'intérieur", 
                            1920: "Literie / Linge de maison", 1940: "Épicerie / Boissons", 
                            2060: "Décoration d'intérieur", 2220: "Animalerie", 
                            2280: "Magazines", 2403: "Livres anciens", 2462: "Jeux PC", 
                            2522: "Papeterie", 2582: "Meubles de jardin", 
                            2583: "Piscine / Accessoires", 2585: "Bricolage / Outillage", 
                            2705: "Livres", 2905: "Jeu PC (Dématérialisé)"
                        }
                        nom_categorie = dict_categories.get(code_rakuten, "Catégorie inconnue")
                        
                        st.success(f"Catégorie prédite : **{nom_categorie}**")
                        st.caption(f"Code Rakuten correspondant : {code_rakuten}")
                        
                    except Exception as e:
                        st.error(f"Erreur pendant l'analyse : {e}")

# PAGE 5 : MOTEUR MULTIMODAL (Interface)

elif page == "5. Moteur Multimodal":
    st.title("Recherche Intelligente (Texte -> Image)")
    st.markdown("Cet assistant le lien entre vos mots et les pixels de notre catalogue.")

    clip_model, tokenizer = load_clip_engine()
    gallery_embeddings, df_gallery = load_gallery_data()

    if clip_model is None or gallery_embeddings is None:
       st.warning("**Moteur inactif.** Vérifiez que le fichier `multimodal_clip_cache.npz` est dans `models/` et que vos dossiers `data brut` et `data` sont bien présents.")
    else:
        search_query = st.text_input("Que recherchez-vous dans le catalogue Rakuten ?", placeholder="Ex: Une chaise scandinave en bois blanc...")

        if st.button("Chercher les produits", type="primary"):
            if not search_query:
                st.error("Veuillez entrer une description pour lancer la recherche.")
            else:
                with st.spinner(f"Recherche de '{search_query}' parmi {len(df_gallery)} produits..."):
                    
                    # 1. Encodage du texte
                    tokens = tokenizer([search_query])
                    with torch.no_grad():
                        text_features = clip_model.encode_text(tokens)
                        
                    # 2. Normalisation L2 
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.cpu().numpy()
                    
                    # 3. Calcul de similarité 
                    sims = (text_features @ gallery_embeddings.T).flatten()
                    
                    # 4. Top 3 des meilleurs matchs
                    top3_indices = np.argsort(sims)[::-1][:3]
                    
                    st.success("Voici les produits les plus pertinents :")
                    cols = st.columns(3)
                    
                    for i, col in enumerate(cols):
                        best_idx = top3_indices[i]
                        score = sims[best_idx] * 100
                        
                        # Récupération sécurisée du chemin de l'image
                        image_path = df_gallery.iloc[best_idx]['image_path']
                        code_categorie = df_gallery.iloc[best_idx]['prdtypecode']
                        
                        with col:
                            full_path = Path(".") / image_path
                            if full_path.exists():
                                st.image(str(full_path), use_container_width=True)
                            else:
                                st.info(f"Image introuvable : {image_path}")
                                
                            st.caption(f"Match #{i+1} : {score:.1f}%")
                            # On utilise la fonction .get() pour trouver le nom, sinon on affiche le code par défaut
                            nom_categorie = DICT_CATEGORIES.get(code_categorie, f"Code {code_categorie}")
                            # On affiche le nom
                            st.caption(f"Catégorie : **{nom_categorie}**")

    # PAGE 6 : Resultats et conclusion
    
elif page == "6. Résultats & Conclusion":
    st.title("Résultats & Conclusion")

    # --- PARTIE 1 : COMPARAISON ---
    st.header("1. Comparaison des Performances")
    st.write(
        "L'objectif de cette étape était de comparer nos différentes approches sur une même métrique : "
        "le **F1-Score Macro** (qui, contrairement à l'Accuracy, ne triche pas sur les classes minoritaires)."
    )

    st.markdown("""
    | Approche | Modèle | F1-Score (Macro) | Verdict |
    | :--- | :--- | :--- | :--- |
    | **Baseline Texte** | Naive Bayes / CatBoost | 0.65 - 0.68 | Rapide, mais manque de finesse. |
    | **Champion Texte seul** | SVM Linéaire (TF-IDF) | 0.78 | Très robuste, notre référence texte. |
    | **Baseline Image** | ResNet50 | 0.51 | Faible (trop d'ambiguïtés visuelles). |
    | **Champion Image seul** | CLIP (ViT-B/32) | 0.66 | Capture mieux le "sens" des images. |
    | **Fusion Multimodale** | **SVM + TF-IDF + CLIP** | **0.84** | **🏆 LE GAGNANT (Synergie maximale).** |
    """)

    st.divider()

    # --- PARTIE 2 : CONCLUSION ---
    st.header("2. Conclusion du projet Rakuten")
    st.success(
        "Le succès de ce projet réside dans la transition d'une approche textuelle classique "
        "vers un **système multimodal performant**, capable de transformer un catalogue brut et "
        "hétérogène en une base de données structurée."
    )

    # Affichage des KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Performance Multimodale", "84%", "F1-Score Macro")
    col2.metric("Fiabilité Opérationnelle", "8.4 / 10", "Produits classés auto.")
    col3.metric("Modération", "Efficience", "Améliorée")
    col4.metric("Levier de Croissance", "ROI", "SEO & Conversion")

    st.divider()

    # --- PARTIE 3 : PERSPECTIVES ---
    st.header("3. Critique et Perspectives")
    st.write("Afin d'industrialiser ce modèle et de le mettre en production, 4 axes d'amélioration ont été identifiés :")

    colA, colB = st.columns(2)
    with colA:
        st.info(
            "**Pipeline de 'Fallback' Dynamique**\n\n"
            "Mise en œuvre d'un routage intelligent basculant sur le modèle SVM (0.78) ou CatBoost (0.68) "
            "en l'absence d'image, assurant une couverture de 100 % du catalogue."
        )
        st.info(
            "**Sémantique Avancée (CamemBERT)**\n\n"
            "Remplacement du TF-IDF par des modèles de langage (Transformers) pour capturer les nuances "
            "contextuelles et le jargon spécifique au e-commerce français."
        )

    with colB:
        st.info(
            "**Fine-tuning Multimodal**\n\n"
            "Entraînement d'un Dual Encoder propriétaire sur les paires (image, texte) de Rakuten "
            "pour affiner l'alignement sémantique par rapport au modèle CLIP générique."
        )
        st.info(
            "**IA Explicable (XAI)**\n\n"
            "Intégration de la méthode SHAP pour fournir aux modérateurs les mots-clés et zones d'images "
            "ayant justifié la décision du modèle, renforçant la confiance et la transparence."
        )