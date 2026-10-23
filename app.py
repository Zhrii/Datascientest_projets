"""
Application Streamlit — vitrine Rakuten.
Les prédictions passent par FastAPI (API_URL) si disponible,
sinon fallback local via ProductPredictor.
"""

import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st

sys.path.append(os.path.abspath("."))

from src.inference.categories import DICT_CATEGORIES
from src.inference.api_client import api_available, predict_via_api, trigger_retrain

# ---------------------------------------------------------------------------
# Page config & style
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rakuten E-commerce AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
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
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Prédiction : API puis fallback local
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du prédicteur local (fallback)...")
def _local_predictor():
    from src.inference.predictor import ProductPredictor

    return ProductPredictor()


def run_predict(text=None, image_file=None):
    """Retourne (result_dict, source) où source ∈ {'api', 'local'}."""
    image_bytes = None
    filename = "image.jpg"
    if image_file is not None:
        image_bytes = image_file.getvalue()
        filename = getattr(image_file, "name", filename) or filename

    if api_available():
        try:
            return predict_via_api(text=text, image_bytes=image_bytes, filename=filename), "api"
        except Exception as exc:  # noqa: BLE001
            st.warning(f"API indisponible ({exc}) — fallback local.")

    predictor = _local_predictor()
    from PIL import Image

    image = Image.open(BytesIO(image_bytes)) if image_bytes else None
    return predictor.predict(text=text, image=image), "local"


def show_prediction(result: dict, source: str):
    st.success(f"Catégorie prédite : **{result['category']}**")
    st.caption(
        f"Code Rakuten : {result['prdtypecode']} · "
        f"Modalité : {result['modality']} · "
        f"Modèle : {result['model']} · "
        f"F1-macro : {result.get('f1_macro', '—')} · "
        f"via {source.upper()}"
    )
    if result.get("fallback"):
        st.info(f"Fallback : {result['fallback']}")
    if result.get("latency_ms") is not None:
        st.caption(f"Latence API : {result['latency_ms']} ms")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

logo_path = Path("images/logo_rakuten.png")
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=350)

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choisissez un module :",
    [
        "1. Accueil & Contexte",
        "2. Défis & Modélisation",
        "3. Résultats",
        "4. Assistant Texte",
        "5. Assistant Image",
        "6. Moteur Multimodal",
        "7. Conclusion",
        "8. MLOps / API",
    ],
)

st.sidebar.markdown("---")
api_ok = api_available()
st.sidebar.caption(f"API FastAPI : {'🟢 online' if api_ok else '🔴 offline (fallback local)'}")
st.sidebar.caption("Projet Data Science - Rakuten 2026")
st.sidebar.caption("Équipe : Morgane, Tristan et Nicolas")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

if page == "1. Accueil & Contexte":
    st.title("Projet Rakuten 2 : Présentation de la solution data")
    st.markdown("**Le Contexte Métier**")
    st.info(
        "Rakuten est une marketplace mondiale mettant en relation des milliers d'acheteurs et de vendeurs.\n\n"
        "Dans cet écosystème décentralisé, la plateforme fait face à un défi de standardisation : "
        "les vendeurs saisissent leurs produits de manière hétérogène.\n\n"
        "Ce manque de structure dégrade la pertinence des recherches et impacte directement le taux de conversion."
    )
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**La donnée du catalogue : Structure et Volumétrie**")
        st.markdown(
            """
        Le jeu de données totalise près de **100 000 produits** issus de la marketplace Rakuten.
        Chaque produit est défini par 4 variables sources :
        - **designation** : Le titre du produit (présent à 100%).
        - **description** : Détails techniques (manquant dans 35% des cas).
        - **productid** : Clé de jointure unique.
        - **imageid** : Identifiant permettant de lier le texte à l'image stockée.

        **Architecture de la Cible (Target) :** La variable à prédire est le **prdtypecode**,
        qui contient **27 catégories distinctes**.
        """
        )
    with col2:
        st.markdown("**L'Équipe Data - Mars 2026**")
        st.markdown(
            """
        - **Nicolas NEGUIRAL**
        - **Tristan PRUVOST**
        - **Morgane BERNARD**
        """
        )

elif page == "2. Défis & Modélisation":
    st.title("Défis Majeurs & Résolution Multimodale")
    st.markdown("### A. Les 4 défis majeurs du Dataset")
    st.info(
        "Le jeu de données Rakuten présente plusieurs anomalies structurelles "
        "qu'il a fallu traiter avec précision avant toute modélisation."
    )
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
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**3. Le Bruit Structurel (Pollution HTML)**")
        st.markdown("- **Constat :** 18,4% des textes sont parasités par des balises (br, li, ul).")
        st.markdown("- **Impact :** Apprentissage de corrélations fallacieuses.")
        st.success("**Stratégie :** Nettoyage via BeautifulSoup.")
    with col4:
        st.markdown("**4. Hétérogénéité des Séquences**")
        st.markdown("- **Constat :** Longueurs instables (11 à 12 000 caractères) et titres multilingues.")
        st.markdown("- **Impact :** Matrices très creuses pour des modèles linéaires.")
        st.success("**Stratégie :** Normalisation NLP et réduction via TruncatedSVD.")

    st.divider()
    st.markdown("### B. Les 3 étapes de résolution")
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("**1. Preprocessing**")
        st.markdown("- Nettoyage BeautifulSoup (99,88%).")
        st.markdown("- NLP : minuscules, accents conservés.")
        st.markdown("- Vectorisation TF-IDF → 300 dim (SVD).")
    with step2:
        st.markdown("**2. Modèles Unimodaux**")
        st.markdown("- Texte : SVM Linéaire F1-Macro 0,78.")
        st.markdown("- Image : CLIP ViT-B/32 → 0,66.")
        st.warning("45% des produits sans image exploitable.")
    with step3:
        st.markdown("**3. Fusion Multimodale**")
        st.markdown("- Late Fusion [TF-IDF + CLIP Image + CLIP Texte].")
        st.markdown("- Matching Recall@10 : 69,1%.")
        st.success("**Résultat final : F1-Macro 0,84.**")

elif page == "3. Résultats":
    st.title("Résultats")
    st.header("1. Comparaison des Performances")
    st.write(
        "Comparaison sur le **F1-Score Macro** "
        "(qui, contrairement à l'Accuracy, ne triche pas sur les classes minoritaires)."
    )
    st.markdown(
        """
    | Approche | Modèle | F1-Score (Macro) | Verdict |
    | :--- | :--- | :--- | :--- |
    | **Baseline Texte** | Naive Bayes / CatBoost | 0.65 - 0.68 | Rapide, mais manque de finesse. |
    | **Champion Texte seul** | SVM Linéaire (TF-IDF) | 0.78 | Très robuste, notre référence texte. |
    | **Baseline Image** | ResNet50 | 0.51 | Faible (ambiguïtés visuelles). |
    | **Champion Image seul** | CLIP (ViT-B/32) | 0.66 | Capture mieux le "sens" des images. |
    | **Fusion Multimodale** | **SVM + TF-IDF + CLIP** | **0.84** | **LE GAGNANT.** |
    """
    )
    st.divider()
    st.subheader("Analyse des erreurs : La Matrice de Confusion")
    cm = Path("images/matrice_confusion.png")
    if cm.exists():
        st.image(str(cm), caption="Matrice de Confusion du Modèle de Fusion", use_container_width=True)

elif page == "4. Assistant Texte":
    st.title("Auto-catégorisation par le Texte")
    st.markdown("L'assistant analyse la désignation et la description pour déduire la catégorie.")
    user_text = st.text_area("Entrez la fiche produit (Titre + Description) :", height=150)
    if st.button("Analyser le texte", type="primary"):
        if not user_text:
            st.error("Veuillez entrer du texte.")
        else:
            with st.spinner("Analyse sémantique en cours..."):
                try:
                    result, source = run_predict(text=user_text)
                    show_prediction(result, source)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Erreur : {e}")

elif page == "5. Assistant Image":
    st.title("Auto-catégorisation par l'Image")
    st.markdown("Importez une photo du produit. L'assistant analyse les pixels pour déduire la catégorie.")
    uploaded_file = st.file_uploader("Uploadez l'image d'un produit (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, width=300)
        if st.button("Analyser l'image", type="primary"):
            with st.spinner("Extraction des caractéristiques visuelles..."):
                try:
                    result, source = run_predict(image_file=uploaded_file)
                    show_prediction(result, source)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Erreur : {e}")

elif page == "6. Moteur Multimodal":
    st.title("Assistant Fusion : TF-IDF + CLIP + SVM")
    st.markdown(
        "Le **3ᵉ modèle** combine texte et image (F1-macro 0,84). "
        "Saisissez la fiche produit et uploadez l'image."
    )
    user_text = st.text_area(
        "Fiche produit (Titre + Description) :",
        height=120,
        placeholder="Ex: Chaise scandinave en bois blanc, design nordique...",
    )
    uploaded_file = st.file_uploader("Image du produit (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if st.button("Analyser (Texte + Image)", type="primary"):
        if not user_text or uploaded_file is None:
            st.error("Veuillez saisir le texte ET uploader une image.")
        else:
            with st.spinner("Extraction des features et prédiction..."):
                try:
                    result, source = run_predict(text=user_text, image_file=uploaded_file)
                    show_prediction(result, source)
                    st.image(uploaded_file, width=200, caption="Image analysée")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Erreur : {e}")

elif page == "7. Conclusion":
    st.header("7. Conclusion du projet Rakuten")
    st.success(
        "Le succès de ce projet réside dans la transition d'une approche textuelle classique "
        "vers un **système multimodal performant**, capable de transformer un catalogue brut et "
        "hétérogène en une base de données structurée."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Performance Multimodale", "84%", "F1-Score Macro")
    col2.metric("Fiabilité Opérationnelle", "8.4 / 10", "Produits classés auto.")
    col3.metric("Modération", "Efficience", "Améliorée")
    col4.metric("Levier de Croissance", "ROI", "SEO & Conversion")
    st.divider()
    st.header("Critique et Perspectives")
    colA, colB = st.columns(2)
    with colA:
        st.info(
            "**Pipeline de Fallback Dynamique**\n\n"
            "Routage intelligent : fusion → SVM texte (0.78) si image absente."
        )
        st.info(
            "**Sémantique Avancée (CamemBERT)**\n\n"
            "Remplacement du TF-IDF par des Transformers pour le jargon e-commerce français."
        )
    with colB:
        st.info(
            "**Fine-tuning Multimodal**\n\n"
            "Dual Encoder propriétaire sur les paires (image, texte) Rakuten."
        )
        st.info(
            "**IA Explicable (XAI)**\n\n"
            "SHAP pour justifier les décisions auprès des modérateurs."
        )

elif page == "8. MLOps / API":
    st.title("Couche MLOps")
    st.markdown(
        """
    Architecture déployée (voir `ARCHITECTURE.md`) :

    | Composant | Port | Rôle |
    |-----------|------|------|
    | FastAPI | 8000 | `/predict`, `/health`, `/model/info`, `/retrain`, `/metrics` |
    | Streamlit | 8501 | UI (cette app) |
    | Pipeline Runner (`train`) | — | ZenML : preprocess_data → train_model → promote_model |
    | ZenML | 8888 | Suivi des pipelines (métadonnées dans MySQL) |
    | MinIO | 9001 | Artefacts S3 (ZenML + MLflow) |
    | MLflow | 5000 | Expériences + Model Registry (alias `production`) |
    | Prometheus | 9092 | Scraping métriques |
    | Grafana | 3002 | Dashboards |
    """
    )
    st.code(".\\make.ps1 up\n.\\make.ps1 train\n.\\make.ps1 down", language="powershell")

    if st.button("Déclencher /retrain via API"):
        if not api_available():
            st.error("API offline. Lancez `.\\make.ps1 api` d'abord.")
        else:
            try:
                resp = trigger_retrain()
                st.success(resp)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    st.markdown("**Catégories référencées :**")
    st.json(DICT_CATEGORIES)
