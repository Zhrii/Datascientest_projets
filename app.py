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
import requests

from src.inference.api_client import (
    api_available,
    burst_predict,
    predict_via_api,
    probe_without_auth,
    retrain_as_user,
    send_drift_traffic,
    send_feedback,
    simulate_feedback,
    trigger_retrain,
)

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
        except requests.HTTPError as exc:
            # refus de l'API (mot de passe, limite de requêtes) : on l'affiche tel quel
            code = exc.response.status_code
            if code == 401:
                raise RuntimeError("API : identifiants refusés (401)") from exc
            if code == 429:
                wait = exc.response.headers.get("Retry-After", "?")
                raise RuntimeError(f"API : limite de requêtes atteinte (429), réessayer dans {wait} s") from exc
            st.warning(f"API en erreur ({code}) — fallback local.")
        except requests.RequestException as exc:
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
    if result.get("confidence") is not None:
        if result.get("low_confidence"):
            st.warning(f"⚠️ Prédiction peu sûre (confiance {result['confidence']:.2f}) : à faire vérifier par un modérateur.")
        else:
            st.caption(f"Confiance du modèle : {result['confidence']:.2f}")
    if result.get("fallback"):
        st.info(f"Fallback : {result['fallback']}")
    if result.get("latency_ms") is not None:
        st.caption(f"Latence API : {result['latency_ms']} ms")
    if source == "api" and result.get("prediction_id"):
        # gardé en session : le formulaire de correction survit au rechargement de la page
        st.session_state["last_prediction"] = {**result, "page": st.session_state.get("page")}


def feedback_form():
    """Correction de la dernière prédiction par un modérateur (POST /feedback)."""
    last = st.session_state.get("last_prediction")
    if not last or last.get("page") != st.session_state.get("page"):
        return
    st.divider()
    st.markdown("**Retour modérateur** — la catégorie prédite est-elle la bonne ?")
    codes = list(DICT_CATEGORIES)
    choice = st.selectbox(
        "Vraie catégorie", codes, index=codes.index(last["prdtypecode"]) if last["prdtypecode"] in codes else 0,
        format_func=lambda c: f"{DICT_CATEGORIES[c]} ({c})", key=f"fb_{last['prediction_id']}",
    )
    if st.button("Envoyer le retour"):
        try:
            resp = send_feedback(last["prediction_id"], choice)
            verdict = "✅ le modèle avait raison" if resp["correct"] else "❌ le modèle s'était trompé"
            st.success(f"Retour enregistré : {verdict}. Il compte dans le F1 en production (Grafana).")
            st.session_state.pop("last_prediction", None)
        except requests.HTTPError as exc:
            st.error(f"Retour refusé ({exc.response.status_code}) : {exc.response.json().get('detail')}")


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

st.session_state["page"] = page

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
    | nginx (`api-gateway`) | 8000 | Répartit la charge sur les répliques de l'API |
    | FastAPI (×2) | — | `/predict` 🔒, `/feedback` 🔒, `/model/info` 🔒👑, `/retrain` 🔒👑, `/health`, `/metrics` |
    | Redis | — | État partagé des répliques (limites, verrou, journal des prédictions) |
    | Streamlit | 8501 | UI (cette app) |
    | Pipeline Runner (`train`) | — | ZenML : collect → validate → preprocess → train → promote |
    | ZenML | 8888 | Suivi des pipelines (métadonnées dans MySQL) |
    | MinIO | 9001 | S3 : artefacts ZenML, MLflow et données versionnées par DVC |
    | MLflow | 5000 | Expériences + Model Registry (alias `production`) |
    | Scheduler | — | Déclenche `/retrain` tout seul (modèle trop vieux / +10 % de données) |
    | Drift monitor | — | Dérive des données (Evidently) + F1 en production (retours modérateurs) |
    | Prometheus | 9092 | Métriques + règles d'alerte |
    | Grafana | 3002 | Dashboards + alertes actives |

    🔒 login / mot de passe · 👑 rôle admin
    """
    )
    st.code(".\\make.ps1 up\n.\\make.ps1 train\n.\\make.ps1 down", language="powershell")

    # ---------------------------------------------------------------- sécurité
    st.subheader("1. Sécurité de l'API")
    st.markdown(
        "- 🔒 **Authentification** (HTTP Basic) : compte `demo` (rôle *user*) ou `admin` (rôle *admin*)\n"
        "- 👑 **Autorisation** : seul un *admin* peut réentraîner ou lire `/model/info` (sinon **403**)\n"
        "- ⏱️ **Limite de requêtes** : 30 prédictions / minute pour un *user* (au-delà : **429**)\n"
        "- 📈 Chaque refus est compté dans Prometheus → alertes Grafana"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("/predict sans mot de passe"):
            try:
                st.info(f"Réponse : **{probe_without_auth()}** (401 = non authentifié)")
            except requests.RequestException as e:
                st.error(str(e))
    with col_b:
        if st.button("/retrain avec le compte user"):
            try:
                st.info(f"Réponse : **{retrain_as_user()}** (403 = rôle admin requis)")
            except requests.RequestException as e:
                st.error(str(e))
    with col_c:
        if st.button("40 prédictions d'affilée"):
            with st.spinner("Rafale en cours..."):
                try:
                    burst = burst_predict(40)
                    st.info(f"Codes HTTP : {burst['codes']} (200 = OK, 429 = limite)")
                    st.session_state["burst_instances"] = burst["instances"]
                except requests.RequestException as e:
                    st.error(str(e))

    # ---------------------------------------------------------------- scalabilité
    st.subheader("2. Scalabilité")
    st.markdown(
        "Deux répliques de l'API tournent derrière **nginx**. Redis partage leur état : la limite de requêtes "
        "est la même quelle que soit la réplique, un seul réentraînement à la fois, et chaque réplique recharge "
        "seule le nouveau modèle *production* du Model Registry."
    )
    if st.session_state.get("burst_instances"):
        st.write("Répartition de la dernière rafale entre les répliques :")
        st.bar_chart(st.session_state["burst_instances"])
    else:
        st.caption("Lancez la rafale ci-dessus pour voir la répartition entre les répliques.")

    # ---------------------------------------------------------------- données
    st.subheader("3. Qualité et dérive des données")
    st.markdown(
        "- ✅ **Validation** : le pipeline s'arrête si les données sont cassées (colonnes, 27 classes, doublons...)\n"
        "- 📦 **Versioning** : les CSV sont versionnés avec **DVC** (stockage MinIO) ; chaque run MLflow "
        "garde l'empreinte des données utilisées\n"
        "- 📉 **Dérive** : le drift-monitor compare les derniers produits reçus aux données d'entraînement "
        "(Evidently) toutes les 2 minutes"
    )
    reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
    status_file = reports_dir / "drift_status.json"
    if status_file.exists():
        import json

        status = json.loads(status_file.read_text(encoding="utf-8"))
        if status["status"] == "waiting":
            st.info(f"Dérive : {status['message']}.")
        else:
            label = "🔴 Dérive détectée" if status["drift_detected"] else "🟢 Pas de dérive"
            st.metric(label, f"{status['drift_share']:.0%} des variables",
                      help=f"{status['n_current']} prédictions comparées à {status['n_reference']} produits d'entraînement")
    if st.button("Simuler une dérive (250 produits inhabituels)"):
        with st.spinner("Envoi de produits très différents des données d'entraînement..."):
            try:
                st.info(f"Codes HTTP : {send_drift_traffic(250)} — résultat à la prochaine analyse (≤ 2 min)")
            except requests.RequestException as e:
                st.error(str(e))
    report = reports_dir / "drift_report.html"
    if report.exists():
        with st.expander("Rapport Evidently complet"):
            import streamlit.components.v1 as components

            components.html(report.read_text(encoding="utf-8"), height=900, scrolling=True)

    # ---------------------------------------------------------------- performance
    st.subheader("4. Performance du modèle en production")
    st.markdown(
        "Le F1 ne se mesure que si l'on connaît la **vraie catégorie** : les modérateurs la renvoient via "
        "`POST /feedback` (formulaire sous chaque prédiction). Le drift-monitor calcule le **F1 en production** "
        "sur les 500 derniers retours, et l'**accuracy selon la confiance** du modèle."
    )
    perf_file = reports_dir / "performance_status.json"
    if perf_file.exists():
        import json

        perf = json.loads(perf_file.read_text(encoding="utf-8"))
        if perf["status"] == "waiting":
            st.info(f"F1 en production : {perf['message']}.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("F1-macro en production", f"{perf['f1_macro']:.3f}", help=f"{perf['n_feedback']} retours")
            by_conf = perf.get("by_confidence", {})
            if "high" in by_conf:
                c2.metric("Accuracy — prédictions sûres", f"{by_conf['high']['accuracy']:.0%}")
            if "low" in by_conf:
                c3.metric("Accuracy — prédictions peu sûres", f"{by_conf['low']['accuracy']:.0%}",
                          help=f"{perf['low_confidence_share']:.0%} des prédictions")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("Simuler 300 retours de modérateurs"):
            with st.spinner("Prédiction de produits mis de côté puis envoi de leur vraie catégorie..."):
                try:
                    st.info(f"Prédictions : {simulate_feedback(300)} — F1 mis à jour à la prochaine analyse (≤ 2 min)")
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
    with col_f2:
        if st.button("Simuler 300 fiches produit bâclées"):
            with st.spinner("Titres réduits à un seul mot : le modèle va se tromper davantage..."):
                try:
                    st.info(f"Prédictions : {simulate_feedback(300, degraded=True)} — le F1 en production doit chuter")
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))

    # ---------------------------------------------------------------- réentraînement
    st.subheader("5. Réentraînement")
    if st.button("Déclencher /retrain (compte admin)"):
        if not api_available():
            st.error("API offline. Lancez `.\\make.ps1 up` d'abord.")
        else:
            try:
                st.success(trigger_retrain())
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    st.markdown("**Catégories référencées :**")
    st.json(DICT_CATEGORIES)


# formulaire de retour modérateur sous les assistants de prédiction
if page in ("4. Assistant Texte", "5. Assistant Image", "6. Moteur Multimodal"):
    feedback_form()
