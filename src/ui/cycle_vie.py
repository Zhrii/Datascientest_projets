import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from scripts.demo_soutenance import SANDBOX

F1_PAR_DEFAUT = 0.773


def _lire_metrics():
    try:
        return json.loads(Path("models/metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _lire_json(chemin):
    try:
        return json.loads(Path(chemin).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _historique():
    lignes = []
    for dossier in sorted(Path("models/archive").glob("*"), reverse=True):
        if not dossier.is_dir():
            continue
        info = _lire_json(dossier / "metrics.json") or {}
        f1 = info.get("f1_macro")
        lignes.append(
            {
                "Archivé le": dossier.name[:8][:4] + "-" + dossier.name[4:6] + "-" + dossier.name[6:8],
                "Score F1": f"{f1:.3f}" if f1 else "inconnu",
                "Entraîné le": str(info.get("trained_at", "inconnu"))[:10],
            }
        )
    return lignes


def _decision(f1_prod, f1_candidat, gain_min):
    if f1_candidat >= f1_prod + gain_min:
        return True, f"{f1_candidat:.4f} ≥ {f1_prod:.4f} + {gain_min:.4f}"
    return False, f"{f1_candidat:.4f} < {f1_prod:.4f} + {gain_min:.4f}"


def _essayer(fonction, *args, **kwargs):
    try:
        return True, fonction(*args, **kwargs)
    except Exception as exc:
        return False, f"{type(exc).__name__} : {exc}"


def _produit_complet():
    from src.inference import api_client as api

    st.caption(
        "Ici, plus de copie : on parle au **vrai système** (l'API et ses deux copies derrière nginx, la sécurité, "
        "la surveillance). Il faut que la pile Docker tourne : `.\\make.ps1 up`."
    )
    en_ligne = api.api_available()
    if en_ligne:
        st.success("Le système répond.")
    else:
        st.warning("Le système ne répond pas. Lance `.\\make.ps1 up`, attends 1 à 2 minutes, puis recharge la page.")

    onglets = st.tabs(["1. Le service", "2. Sécurité", "3. Montée en charge", "4. Surveillance", "5. Mise à jour"])

    with onglets[0]:
        st.markdown("**On envoie un produit, le service répond sa catégorie.**")
        titre = st.text_input("Titre du produit", "Chaise scandinave en bois massif", key="p9_titre")
        col_a, col_b = st.columns(2)
        if col_a.button("Prédire avec le vrai système", disabled=not en_ligne, key="p9_predire"):
            ok, res = _essayer(api.predict_via_api, text=titre)
            if ok:
                st.success(f"Catégorie prédite : **{res.get('category')}**")
                st.caption(
                    f"Code {res.get('prdtypecode')} · confiance {res.get('confidence')} · "
                    f"modèle {res.get('model')} · latence {res.get('latency_ms')} ms"
                )
                if res.get("low_confidence"):
                    st.warning("Prédiction peu sûre : un modérateur devrait la vérifier.")
            else:
                st.error(res)
        if col_b.button("Quel modèle est en production ?", disabled=not en_ligne, key="p9_modele"):
            ok, res = _essayer(api.model_info)
            if ok:
                st.json(res)
            else:
                st.error(res)

    with onglets[1]:
        st.markdown(
            "**L'API est protégée.** Il faut un mot de passe, et seul un administrateur peut réentraîner le modèle."
        )
        col_a, col_b = st.columns(2)
        if col_a.button("/predict sans mot de passe", disabled=not en_ligne, key="p9_401"):
            ok, res = _essayer(api.probe_without_auth)
            st.info(f"Réponse : **{res}** (attendu : 401, non authentifié)" if ok else res)
        if col_b.button("/retrain avec un compte simple", disabled=not en_ligne, key="p9_403"):
            ok, res = _essayer(api.retrain_as_user)
            st.info(f"Réponse : **{res}** (attendu : 403, rôle administrateur requis)" if ok else res)

    with onglets[2]:
        st.markdown(
            "**Deux copies de l'API travaillent ensemble** derrière nginx, et une limite protège le service "
            "(30 prédictions par minute pour un compte simple)."
        )
        if st.button("Envoyer 40 prédictions d'affilée", disabled=not en_ligne, key="p9_rafale"):
            with st.spinner("Rafale en cours..."):
                ok, res = _essayer(api.burst_predict, 40)
            if ok:
                st.session_state["p9_rafale_res"] = res
            else:
                st.error(res)
        rafale = st.session_state.get("p9_rafale_res")
        if rafale:
            st.info(f"Codes HTTP : {rafale['codes']} (200 = accepté, 429 = limite atteinte)")
            st.write("Répartition entre les deux copies de l'API :")
            st.bar_chart(rafale["instances"])

    with onglets[3]:
        st.markdown(
            "**Le système se surveille tout seul** : il compare les produits reçus à ceux de l'entraînement (dérive) "
            "et mesure son vrai score grâce aux corrections des modérateurs. Les analyses se mettent à jour "
            "toutes les 2 minutes."
        )
        dossier = Path(os.getenv("REPORTS_DIR", "reports"))
        derive = _lire_json(dossier / "drift_status.json")
        perf = _lire_json(dossier / "performance_status.json")
        col_d, col_p = st.columns(2)
        with col_d:
            if not derive:
                st.caption("Dérive : pas encore d'analyse.")
            elif derive.get("status") == "waiting":
                st.info(f"Dérive : {derive.get('message')}.")
            else:
                etiquette = "Dérive détectée" if derive.get("drift_detected") else "Pas de dérive"
                st.metric(etiquette, f"{derive.get('drift_share', 0):.0%} des variables")
        with col_p:
            if not perf:
                st.caption("Score en production : pas encore d'analyse.")
            elif perf.get("status") == "waiting":
                st.info(f"Score en production : {perf.get('message')}.")
            else:
                st.metric("F1-macro en production", f"{perf.get('f1_macro', 0):.3f}")
        b1, b2, b3 = st.columns(3)
        action = None
        if b1.button("Simuler une dérive (250 produits inhabituels)", disabled=not en_ligne, key="p9_derive"):
            action = ("la dérive", api.send_drift_traffic, (250,), {})
        if b2.button("Simuler 300 retours de modérateurs", disabled=not en_ligne, key="p9_retours"):
            action = ("les retours", api.simulate_feedback, (300,), {})
        if b3.button("Simuler 300 fiches bâclées", disabled=not en_ligne, key="p9_baclees"):
            action = ("les fiches bâclées", api.simulate_feedback, (300,), {"degraded": True})
        if action:
            nom, fonction, args, kwargs = action
            with st.spinner(f"Simulation de {nom}..."):
                ok, res = _essayer(fonction, *args, **kwargs)
            if ok:
                st.info(f"Envoyé : {res}. Résultat visible à la prochaine analyse (2 minutes au plus).")
            else:
                st.error(res)
                if "holdout" in res:
                    st.caption(
                        "Il manque le fichier data/holdout/holdout.csv (les produits mis de côté)."
                    )
        l1, l2, l3 = st.columns(3)
        l1.link_button("Ouvrir Grafana (alertes)", "http://localhost:3002")
        l2.link_button("Ouvrir Prometheus", "http://localhost:9092")
        l3.link_button("Ouvrir MLflow", "http://localhost:5000")

    with onglets[4]:
        st.markdown(
            "**Le système peut se mettre à jour.** Le réentraînement lance le pipeline (2 à 3 minutes). À la fin, "
            "le même match qu'à la partie 3 décide si le nouveau modèle remplace l'ancien."
        )
        st.warning("Ceci lance un vrai réentraînement et peut changer le vrai modèle en production.")
        confirme = st.checkbox("Je veux lancer le vrai réentraînement", key="p9_confirme")
        if st.button("Lancer le réentraînement", disabled=not (en_ligne and confirme), key="p9_retrain"):
            ok, res = _essayer(api.trigger_retrain)
            if ok:
                st.success(f"Réentraînement lancé : {res}")
            else:
                st.error(res)
        z1, z2 = st.columns(2)
        z1.link_button("Suivre le pipeline (ZenML)", "http://localhost:8888")
        z2.link_button("Voir les modèles (MLflow)", "http://localhost:5000")


def render():
    metrics = _lire_metrics()
    f1_prod = float(metrics.get("f1_macro", F1_PAR_DEFAUT))

    st.title("Test du produit final")
    st.success(
        "**En deux mots :** un modèle finit toujours par vieillir, sans prévenir. "
        "Alors on le réentraîne de temps en temps. Mais on ne remplace jamais l'ancien à l'aveugle : "
        "on vérifie d'abord que le nouveau est meilleur. Et si ça tourne mal, on revient en arrière en 1 minute."
    )
    st.info(
        "**Où voir quoi :** phase 1 (scripts, API) → page 4 « Assistant Texte » · phase 2 (suivi et promotion) → "
        "cette page · phases 3 et 4 (orchestration, surveillance) → page 8."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle en production", metrics.get("model", "SVM (Linear)"))
    c2.metric("Score F1-macro", f"{f1_prod:.3f}", help="Un score entre 0 et 1. Il compte les 27 catégories à égalité, même les petites.")
    c3.metric("Entraîné le", str(metrics.get("trained_at", "—"))[:10])
    c4.metric("Produits pour l'entraînement", f"{metrics['n_train']:,}".replace(",", " ") if "n_train" in metrics else "—")

    st.subheader("1. Le principe, en 5 étapes")
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR; bgcolor="transparent";
            node [shape=box, style="rounded,filled", fillcolor="#FFF5F5", color="#BF0000", fontname="Helvetica"];
            edge [color="#555555", fontname="Helvetica", fontsize=10];
            donnees [label="1. Les données\\nvérifiées et versionnées"];
            candidat [label="2. On entraîne\\nun nouveau modèle (le candidat)"];
            porte [label="3. Le match\\ncandidat contre ancien", fillcolor="#BF0000", fontcolor="white"];
            prod [label="4. Il devient\\nle modèle de production"];
            veille [label="5. On surveille\\ndérive, score, confiance"];
            refus [label="Perdu : on garde l'ancien", fillcolor="#EEEEEE", color="#888888"];
            donnees -> candidat -> porte;
            porte -> prod [label=" gagne"];
            porte -> refus [label=" perd"];
            prod -> veille;
            veille -> donnees [label=" alerte ou rythme prévu", style=dashed, constraint=false];
        }
        """
    )
    with st.expander("Les 5 étapes en détail"):
        st.markdown(
            "1. **Les données.** On met de côté 3 % des produits, une fois pour toutes, sans jamais s'en servir pour "
            "entraîner. Ils servent d'examen. Si les données sont cassées (colonnes manquantes, doublons...), "
            "tout s'arrête. On note aussi la version des données (avec DVC).\n"
            "2. **Le candidat.** Le nouveau modèle est rangé dans `models/candidate/`. Il ne touche jamais "
            "directement au modèle en production. Tout est noté dans MLflow.\n"
            "3. **Le match.** L'ancien et le nouveau passent le même examen (les produits mis de côté). "
            "Le nouveau doit faire au moins aussi bien.\n"
            "4. **La mise en service.** L'ancien est archivé, le nouveau prend sa place. L'API s'en aperçoit "
            "toute seule en moins d'une minute, sans être coupée.\n"
            "5. **La surveillance.** C'est la phase 4, présentée par Tristan : dérive des données, "
            "score réel, confiance du modèle."
        )

    st.subheader("2. Le produit complet, en vrai")
    _produit_complet()

    st.subheader("3. Le match, avec nos vrais modèles")
    st.caption("**L'état réel** : ces chiffres sont lus dans les fichiers du projet (`models/` et `models/candidate/`). Rien n'est inventé.")
    prod_info = _lire_json("models/metrics.json")
    cand_info = _lire_json("models/candidate/metrics.json")
    col_p, col_c = st.columns(2)
    with col_p:
        st.markdown("**Modèle en production**")
        if prod_info:
            st.metric("Score F1", f"{prod_info['f1_macro']:.4f}")
            st.caption(f"Entraîné le {str(prod_info.get('trained_at', '—'))[:10]} · run MLflow {str(prod_info.get('mlflow_run_id', '—'))[:8]}")
        else:
            st.info("Aucun modèle en production pour l'instant.")
    with col_c:
        st.markdown("**Candidat (le nouveau, en attente)**")
        if cand_info:
            st.metric("Score F1", f"{cand_info['f1_macro']:.4f}")
            st.caption(f"Entraîné le {str(cand_info.get('trained_at', '—'))[:10]} · run MLflow {str(cand_info.get('mlflow_run_id', '—'))[:8]}")
        else:
            st.info("Pas de candidat : lance un réentraînement pour en obtenir un.")
    if prod_info and cand_info:
        if prod_info.get("mlflow_run_id") and prod_info.get("mlflow_run_id") == cand_info.get("mlflow_run_id"):
            st.info("Le candidat est **déjà** le modèle en production (même run MLflow) : il n'y a rien à comparer.")
        else:
            gagne, raison = _decision(prod_info["f1_macro"], cand_info["f1_macro"], 0.0)
            if gagne:
                st.success(f"Le candidat passerait en production ({raison}).")
            else:
                st.error(f"Le candidat serait refusé ({raison}).")
    st.markdown("**Les anciens modèles gardés en secours** (pour revenir en arrière) :")
    historique = _historique()
    if historique:
        st.table(historique)
    else:
        st.caption("Pas encore d'ancien modèle archivé.")
    st.markdown("---")
    st.markdown("#### À toi de jouer : le test qu'on peut rejouer")
    st.caption(
        "On travaille sur une **copie** de notre modèle, dans un dossier de démo. Le vrai modèle n'est jamais touché : "
        "on peut donc recommencer autant de fois qu'on veut. Le match est fait par le vrai script `promote_model.py`."
    )
    sandbox_prod = SANDBOX / "production"
    sandbox_cand = SANDBOX / "candidate"

    scenario = st.radio(
        "Quel nouveau modèle (candidat) veut-on tester ?",
        [
            "Un réentraînement sans nouveauté (mêmes données, mêmes réglages)",
            "Un entraînement raté (seulement 10 % des données, ~10 secondes)",
        ],
    )
    bonus = st.number_input(
        "Bonus exigé pour gagner (0 = il suffit d'être aussi bon que l'ancien)",
        min_value=0.0, max_value=0.05, value=0.0, step=0.001, format="%.3f",
    )
    b1, b2, b3, b4 = st.columns(4)
    action = None
    if b1.button("1. Remettre à zéro"):
        action = ("Remise à zéro", ["scripts/demo_soutenance.py", "reset"], 120)
    if b2.button("2. Préparer le candidat"):
        if scenario.startswith("Un réentraînement"):
            action = ("Préparation du candidat", ["scripts/demo_soutenance.py", "identique"], 120)
        else:
            action = ("Préparation du candidat", ["scripts/demo_soutenance.py", "leger", "--fraction", "0.1"], 300)
    if b3.button("3. Lancer le match"):
        action = (
            "Le match",
            ["scripts/promote_model.py", "--candidate-dir", str(sandbox_cand),
             "--production-dir", str(sandbox_prod), "--min-gain", str(bonus)],
            180,
        )
    if b4.button("4. Revenir en arrière"):
        action = ("Retour en arrière", ["scripts/demo_soutenance.py", "retour"], 120)

    if action:
        titre, commande, delai = action
        with st.spinner(f"{titre}..."):
            try:
                env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
                env["PYTHONIOENCODING"] = "utf-8"
                res = subprocess.run(
                    [sys.executable] + commande, capture_output=True, text=True,
                    encoding="utf-8", env=env, timeout=delai,
                )
                st.session_state["demo_log"] = (titre, (res.stdout + res.stderr).strip(), res.returncode)
            except (OSError, subprocess.TimeoutExpired) as exc:
                st.session_state["demo_log"] = (titre, f"Impossible de lancer le script : {exc}", 1)

    log = st.session_state.get("demo_log")
    if log:
        titre, texte, code = log
        st.markdown(f"**Résultat : {titre}**")
        st.code(texte or "(aucune sortie)", language="text")
        if titre == "Le match":
            if code == 2:
                st.error("Le candidat a perdu : le modèle en production ne change pas.")
            elif code == 0 and "déjà" in texte:
                st.info("Ce candidat est déjà en production : rien à faire.")
            elif code == 0:
                st.success("Le candidat a gagné : l'ancien est archivé et le nouveau prend sa place.")

    st.markdown("**Où en est la démo ?**")
    demo_prod = _lire_json(sandbox_prod / "metrics.json")
    demo_cand = _lire_json(sandbox_cand / "metrics.json")
    if not demo_prod:
        st.caption("La démo n'est pas encore prête : clique sur « 1. Remettre à zéro ».")
    else:
        d1, d2 = st.columns(2)
        d1.metric("Modèle en production (démo)", f"{demo_prod['f1_macro']:.4f}")
        d1.caption(f"Entraîné le {str(demo_prod.get('trained_at', '—'))[:19].replace('T', ' ')}")
        if demo_cand:
            d2.metric("Candidat en attente", f"{demo_cand['f1_macro']:.4f}")
            d2.caption(f"Entraîné le {str(demo_cand.get('trained_at', '—'))[:19].replace('T', ' ')}")
        else:
            d2.caption("Pas encore de candidat : clique sur « 2. Préparer le candidat ».")
        archives = sorted((sandbox_prod / "archive").glob("*"), reverse=True) if (sandbox_prod / "archive").exists() else []
        if archives:
            st.caption("Anciens modèles gardés dans la démo : " + ", ".join(a.name for a in archives))
    st.caption(
        "Pour essayer : (1) remettre à zéro, (2) préparer « sans nouveauté » avec un bonus de 0,002, (3) lancer le match "
        "→ refusé. Puis re-préparer avec un bonus de 0 → promu. Le bouton 4 remet l'ancien modèle."
    )

    st.subheader("4. Et si ça se passe mal ?")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.info(
            "**Le plus simple : dans MLflow**\n\n"
            "On ouvre l'onglet *Models*, on prend `rakuten-text-svm`, et on remet l'étiquette `production` "
            "sur la version d'avant. En moins d'une minute, l'API la recharge."
        )
    with col_r2:
        st.info(
            "**Le plan B : la copie de secours**\n\n"
            "À chaque changement, l'ancien modèle est gardé dans `models/archive/<date>/`. "
            "On le recopie dans `models/` et c'est réglé."
        )
    st.markdown(
        "**Les garde-fous :**\n"
        "- Le candidat doit être complet (3 fichiers + `metrics.json`, écrit en dernier), sinon on s'arrête.\n"
        "- Les fichiers sont copiés sous un nom temporaire puis renommés d'un coup : l'API ne lit jamais un fichier à moitié copié.\n"
        "- Un seul réentraînement à la fois (un verrou partagé dans Redis).\n"
        "- Les deux modèles passent exactement le même examen : la comparaison est juste.\n"
        "- Tout est noté dans MLflow (version des données, scores, décision)."
    )

    st.subheader("5. Données personnelles et éthique")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(
            "**Les données**\n\n"
            "- On traite des **fiches produit**, pas des profils de clients.\n"
            "- Le texte envoyé à l'API est gardé dans un journal (Redis) : les **5 000 dernières prédictions** "
            "au maximum, corrigeables pendant **7 jours**.\n"
            "- Dans ce journal, on ne garde aucun identifiant d'utilisateur.\n"
            "- À surveiller : le champ texte est libre. En vrai, il faudrait le filtrer ou le vider régulièrement."
        )
    with col_e2:
        st.markdown(
            "**Les biais et la transparence**\n\n"
            "- Les 27 catégories sont **très inégales**. C'est pour ça qu'on suit le F1-macro et pas seulement "
            "le pourcentage de bonnes réponses.\n"
            "- Chaque prédiction a un **niveau de confiance**. Quand il est bas, on le dit et un modérateur vérifie.\n"
            "- Un humain reste dans la boucle : ses corrections (`/feedback`) mesurent le vrai score en production."
        )

    st.subheader("6. Ce qu'on referait autrement")
    st.warning(
        "**Ce qu'on a appris sur nos deux phases**\n\n"
        "- **Comparer sur les mêmes produits dès le début.** Notre ancien modèle avait été entraîné sur les produits "
        "mis de côté : il y obtenait 0,873, un score gonflé. Il a fallu corriger.\n"
        "- **Tester l'installation sur une machine neuve.** Sur mon PC, une image Docker du projet n'a pas pu être téléchargée.\n"
        "- **Travailler par pull requests dès le départ** et partager le stockage des données dans le groupe.\n"
        "- **Décider avec plusieurs critères**, pas seulement le F1-macro (par exemple le score de chaque petite catégorie)."
    )

    st.subheader("7. Où voir chaque critère de la grille")
    st.table(
        {
            "Critère": [
                "Cas d'usage métier",
                "Documentation technique",
                "Données propres et versionnées",
                "Choix et validation du modèle",
                "Résultats reproductibles",
                "Code sur GitHub",
                "Conteneurs et orchestration",
                "Serving (API)",
                "Monitoring système et ML",
                "Détection de dérive",
                "Infrastructure scalable (Kubernetes)",
                "Sécurité des API",
                "RGPD et éthique",
                "Stratégie de déploiement",
                "Collaboration d'équipe",
                "Suivi d'expériences (MLflow)",
                "Plan de maintenance",
            ],
            "État": [
                "Couvert", "Couvert", "Couvert", "Couvert", "Partiel", "Couvert", "Couvert", "Couvert", "Couvert",
                "Couvert", "Non couvert", "Couvert", "Couvert", "Couvert", "Partiel", "Couvert", "Couvert",
            ],
            "Où le voir": [
                "Pages 1 et 2",
                "README, ARCHITECTURE.md, docs/STRATEGIE_REENTRAINEMENT.md (slides à faire)",
                "Page 8, partie 3 (validation + DVC)",
                "Pages 2 et 3",
                "MLflow + DVC. Reste à partager le stockage des données dans le groupe",
                "Le dépôt GitHub",
                "Page 8 : Docker Compose (14 services) + pipeline ZenML",
                "Page 9, partie 2 (onglet Le service) et page 8 : FastAPI derrière nginx",
                "Page 8, partie 4, et Grafana / Prometheus",
                "Page 9, partie 2 (onglet Surveillance) et page 8, partie 3",
                "Seulement 2 copies de l'API derrière nginx (page 9, partie 2, onglet Montée en charge). Kubernetes = la suite",
                "Page 9, partie 2 (onglet Sécurité) et page 8, partie 1",
                "Page 9, partie 5",
                "Page 9, parties 1, 3 et 4 + docs/STRATEGIE_REENTRAINEMENT.md",
                "Historique Git. Reste à ajouter les pull requests et la répartition des tâches",
                "Page 9 (traçabilité) + MLflow",
                "Diapos 7, 8 et 13 (scheduler, alertes, perspectives) + docs/STRATEGIE_REENTRAINEMENT.md",
            ],
        }
    )
