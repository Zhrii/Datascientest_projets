"""
Page 9 de l'appli Streamlit : le cycle de vie du modèle, expliqué simplement.

On y montre comment on remplace le modèle sans prendre de risque :
on entraîne un nouveau modèle, on le compare à l'ancien, et on ne change que s'il est meilleur.
Cette page n'a pas besoin de l'API. Elle lit juste models/metrics.json pour les chiffres.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from scripts.demo_soutenance import SANDBOX

# si le fichier metrics.json n'existe pas, on affiche ce chiffre par défaut
F1_PAR_DEFAUT = 0.773


def _lire_metrics():
    """Lit la carte d'identité du modèle en production (écrite quand on le promeut)."""
    try:
        return json.loads(Path("models/metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _lire_json(chemin):
    """Lit un fichier metrics.json, renvoie None s'il n'existe pas."""
    try:
        return json.loads(Path(chemin).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _historique():
    """Liste les anciens modèles gardés dans models/archive (du plus récent au plus ancien)."""
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
    """Même règle que dans scripts/promote_model.py : candidat >= production + gain minimal."""
    if f1_candidat >= f1_prod + gain_min:
        return True, f"{f1_candidat:.4f} ≥ {f1_prod:.4f} + {gain_min:.4f}"
    return False, f"{f1_candidat:.4f} < {f1_prod:.4f} + {gain_min:.4f}"


def render():
    metrics = _lire_metrics()
    f1_prod = float(metrics.get("f1_macro", F1_PAR_DEFAUT))

    st.title("Cycle de vie du modèle")
    st.success(
        "**En deux mots :** un modèle finit toujours par vieillir, sans prévenir. "
        "Alors on le réentraîne de temps en temps. Mais on ne remplace jamais l'ancien à l'aveugle : "
        "on vérifie d'abord que le nouveau est meilleur. Et si ça tourne mal, on revient en arrière en 1 minute."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle en production", metrics.get("model", "SVM (Linear)"))
    c2.metric("Score F1-macro", f"{f1_prod:.3f}", help="Un score entre 0 et 1. Il compte les 27 catégories à égalité, même les petites.")
    c3.metric("Entraîné le", str(metrics.get("trained_at", "—"))[:10])
    c4.metric("Produits pour l'entraînement", f"{metrics['n_train']:,}".replace(",", " ") if "n_train" in metrics else "—")

    # ------------------------------------------------------------------ le cycle
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
            "5. **La surveillance.** On regarde si les produits reçus changent (dérive), si le score réel baisse "
            "(grâce aux corrections des modérateurs), et si le modèle hésite beaucoup (confiance)."
        )

    # ------------------------------------------------------------------ état réel
    st.subheader("2. Le match, avec nos vrais modèles")
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
                st.success(f"✅ Le candidat passerait en production ({raison}).")
            else:
                st.error(f"⛔ Le candidat serait refusé ({raison}).")
    st.markdown("**Les anciens modèles gardés en secours** (pour revenir en arrière) :")
    historique = _historique()
    if historique:
        st.table(historique)
    else:
        st.caption("Pas encore d'ancien modèle archivé.")
    # ---- le test rejouable (bac à sable)
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
                # on retire l'adresse MLflow : la démo ne doit rien écrire dans le vrai MLflow
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
                st.error("⛔ Le candidat a perdu : le modèle en production ne change pas.")
            elif code == 0 and "déjà" in texte:
                st.info("Ce candidat est déjà en production : rien à faire.")
            elif code == 0:
                st.success("✅ Le candidat a gagné : l'ancien est archivé et le nouveau prend sa place.")

    # ---- l'état du bac à sable
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

    # ------------------------------------------------------------------ quand
    st.subheader("3. Quand est-ce qu'on réentraîne ?")
    st.table(
        {
            "Situation": [
                "Il y a beaucoup de nouvelles données",
                "Le modèle se trompe de plus en plus",
                "Le modèle est vieux",
                "On veut le faire à la main",
            ],
            "Seuil": [
                "+10 % de produits depuis le dernier entraînement",
                "Score en production sous 0,75 (alerte Grafana)",
                "Plus de 30 jours (alerte Prometheus)",
                "Bouton admin, `POST /retrain` ou `.\\make.ps1 train`",
            ],
            "Qui s'en occupe": [
                "Le scheduler (il vérifie chaque heure)",
                "Une alerte, puis un humain décide",
                "Le scheduler",
                "L'administrateur",
            ],
        }
    )
    st.caption(
        "Quand le score baisse, on reçoit une alerte mais on ne réentraîne pas tout seul. "
        "Il faudrait d'abord ajouter les corrections des modérateurs aux données d'entraînement."
    )

    # ------------------------------------------------------------------ rollback
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

    # ------------------------------------------------------------------ éthique / RGPD
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

    # ------------------------------------------------------------------ limites
    st.subheader("6. Ce qu'on referait mieux")
    st.warning(
        "**Nos limites, honnêtement**\n\n"
        "- On décide avec **un seul score** (le F1-macro). On pourrait aussi regarder les petites catégories une par une.\n"
        "- Les produits mis de côté pour l'examen ne changent jamais. S'ils vieillissent, il faudra les renouveler.\n"
        "- Sans les corrections des modérateurs, on n'a que des signes (dérive, confiance), pas le vrai score.\n"
        "- Tout tourne sur une seule machine avec Docker Compose. Pas de cloud, et pas encore de Kubernetes."
    )
    st.info(
        "**La suite**\n\n"
        "Réutiliser les corrections des modérateurs pour réentraîner · envoyer les alertes par mail ou message "
        "(Alertmanager) · une connexion plus solide (OAuth2) · Kubernetes pour tenir une plus grosse charge."
    )

    # ------------------------------------------------------------------ grille
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
                "✅", "✅", "✅", "✅", "⚠️", "✅", "✅", "✅", "✅",
                "✅", "❌", "✅", "✅", "✅", "⚠️", "✅", "✅",
            ],
            "Où le voir": [
                "Pages 1 et 2",
                "README, ARCHITECTURE.md, docs/STRATEGIE_REENTRAINEMENT.md (slides à faire)",
                "Page 8, partie 3 (validation + DVC)",
                "Pages 2 et 3",
                "MLflow + DVC. Reste à partager le stockage des données dans le groupe",
                "Le dépôt GitHub",
                "Page 8 : Docker Compose (14 services) + pipeline ZenML",
                "Page 8 : FastAPI derrière nginx",
                "Page 8, partie 4, et Grafana / Prometheus",
                "Page 8, partie 3 (Evidently)",
                "Seulement 2 copies de l'API derrière nginx. Kubernetes = la suite",
                "Page 8, partie 1 (login, rôles, limite de requêtes)",
                "Page 9, partie 5",
                "Page 9, parties 1 à 4",
                "Historique Git. Reste à ajouter les pull requests et la répartition des tâches",
                "Page 9 (traçabilité) + MLflow",
                "Page 9, parties 3 et 6",
            ],
        }
    )
