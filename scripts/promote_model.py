"""
promote_model.py : le nouveau modèle est-il meilleur que celui en production ?

L'idée :
  - training.py met le nouveau modèle (le "candidat") dans models/candidate
  - ce script compare son score (F1-macro) avec celui du modèle en production
  - si le candidat est au moins aussi bon : il prend la place de l'ancien
    (l'ancien est mis de côté dans models/archive, au cas où)
  - sinon : on ne touche à rien

Code de sortie : 0 = promu | 2 = refusé | 1 = erreur

Pour lancer :
    python scripts/promote_model.py
    python scripts/promote_model.py --min-gain 0.002   # exige +0,2 point de F1
    python scripts/promote_model.py --dry-run          # simule, ne change rien
    python scripts/promote_model.py --force            # promeut quoi qu'il arrive
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_FILES = ["tfidf_vectorizer.pkl", "svm_text_model.pkl", "label_encoder.pkl"]
INFO_FILE = "metrics.json"
METRIC = "f1_macro"  # le score qu'on compare
REGISTERED_MODEL = "rakuten-text-svm"  # même nom que dans training.py
PRODUCTION_ALIAS = "production"        # alias lu par l'API dans le Model Registry

# les 3 résultats possibles du script
EXIT_PROMOTED, EXIT_ERROR, EXIT_REJECTED = 0, 1, 2


def missing_files(directory):
    """Liste les fichiers qui manquent dans un dossier modèle."""
    return [name for name in ARTIFACT_FILES + [INFO_FILE] if not (directory / name).exists()]


def load_info(directory):
    """Lit metrics.json (renvoie None s'il n'existe pas)."""
    path = directory / INFO_FILE
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def archive_production(production_dir):
    """Copie le modèle actuel dans archive/<date> pour pouvoir revenir en arrière."""
    present = [n for n in ARTIFACT_FILES + [INFO_FILE] if (production_dir / n).exists()]
    if not present:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = production_dir / "archive" / stamp
    target.mkdir(parents=True, exist_ok=True)
    for name in present:
        shutil.copy2(production_dir / name, target / name)
    return target


def install_candidate(candidate_dir, production_dir, extra_info):
    """Copie le candidat dans le dossier de production.
    On copie sous un nom temporaire puis on renomme d'un coup,
    comme ça l'API ne lit jamais un fichier à moitié copié."""
    production_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        tmp = production_dir / f".{name}.tmp"
        shutil.copy2(candidate_dir / name, tmp)
        os.replace(tmp, production_dir / name)

    # metrics.json = celui du candidat + la date de promotion et l'ancien score
    info = load_info(candidate_dir) or {}
    info.update(extra_info)
    tmp = production_dir / f".{INFO_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    os.replace(tmp, production_dir / INFO_FILE)


def tag_mlflow_run(run_id, promoted, model_version=None):
    """Met une étiquette 'promoted_to_production' sur le run dans MLflow et,
    si le candidat est promu, pose l'alias "production" sur sa version
    dans le Model Registry. Si ça rate, la promotion locale reste valable."""
    if not run_id or not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.MlflowClient()
        client.set_tag(run_id, "promoted_to_production", str(promoted).lower())
        print(f"🏷️  Run MLflow {run_id[:8]}… marqué promoted_to_production={str(promoted).lower()}")
        if promoted and model_version:
            client.set_registered_model_alias(REGISTERED_MODEL, PRODUCTION_ALIAS, str(model_version))
            print(f"📚 Model Registry : {REGISTERED_MODEL}@{PRODUCTION_ALIAS} -> version {model_version}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Impossible de mettre à jour MLflow ({type(exc).__name__}), on continue.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Promotion d'un modèle candidat en production")
    parser.add_argument("--candidate-dir", default="models/candidate")
    parser.add_argument("--production-dir", default="models")
    parser.add_argument("--min-gain", type=float, default=0.0,
                        help="gain minimal de F1 exigé (0 = on accepte l'égalité)")
    parser.add_argument("--dry-run", action="store_true", help="simule, ne change rien")
    parser.add_argument("--force", action="store_true", help="promeut sans comparer")
    args = parser.parse_args(argv)

    candidate_dir = Path(args.candidate_dir)
    production_dir = Path(args.production_dir)

    # 1) le candidat est-il complet ?
    missing = missing_files(candidate_dir)
    if missing:
        print(f"❌ Candidat incomplet dans {candidate_dir}/ : il manque {', '.join(missing)}")
        print("   Lance d'abord l'entraînement (make train, ou scripts/training.py).")
        return EXIT_ERROR

    # 2) on lit les scores
    candidate = load_info(candidate_dir)
    production = load_info(production_dir)
    cand_score = candidate[METRIC]
    print(f"📦 Candidat   : {METRIC} = {cand_score:.4f}  (entraîné le {candidate.get('trained_at', '?')})")

    if production is None:
        prod_score = None
        print("📦 Production : pas de metrics.json (premier passage), donc pas de score de référence")
    else:
        prod_score = production[METRIC]
        print(f"📦 Production : {METRIC} = {prod_score:.4f}  (entraîné le {production.get('trained_at', '?')})")

    run_id = candidate.get("mlflow_run_id")
    model_version = candidate.get("mlflow_model_version")

    # 3) si ce candidat est déjà en production (script relancé deux fois), on ne fait rien
    if production is not None and run_id and production.get("mlflow_run_id") == run_id and not args.force:
        print("ℹ️  Ce candidat est déjà le modèle en production, rien à faire.")
        return EXIT_PROMOTED

    # 4) la décision
    if args.force:
        promote, reason = True, "--force"
    elif prod_score is None:
        promote, reason = True, "pas de référence : le candidat devient la référence (l'ancien modèle est archivé)"
    elif cand_score >= prod_score + args.min_gain:
        promote, reason = True, f"{cand_score:.4f} ≥ {prod_score:.4f} + {args.min_gain}"
    else:
        promote, reason = False, f"{cand_score:.4f} < {prod_score:.4f} + {args.min_gain}"

    # 5a) refusé
    if not promote:
        print(f"⛔ Candidat REFUSÉ ({reason}). Le modèle en production ne change pas.")
        if not args.dry_run:
            tag_mlflow_run(run_id, False)
        return EXIT_REJECTED

    # 5b) simulation seulement
    if args.dry_run:
        print(f"✅ [simulation] Le candidat SERAIT promu ({reason}). Rien n'a été modifié.")
        return EXIT_PROMOTED

    # 5c) promu pour de vrai : on archive l'ancien, puis on installe le nouveau
    archived = archive_production(production_dir)
    install_candidate(candidate_dir, production_dir, {
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "previous_f1_macro": prod_score,
    })
    print(f"✅ Candidat PROMU en production ({reason}).")
    if archived:
        print(f"🗄️  Ancien modèle archivé dans {archived}/")
    print("   Redémarre l'API pour qu'elle charge le nouveau modèle : docker compose restart api")
    tag_mlflow_run(run_id, True, model_version)
    return EXIT_PROMOTED


if __name__ == "__main__":
    sys.exit(main())
