import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

SANDBOX = Path(tempfile.gettempdir()) / "rakuten_demo_soutenance"
FICHIERS = ["tfidf_vectorizer.pkl", "svm_text_model.pkl", "label_encoder.pkl", "metrics.json"]


def vider(dossier):
    if dossier.exists():
        shutil.rmtree(dossier)
    dossier.mkdir(parents=True)


def copier(source, cible):
    for nom in FICHIERS:
        shutil.copy2(source / nom, cible / nom)


def reset(args):
    source = Path(args.source)
    manque = [n for n in FICHIERS if not (source / n).exists()]
    if manque:
        print(f"Il manque {', '.join(manque)} dans {source}/ : impossible de copier le modèle.")
        return 1
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    (SANDBOX / "production").mkdir(parents=True)
    copier(source, SANDBOX / "production")
    info = json.loads((SANDBOX / "production" / "metrics.json").read_text(encoding="utf-8"))
    print("Démo remise à zéro.")
    print(f"   Le modèle de démo est une copie du vrai modèle (F1-macro = {info['f1_macro']:.4f}).")
    print("   Le vrai dossier models/ n'est pas touché.")
    return 0


def _production_ou_erreur():
    production = SANDBOX / "production"
    if not (production / "metrics.json").exists():
        print("La démo n'est pas prête : commence par la remettre à zéro (étape 1).")
        return None
    return production


def identique(args):
    production = _production_ou_erreur()
    if production is None:
        return 1
    candidat = SANDBOX / "candidate"
    vider(candidat)
    copier(production, candidat)
    maintenant = datetime.now(timezone.utc)
    info = json.loads((candidat / "metrics.json").read_text(encoding="utf-8"))
    info["trained_at"] = maintenant.isoformat(timespec="seconds")
    info["mlflow_run_id"] = "demo-" + maintenant.strftime("%H%M%S")
    info["mlflow_model_version"] = None
    for cle in ("promoted_at", "previous_f1_macro", "compared_on"):
        info.pop(cle, None)
    (candidat / "metrics.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Candidat prêt : un réentraînement sans nouveauté (mêmes données, mêmes réglages).")
    print(f"   Son score attendu est le même que celui de la production ({info['f1_macro']:.4f}).")
    return 0


def leger(args):
    production = _production_ou_erreur()
    if production is None:
        return 1
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from scripts.training import train

    donnees = Path(args.input)
    if not donnees.exists():
        print(f"Données introuvables : {donnees}")
        return 1
    df = pd.read_csv(donnees).dropna(subset=["text_combined", "prdtypecode"])
    _, petit = train_test_split(df, test_size=args.fraction, random_state=1, stratify=df["prdtypecode"])
    candidat = SANDBOX / "candidate"
    vider(candidat)
    echantillon = SANDBOX / "echantillon.csv"
    petit.to_csv(echantillon, index=False)
    print(f"Entraînement d'un candidat sur seulement {len(petit)} produits ({args.fraction:.0%} des données)...")
    train(str(echantillon), str(candidat), use_mlflow=False)
    echantillon.unlink()
    return 0


def retour(args):
    production = _production_ou_erreur()
    if production is None:
        return 1
    archives = sorted((production / "archive").glob("*")) if (production / "archive").exists() else []
    archives = [a for a in archives if a.is_dir() and (a / "svm_text_model.pkl").exists()]
    if not archives:
        print("ℹ Pas d'ancien modèle à restaurer : il faut d'abord qu'un candidat soit promu.")
        return 1
    derniere = archives[-1]
    present = [n for n in FICHIERS if (derniere / n).exists()]
    for nom in present:
        shutil.copy2(derniere / nom, production / nom)
    print(f"Retour en arrière : le modèle de {derniere.name} est de nouveau en production (dans la démo).")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Démo rejouable de la promotion de modèle")
    sub = parser.add_subparsers(dest="commande", required=True)
    p = sub.add_parser("reset")
    p.add_argument("--source", default="models")
    p.set_defaults(fonction=reset)
    sub.add_parser("identique").set_defaults(fonction=identique)
    p = sub.add_parser("leger")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--input", default="data/processed/processed_data.csv")
    p.set_defaults(fonction=leger)
    sub.add_parser("retour").set_defaults(fonction=retour)
    args = parser.parse_args()
    return args.fonction(args)


if __name__ == "__main__":
    sys.exit(main())
