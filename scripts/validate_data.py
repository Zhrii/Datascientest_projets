"""
validate_data.py : contrôle qualité des données brutes avant l'entraînement

Si un contrôle bloquant échoue, le pipeline s'arrête : on n'entraîne pas
(et on ne met donc rien en production) sur des données cassées.

Contrôles :
  bloquants
    - colonnes attendues présentes
    - assez de produits (MIN_ROWS)
    - toutes les catégories sont des codes Rakuten connus
    - les 27 catégories sont présentes
    - désignation vide : au plus MAX_EMPTY_DESIGNATION
    - productid en double : au plus MAX_DUPLICATE_IDS
  avertissements (n'arrêtent rien, mais sont tracés)
    - description manquante : au-delà de WARN_MISSING_DESCRIPTION
    - catégorie rare : moins de MIN_PER_CLASS produits

Pour lancer :
    python scripts/validate_data.py --input data/raw/merged.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.inference.categories import DICT_CATEGORIES

REQUIRED_COLUMNS = ["designation", "description", "productid", "imageid", "prdtypecode"]
KNOWN_CLASSES = set(DICT_CATEGORIES)
MIN_ROWS = 1000
MAX_EMPTY_DESIGNATION = 0.01
MAX_DUPLICATE_IDS = 0.01
WARN_MISSING_DESCRIPTION = 0.50
MIN_PER_CLASS = 50


class DataValidationError(ValueError):
    """Les données ne passent pas les contrôles bloquants."""


def validate(df: pd.DataFrame) -> dict:
    """Renvoie le rapport de qualité ; lève DataValidationError si un contrôle bloquant échoue."""
    checks, warnings = {}, []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks["required_columns"] = {"ok": not missing_cols, "missing": missing_cols}
    if missing_cols:
        report = {"ok": False, "n_rows": len(df), "checks": checks, "warnings": warnings}
        raise DataValidationError(f"Colonnes manquantes : {missing_cols}", report)

    n = len(df)
    checks["min_rows"] = {"ok": n >= MIN_ROWS, "value": n, "min": MIN_ROWS}

    classes = set(pd.to_numeric(df["prdtypecode"], errors="coerce").dropna().astype(int))
    unknown = sorted(classes - KNOWN_CLASSES)
    checks["known_classes"] = {"ok": bool(not unknown and df["prdtypecode"].notna().all()), "unknown": unknown,
                               "missing_labels": int(df["prdtypecode"].isna().sum())}
    absent = sorted(KNOWN_CLASSES - classes)
    checks["all_classes_present"] = {"ok": not absent, "n_classes": len(classes), "absent": absent}

    empty_designation = float((df["designation"].fillna("").str.strip() == "").mean()) if n else 1.0
    checks["empty_designation"] = {"ok": empty_designation <= MAX_EMPTY_DESIGNATION,
                                   "rate": round(empty_designation, 4), "max": MAX_EMPTY_DESIGNATION}

    duplicate_ids = float(df["productid"].duplicated().mean()) if n else 0.0
    checks["duplicate_productid"] = {"ok": duplicate_ids <= MAX_DUPLICATE_IDS,
                                     "rate": round(duplicate_ids, 4), "max": MAX_DUPLICATE_IDS}

    missing_description = float(df["description"].isna().mean()) if n else 0.0
    if missing_description > WARN_MISSING_DESCRIPTION:
        warnings.append(f"{missing_description:.0%} de descriptions manquantes")
    counts = df["prdtypecode"].value_counts()
    rare = {int(k): int(v) for k, v in counts[counts < MIN_PER_CLASS].items()}
    if rare:
        warnings.append(f"catégories rares (< {MIN_PER_CLASS} produits) : {rare}")

    report = {
        "ok": all(c["ok"] for c in checks.values()),
        "n_rows": n,
        "n_classes": len(classes),
        "missing_description_rate": round(missing_description, 4),
        "min_class_count": int(counts.min()) if n else 0,
        "checks": checks,
        "warnings": warnings,
    }
    if not report["ok"]:
        failed = [name for name, c in checks.items() if not c["ok"]]
        raise DataValidationError(f"Contrôles qualité en échec : {failed}", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Contrôle qualité des données Rakuten")
    parser.add_argument("--input", default="data/raw/merged.csv")
    parser.add_argument("--report", default="data/raw/quality_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    try:
        report = validate(df)
        code = 0
        print(f"✅ Données valides ({report['n_rows']:,} produits, {report['n_classes']} catégories)")
    except DataValidationError as exc:
        report = exc.args[1]
        code = 1
        print(f"❌ {exc.args[0]}")
    for w in report["warnings"]:
        print(f"⚠️  {w}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
