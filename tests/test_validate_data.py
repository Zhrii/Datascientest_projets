"""Tests du contrôle qualité des données (scripts/validate_data.py)."""

import pandas as pd
import pytest

from scripts.validate_data import DataValidationError, KNOWN_CLASSES, validate


def make_data(n_per_class=60):
    rows = []
    for code in sorted(KNOWN_CLASSES):
        for i in range(n_per_class):
            rows.append({"designation": f"produit {code} {i}", "description": "texte",
                         "productid": f"{code}-{i}", "imageid": i, "prdtypecode": code})
    return pd.DataFrame(rows)


def failed_checks(df):
    with pytest.raises(DataValidationError) as exc:
        validate(df)
    report = exc.value.args[1]
    return [name for name, check in report["checks"].items() if not check["ok"]]


def test_clean_data_passes():
    report = validate(make_data())
    assert report["ok"] and report["n_classes"] == 27 and report["warnings"] == []


def test_missing_column_fails():
    with pytest.raises(DataValidationError):
        validate(make_data().drop(columns=["designation"]))


def test_unknown_category_fails():
    df = make_data()
    df.loc[0, "prdtypecode"] = 9999
    assert "known_classes" in failed_checks(df)


def test_missing_category_fails():
    df = make_data()
    assert "all_classes_present" in failed_checks(df[df["prdtypecode"] != 10])


def test_too_many_duplicates_fails():
    df = make_data()
    df["productid"] = "meme-id"
    assert "duplicate_productid" in failed_checks(df)


def test_empty_designations_fail():
    df = make_data()
    df.loc[: len(df) // 10, "designation"] = "  "
    assert "empty_designation" in failed_checks(df)


def test_too_few_rows_fails():
    assert "min_rows" in failed_checks(make_data(n_per_class=10))


def test_missing_descriptions_only_warn():
    df = make_data()
    df["description"] = None
    report = validate(df)
    assert report["ok"] and any("descriptions manquantes" in w for w in report["warnings"])
