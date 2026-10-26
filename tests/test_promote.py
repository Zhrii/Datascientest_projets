"""Tests de la promotion d'un modèle candidat (scripts/promote_model.py)."""

import json

import pandas as pd
import pytest

from scripts.promote_model import ARTIFACT_FILES, EXIT_ERROR, EXIT_PROMOTED, EXIT_REJECTED, INFO_FILE, main


def make_model(directory, f1, content="v1"):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        (directory / name).write_text(content)
    (directory / INFO_FILE).write_text(json.dumps({"f1_macro": f1, "trained_at": "2026-10-01T00:00:00+00:00"}))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return tmp_path / "models" / "candidate", tmp_path / "models"


def run(candidate, production, *extra):
    return main(["--candidate-dir", str(candidate), "--production-dir", str(production), *extra])


def test_first_model_is_promoted(dirs):
    candidate, production = dirs
    make_model(candidate, 0.78, "nouveau")
    assert run(candidate, production) == EXIT_PROMOTED
    assert (production / "svm_text_model.pkl").read_text() == "nouveau"


def test_better_candidate_is_promoted_and_old_archived(dirs):
    candidate, production = dirs
    make_model(production, 0.70, "ancien")
    make_model(candidate, 0.78, "nouveau")
    assert run(candidate, production) == EXIT_PROMOTED
    assert (production / "svm_text_model.pkl").read_text() == "nouveau"
    archives = list((production / "archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "svm_text_model.pkl").read_text() == "ancien"


def test_worse_candidate_is_rejected(dirs):
    candidate, production = dirs
    make_model(production, 0.78, "ancien")
    make_model(candidate, 0.70, "nouveau")
    assert run(candidate, production) == EXIT_REJECTED
    assert (production / "svm_text_model.pkl").read_text() == "ancien"


def test_min_gain_is_required(dirs):
    candidate, production = dirs
    make_model(production, 0.780, "ancien")
    make_model(candidate, 0.781, "nouveau")
    assert run(candidate, production, "--min-gain", "0.01") == EXIT_REJECTED


def test_dry_run_changes_nothing(dirs):
    candidate, production = dirs
    make_model(production, 0.70, "ancien")
    make_model(candidate, 0.78, "nouveau")
    assert run(candidate, production, "--dry-run") == EXIT_PROMOTED
    assert (production / "svm_text_model.pkl").read_text() == "ancien"


def test_incomplete_candidate_is_an_error(dirs):
    candidate, production = dirs
    make_model(candidate, 0.78)
    (candidate / "label_encoder.pkl").unlink()
    assert run(candidate, production) == EXIT_ERROR


def save_real_model(directory, texts, labels, stored_f1):
    """Vrai petit modèle (TF-IDF + SVM) avec les 3 fichiers attendus par promote_model."""
    import pickle

    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import LinearSVC

    from src.modeling.vectorization import TFIDFVectorizer

    directory.mkdir(parents=True, exist_ok=True)
    vectorizer = TFIDFVectorizer(min_df=1, ngram_range=(1, 1), stop_words=None)
    X = vectorizer.fit_transform(pd.Series(texts))
    encoder = LabelEncoder()
    model = LinearSVC(dual=False).fit(X, encoder.fit_transform(labels))
    vectorizer.save(directory / "tfidf_vectorizer.pkl")
    for name, obj in (("svm_text_model.pkl", model), ("label_encoder.pkl", encoder)):
        with open(directory / name, "wb") as f:
            pickle.dump(obj, f)
    (directory / INFO_FILE).write_text(json.dumps({"f1_macro": stored_f1, "trained_at": "2026-10-01T00:00:00+00:00"}))


def test_holdout_comparison_beats_stored_scores(dirs, tmp_path):
    """Le "mauvais" modèle annonce un meilleur F1 (sa propre validation) : sur les mêmes
    produits mis de côté, c'est bien le bon modèle qui l'emporte."""
    words = {10: "livre roman poche", 40: "jeu console manette", 1560: "chaise table bois"}
    texts = [f"{words[c]} {i}" for c in words for i in range(30)]
    labels = [c for c in words for _ in range(30)]
    shuffled = labels[1:] + labels[:1]  # étiquettes décalées : modèle mauvais
    candidate, production = dirs
    save_real_model(candidate, texts, labels, stored_f1=0.50)    # bon modèle, mauvais score annoncé
    save_real_model(production, texts, shuffled, stored_f1=0.99)  # mauvais modèle, bon score annoncé
    holdout = tmp_path / "holdout.csv"
    pd.DataFrame({"designation": [words[c] for c in words] * 3, "description": "",
                  "prdtypecode": list(words) * 3}).to_csv(holdout, index=False)

    assert run(candidate, production) == EXIT_REJECTED                       # scores annoncés : refusé
    assert run(candidate, production, "--holdout", str(holdout)) == EXIT_PROMOTED  # mêmes produits : promu
    assert json.loads((production / INFO_FILE).read_text())["f1_macro_holdout"] == 1.0
