"""Tests de la promotion d'un modèle candidat (scripts/promote_model.py)."""

import json

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
