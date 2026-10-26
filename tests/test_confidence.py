"""Tests de la confiance des prédictions, du holdout et du calibrage du seuil."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from scripts.calibrate_confidence import calibrate
from scripts.holdout import HOLDOUT_SHARE, split_holdout
from src.inference.predictor import confidence_margin


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c * 3, 1, (50, 2)) for c in range(3)])
    y = np.repeat([10, 40, 1560], 50)
    return X, y


@pytest.mark.parametrize("model", [LinearSVC(dual=False), LogisticRegression(max_iter=500)])
def test_margin_agrees_with_predict(model, data):
    X, y = data
    model.fit(X, y)
    for row in X[::10]:
        pred, margin = confidence_margin(model, row.reshape(1, -1))
        assert pred == model.predict(row.reshape(1, -1))[0]
        assert margin >= 0


def test_margin_is_small_between_two_classes(data):
    X, y = data
    model = LinearSVC(dual=False).fit(X, y)
    _, clear = confidence_margin(model, np.array([[0.0, 0.0]]))
    _, between = confidence_margin(model, np.array([[1.5, 1.5]]))  # à mi-chemin de deux classes
    assert between < clear


def test_holdout_is_stable_and_disjoint():
    df = pd.DataFrame({"productid": range(20_000), "prdtypecode": 10})
    train, holdout = split_holdout(df)
    train2, holdout2 = split_holdout(df.sample(frac=1, random_state=3))
    assert set(holdout["productid"]) == set(holdout2["productid"])   # mêmes produits à chaque fois
    assert not set(train["productid"]) & set(holdout["productid"])
    assert len(holdout) / len(df) == pytest.approx(HOLDOUT_SHARE, abs=0.005)


def test_calibrate_threshold():
    margins = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 1.2, 1.5, 2.0])
    correct = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    # fenêtre de 3 : accuracy locale 0, .33, .33, .67, .67, 1, 1, 1 -> dernière <= .5 en 0.3 -> seuil 0.4
    threshold = calibrate(margins, correct, target=0.5, window=3)
    assert threshold == pytest.approx(0.4)
    assert correct[margins < threshold].mean() < 0.5      # sous le seuil : plus souvent faux que juste
    assert calibrate(np.array([0.5, 1.0]), np.array([1, 1]), window=1) == 0.0   # toujours juste : rien de peu sûr
    # un pic de chance dans les petites marges ne fixe pas le seuil trop bas
    noisy = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert calibrate(np.arange(12.0), noisy, window=1) == 7.0
