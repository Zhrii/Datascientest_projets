"""Tests du nettoyage texte (src/preprocessing) utilisé par le pipeline et l'API."""

import pandas as pd

from src.preprocessing.pipeline import PreprocessingPipeline


def run(designation, description):
    pipe = PreprocessingPipeline(combine_text=True, create_features=False)
    df = pd.DataFrame({"designation": [designation], "description": [description]})
    return pipe.fit_transform(df)["text_combined"].iloc[0]


def test_html_is_removed_from_description():
    # seul le champ description contient du HTML dans les données Rakuten
    text = run("Chaise scandinave", "<p>Pieds en <b>h&ecirc;tre</b></p>")
    assert "<" not in text and ">" not in text
    assert "hêtre" in text


def test_text_is_lowercased_and_combined():
    text = run("CHAISE Bois", "Design NORDIQUE")
    assert "chaise" in text and "nordique" in text


def test_missing_description_keeps_designation():
    text = run("Coussin velours", None)
    assert "coussin velours" in text
