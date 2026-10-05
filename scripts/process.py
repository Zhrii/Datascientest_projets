"""
Process : nettoyage texte (HTML, normalisation) et fusion designation + description, via le PreprocessingPipeline existant du projet.

"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.data_loader import save_data


def process(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    pipeline = PreprocessingPipeline(combine_text=True, create_features=False)
    df_processed = pipeline.fit_transform(df)

    cols = [c for c in ["productid", "text_combined", "prdtypecode"] if c in df_processed.columns]
    df_final = df_processed[cols].dropna(subset=["text_combined", "prdtypecode"])

    save_data(df_final, output_path)
    print(f"{len(df_final):,} produits traités -> {output_path}")
    print(f"   Stats preprocessing : {pipeline.get_stats()}")


def main():
    parser = argparse.ArgumentParser(description="Preprocessing texte Rakuten")
    parser.add_argument("--input", default="data/raw/merged.csv")
    parser.add_argument("--output", default="data/processed/processed_data.csv")
    args = parser.parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()
