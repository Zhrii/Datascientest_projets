"""
Collecte : charge les données brutes (X_train, Y_train) et produit un fichier fusionné

"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import load_data, save_data


def collecter(data_dir: str, output: str) -> None:
    X_train, y_train, X_test = load_data(data_dir=data_dir)

    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train ({len(X_train)} lignes) et y_train ({len(y_train)} lignes) "
            "n'ont pas la même taille : vérifiez les fichiers sources."
        )

    df = X_train.reset_index(drop=True).copy()
    df["prdtypecode"] = y_train["prdtypecode"].reset_index(drop=True)

    save_data(df, output)
    print(f"{len(df):,} produits collectés -> {output}")


def main():
    parser = argparse.ArgumentParser(description="Collecte des données brutes Rakuten")
    parser.add_argument("--data-dir", default="data brut", help="Dossier contenant les CSV bruts")
    parser.add_argument("--output", default="data/raw/merged.csv", help="Fichier de sortie")
    args = parser.parse_args()
    collecter(args.data_dir, args.output)


if __name__ == "__main__":
    main()
