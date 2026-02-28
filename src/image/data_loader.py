"""
Chargement des données pour la classification par images.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import warnings
warnings.filterwarnings("ignore")


def load_image_classification_data(
    data_dir: Path,
    image_train_dir: Path,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Charge le dataset pour la classification par images : (image_path, prdtypecode).

    Parameters
    ----------
    data_dir : Path
        Dossier contenant X_train_update.csv et Y_train.csv
    image_train_dir : Path
        Dossier des images (noms : image_{imageid}_product_{productid}.[ext])
    root : Path, optional
        Racine du projet. Si fourni, les chemins d'images sont stockés en relatif.

    Returns
    -------
    pd.DataFrame
        Colonnes : image_path, productid, imageid, prdtypecode
    """
    data_dir = Path(data_dir)
    image_train_dir = Path(image_train_dir)
    root_resolved = Path(root).resolve() if root is not None else None

    X_train = pd.read_csv(data_dir / "X_train_update.csv", index_col=0)
    y_train = pd.read_csv(data_dir / "Y_train.csv", index_col=0)

    # Construire la table des chemins d'images
    image_data = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for f in image_train_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in image_exts:
            continue
        try:
            name = f.stem
            parts = name.split("_")
            if len(parts) >= 4 and parts[0].lower() == "image" and parts[2].lower() == "product":
                imageid = int(parts[1])
                productid = int(parts[3])
                if root_resolved is not None:
                    path_str = str(f.resolve().relative_to(root_resolved))
                else:
                    path_str = str(f.resolve())
                image_data.append({"imageid": imageid, "productid": productid, "image_path": path_str})
        except (ValueError, IndexError):
            continue

    image_df = pd.DataFrame(image_data)
    X_with_images = X_train.merge(image_df, on=["imageid", "productid"], how="inner")
    y_aligned = y_train.loc[X_with_images.index]

    df = pd.DataFrame({
        "image_path": X_with_images["image_path"].values,
        "productid": X_with_images["productid"].values,
        "imageid": X_with_images["imageid"].values,
        "prdtypecode": y_aligned["prdtypecode"].values,
    })
    return df


def create_train_val_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train / validation stratifié par prdtypecode."""
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=1 - train_size,
        random_state=random_state,
        stratify=df["prdtypecode"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
