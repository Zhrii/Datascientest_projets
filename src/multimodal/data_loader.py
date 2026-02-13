"""
Chargement des paires (texte, image) pour le matching multimodal.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import os
import warnings

warnings.filterwarnings("ignore")


def _combine_text_simple(designation: str, description: str) -> str:
    """Combine designation et description sans pipeline lourd."""
    d = str(designation).strip() if pd.notna(designation) else ""
    desc = str(description).strip() if pd.notna(description) else ""
    if d and desc:
        return f"{d} {desc}"
    return d or desc or ""


def load_text_image_pairs(
    data_dir: Path,
    image_train_dir: Path,
    combine_text: bool = True,
    preprocess_text: bool = True,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Charge les paires (texte, image) depuis les CSV et le dossier d'images.

    Parameters
    ----------
    data_dir : Path
        Dossier contenant X_train_update.csv et Y_train.csv
    image_train_dir : Path
        Dossier des images (noms : image_{imageid}_product_{productid}.jpg)
    combine_text : bool
        Combiner designation + description
    preprocess_text : bool
        Utiliser le pipeline de preprocessing (HTML, normalisation) si disponible
    root : Path, optional
        Racine du projet. Si fourni, les chemins d'images sont stockés en relatif (root).

    Returns
    -------
    pd.DataFrame
        Colonnes : text, image_path, productid, imageid, prdtypecode
    """
    data_dir = Path(data_dir)
    image_train_dir = Path(image_train_dir)
    root_resolved = Path(root).resolve() if root is not None else None

    X_train = pd.read_csv(data_dir / "X_train_update.csv", index_col=0)
    y_train = pd.read_csv(data_dir / "Y_train.csv", index_col=0)

    # Construire la table des chemins d'images (relatifs à root si fourni)
    image_data = []
    for f in image_train_dir.glob("*.jpg"):
        try:
            name = f.stem
            parts = name.split("_")
            if len(parts) >= 4 and parts[0] == "image" and parts[2] == "product":
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

    # Texte
    if preprocess_text:
        try:
            from ..preprocessing.pipeline import PreprocessingPipeline
            pipeline = PreprocessingPipeline(combine_text=True, create_features=False)
            tmp = X_with_images[["designation", "description"]].copy()
            tmp = pipeline.fit_transform(tmp)
            texts = tmp["text_combined"]
        except Exception:
            texts = X_with_images.apply(lambda r: _combine_text_simple(r["designation"], r["description"]), axis=1)
    else:
        texts = X_with_images.apply(lambda r: _combine_text_simple(r["designation"], r["description"]), axis=1)

    pairs_df = pd.DataFrame({
        "text": texts.values,
        "image_path": X_with_images["image_path"].values,
        "productid": X_with_images["productid"].values,
        "imageid": X_with_images["imageid"].values,
        "prdtypecode": y_aligned["prdtypecode"].values,
    })
    return pairs_df


def create_pairs_dataset(
    pairs_df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split paires en train / validation (stratifié par prdtypecode)."""
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        pairs_df,
        test_size=1 - train_size,
        random_state=random_state,
        stratify=pairs_df["prdtypecode"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_image(
    image_path: str,
    size: Tuple[int, int] = (224, 224),
    base_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Charge et redimensionne une image, retourne un array (H, W, 3) normalisé [0,1].

    Si image_path est relatif et base_dir est fourni, le chemin est résolu par rapport à base_dir.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL (Pillow) requis: pip install Pillow")
    p = Path(image_path)
    if not p.is_absolute() and base_dir is not None:
        image_path = Path(base_dir) / image_path
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr
