"""
Export du pipeline Fusion (TF-IDF + SVD + CLIP img/txt + SVM)
vers models/ pour FastAPI / Streamlit.

Usage :
  set FUSION_MAX_ROWS=4000
  python scripts/export_fusion.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.vectorization import TFIDFVectorizer
from src.multimodal.data_loader import load_text_image_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_fusion")

MODELS_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
DATA_BRUT = ROOT / "data brut"
IMAGE_DIR = ROOT / "data" / "processed" / "image_train"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_clip(texts, image_paths, model, preprocess, tokenizer, batch_size: int = 32):
    img_embs = []
    txt_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i : i + batch_size])
            batch_paths = list(image_paths[i : i + batch_size])

            imgs = []
            for p in batch_paths:
                path = ROOT / p if not Path(p).is_absolute() else Path(p)
                img = Image.open(path).convert("RGB")
                imgs.append(preprocess(img))
            img_tensor = torch.stack(imgs).to(DEVICE)
            clip_img = model.encode_image(img_tensor)
            clip_img = clip_img / clip_img.norm(dim=-1, keepdim=True)
            img_embs.append(clip_img.cpu().numpy().astype(np.float32))

            tokens = tokenizer(batch_texts).to(DEVICE)
            clip_txt = model.encode_text(tokens)
            clip_txt = clip_txt / clip_txt.norm(dim=-1, keepdim=True)
            txt_embs.append(clip_txt.cpu().numpy().astype(np.float32))

            if (i // batch_size) % 10 == 0:
                logger.info("CLIP %s/%s", min(i + batch_size, len(texts)), len(texts))

    return np.vstack(img_embs), np.vstack(txt_embs)


def main():
    max_rows = int(os.getenv("FUSION_MAX_ROWS", "4000"))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Dossier images introuvable: {IMAGE_DIR}")

    logger.info("Chargement des paires texte-image...")
    df = load_text_image_pairs(
        data_dir=DATA_BRUT,
        image_train_dir=IMAGE_DIR,
        combine_text=True,
        preprocess_text=True,
        root=ROOT,
    )
    logger.info("Paires disponibles: %s", len(df))

    # Échantillon stratifié pour un export raisonnable en CPU
    if len(df) > max_rows:
        frac = max_rows / len(df)
        _, df = train_test_split(
            df,
            test_size=frac,
            random_state=SEED,
            stratify=df["prdtypecode"],
        )
        df = df.reset_index(drop=True)
    logger.info("Échantillon utilisé: %s", len(df))

    le = LabelEncoder()
    y = le.fit_transform(df["prdtypecode"].astype(int))
    train_df, val_df, y_train, y_val = train_test_split(
        df, y, test_size=0.2, random_state=SEED, stratify=y
    )

    logger.info("TF-IDF + SVD...")
    tfidf = TFIDFVectorizer(max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2))
    X_tfidf_tr = tfidf.fit_transform(pd.Series(train_df["text"].values))
    X_tfidf_va = tfidf.transform(pd.Series(val_df["text"].values))
    svd = TruncatedSVD(n_components=300, random_state=SEED)
    X_tfidf_tr = normalize(svd.fit_transform(X_tfidf_tr))
    X_tfidf_va = normalize(svd.transform(X_tfidf_va))

    logger.info("Chargement CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    logger.info("Encodage CLIP train...")
    clip_img_tr, clip_txt_tr = _encode_clip(
        train_df["text"].astype(str).tolist(),
        train_df["image_path"].tolist(),
        model,
        preprocess,
        tokenizer,
    )
    logger.info("Encodage CLIP val...")
    clip_img_va, clip_txt_va = _encode_clip(
        val_df["text"].astype(str).tolist(),
        val_df["image_path"].tolist(),
        model,
        preprocess,
        tokenizer,
    )

    X_tr = normalize(np.hstack([X_tfidf_tr, clip_img_tr, clip_txt_tr]))
    X_va = normalize(np.hstack([X_tfidf_va, clip_img_va, clip_txt_va]))

    logger.info("Entraînement LinearSVC fusion (%sd)...", X_tr.shape[1])
    clf = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, dual=False, max_iter=2000)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_va)
    f1 = float(f1_score(y_val, y_pred, average="macro"))
    logger.info("F1-macro validation (échantillon)=%.4f", f1)

    tfidf.save(MODELS_DIR / "tfidf_fusion.pkl")
    with open(MODELS_DIR / "svd_fusion.pkl", "wb") as f:
        pickle.dump(svd, f)
    with open(MODELS_DIR / "fusion_svm.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / "label_encoder_fusion.pkl", "wb") as f:
        pickle.dump(le, f)

    meta = {
        "model": "fusion_svm_tfidf_clip",
        "f1_macro_sample": f1,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "dims": int(X_tr.shape[1]),
        "note": "Export accéléré sur échantillon; F1 notebook complet ~0.84",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACTS / "fusion_export_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Artefacts fusion écrits dans %s", MODELS_DIR)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
