"""
Chargement des modèles et prédiction texte / image / fusion.

Routage :
- texte + image → fusion (TF-IDF + CLIP + SVM) — champion F1-macro 0.84
- texte seul    → SVM TF-IDF — F1-macro 0.78
- image seule   → ResNet50 + Logistic Regression — baseline image
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .categories import category_name

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
HF_REPO = os.getenv("HF_REPO", "nicolasN2215/ecommerce-resnet-image")
REGISTERED_TEXT_MODEL = "rakuten-text-svm"  # cf. scripts/training.py
PRODUCTION_ALIAS = "production"             # posé par scripts/promote_model.py

# F1-macro de référence (notebooks, jeu complet). Remplacés par les valeurs
# réellement mesurées sur les artefacts chargés quand elles sont disponibles.
MODEL_META = {
    "fusion": {
        "name": "fusion_svm_tfidf_clip",
        "f1_macro": 0.84,
        "description": "Late fusion TF-IDF(300) + CLIP image(512) + CLIP texte(512) + SVM Linear",
    },
    "text": {
        "name": "svm_linear_tfidf",
        "f1_macro": 0.78,
        "description": "SVM Linear sur TF-IDF (texte seul)",
    },
    "image": {
        "name": "resnet50_logistic",
        "f1_macro": 0.51,
        "description": "ResNet50 features + Logistic Regression (image seule)",
    },
}


def _resolve(*candidates: Path) -> Path:
    """Retourne le premier chemin existant, sinon le premier candidat."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _model_paths(models_dir: Path) -> Dict[str, Path]:
    return {
        "vectorizer": models_dir / "tfidf_vectorizer.pkl",
        "label_encoder": _resolve(
            models_dir / "label_encoder.pkl",
            models_dir / "label_encoder_image.pkl",
        ),
        "model_text": _resolve(
            models_dir / "svm_text_model.pkl",
            models_dir / "svm_linear_baseline_final.pkl",
            models_dir / "svm_(linear)_baseline.pkl",
            models_dir / "svm_linear_baseline_27_classes_final.pkl",
        ),
        "model_image": _resolve(
            models_dir / "image_logistic_regression_27_classes_baseline.pkl",
            models_dir / "image_logistic_regression_baseline.pkl",
        ),
        "tfidf_fusion": models_dir / "tfidf_fusion.pkl",
        "svd_fusion": models_dir / "svd_fusion.pkl",
        "fusion_svm": models_dir / "fusion_svm.pkl",
        "label_encoder_fusion": models_dir / "label_encoder_fusion.pkl",
    }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _measured_f1(models_dir: Path) -> Dict[str, float]:
    """F1-macro mesurés par les pipelines d'entraînement / export."""
    out: Dict[str, float] = {}
    # metrics.json : carte d'identité du modèle texte promu (scripts/promote_model.py)
    prod = _read_json(models_dir / "metrics.json")
    if "f1_macro" in prod:
        out["text"] = float(prod["f1_macro"])
    fusion = _read_json(ARTIFACTS_DIR / "fusion_export_meta.json")
    if "f1_macro_sample" in fusion:
        out["fusion"] = float(fusion["f1_macro_sample"])
    return out


def _ensure_hf_file(models_dir: Path, filename: str) -> Path:
    """Télécharge un artefact depuis Hugging Face s'il est absent localement."""
    local_path = models_dir / filename
    if local_path.exists():
        return local_path
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=models_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Impossible de télécharger %s: %s", filename, exc)
    return local_path


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _download_production_text_model() -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Télécharge depuis le Model Registry MLflow la version marquée "production".

    Renvoie (dossier local des fichiers, infos de la version), ou None si MLflow
    n'est pas configuré / joignable ou si aucune version n'est encore promue.
    """
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return None
    try:
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        import mlflow

        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(REGISTERED_TEXT_MODEL, PRODUCTION_ALIAS)
        dst = Path(tempfile.gettempdir()) / "rakuten_registry" / f"v{version.version}"
        dst.mkdir(parents=True, exist_ok=True)
        local = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=str(dst))
        info = {
            "source": f"mlflow-registry:{REGISTERED_TEXT_MODEL}@{PRODUCTION_ALIAS} (v{version.version})",
            "f1_macro": float(version.tags["f1_macro"]) if "f1_macro" in version.tags else None,
        }
        return Path(local), info
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model Registry indisponible, fallback fichiers locaux: %s", exc)
        return None


class ProductPredictor:
    """Charge une fois les pipelines et expose predict()."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
        self.reload()

    # ------------------------------------------------------------------ load
    def reload(self) -> None:
        """(Re)charge tous les pipelines — appelé au démarrage et après retrain."""
        self.paths = _model_paths(self.models_dir)
        self.meta = {k: dict(v) for k, v in MODEL_META.items()}
        for modality, f1 in _measured_f1(self.models_dir).items():
            self.meta[modality]["f1_macro"] = round(f1, 4)

        self.text_ready = self.image_ready = self.fusion_ready = False
        self._text: Dict[str, Any] = {}
        self._image: Dict[str, Any] = {}
        self._fusion: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}

        self._load_text()
        self._load_image()
        self._load_fusion()

    def _load_text(self) -> None:
        try:
            from src.modeling.vectorization import TFIDFVectorizer
            from src.modeling.baseline_models import BaselineModels

            # 1) modèle de production du Model Registry (MLflow / MinIO)
            registry = _download_production_text_model()
            if registry is not None:
                local_dir, info = registry
                self._text["vectorizer"] = TFIDFVectorizer.load(local_dir / "tfidf_vectorizer.pkl")
                self._text["model"] = BaselineModels.load_model(local_dir / "svm_text_model.pkl")
                self._text["encoder"] = _load_pickle(local_dir / "label_encoder.pkl")
                self.meta["text"]["source"] = info["source"]
                if info["f1_macro"] is not None:
                    self.meta["text"]["f1_macro"] = round(info["f1_macro"], 4)
                self.text_ready = True
                logger.info("Pipeline texte prêt (%s)", info["source"])
                return

            # 2) fallback : fichiers locaux de models/ (puis Hugging Face si aucun SVM)
            self.meta["text"]["source"] = f"local:{self.models_dir}"
            if not self.paths["model_text"].exists():
                _ensure_hf_file(self.models_dir, "svm_linear_baseline_27_classes_final.pkl")
                self.paths = _model_paths(self.models_dir)
            if not self.paths["vectorizer"].exists() or not self.paths["model_text"].exists():
                self._errors["text"] = (
                    "Fichiers texte manquants. Attendu: tfidf_vectorizer.pkl "
                    "+ svm_text_model.pkl dans models/. "
                    "Lancez: .\\make.ps1 train"
                )
                return
            self._text["vectorizer"] = TFIDFVectorizer.load(self.paths["vectorizer"])
            self._text["model"] = BaselineModels.load_model(self.paths["model_text"])
            encoder_path = self.paths["label_encoder"]
            self._text["encoder"] = _load_pickle(encoder_path) if encoder_path.exists() else None
            self.text_ready = True
            logger.info("Pipeline texte prêt (%s)", self.paths["vectorizer"])
        except Exception as exc:  # noqa: BLE001
            self._errors["text"] = str(exc)
            logger.exception("Chargement texte échoué")

    def _load_image(self) -> None:
        try:
            from src.image.feature_extractor import ImageFeatureExtractor
            from src.modeling.baseline_models import BaselineModels

            if not self.paths["model_image"].exists():
                _ensure_hf_file(self.models_dir, "image_logistic_regression_27_classes_baseline.pkl")
                self.paths = _model_paths(self.models_dir)
            if not self.paths["model_image"].exists():
                self._errors["image"] = "Modèle image manquant."
                return
            self._image["extractor"] = ImageFeatureExtractor(output_dim=2048)
            self._image["model"] = BaselineModels.load_model(self.paths["model_image"])
            encoder_path = self.paths["label_encoder"]
            if self._text.get("encoder") is not None:
                self._image["encoder"] = self._text["encoder"]
            else:
                self._image["encoder"] = _load_pickle(encoder_path) if encoder_path.exists() else None
            self.image_ready = True
            logger.info("Pipeline image prêt")
        except Exception as exc:  # noqa: BLE001
            self._errors["image"] = str(exc)
            logger.exception("Chargement image échoué")

    def _load_fusion(self) -> None:
        try:
            import open_clip
            from src.modeling.vectorization import TFIDFVectorizer

            needed = ["tfidf_fusion", "svd_fusion", "fusion_svm", "label_encoder_fusion"]
            missing = [self.paths[k].name for k in needed if not self.paths[k].exists()]
            if missing:
                self._errors["fusion"] = (
                    "Artefacts fusion absents: "
                    + ", ".join(missing)
                    + ". Exportez-les via .\\make.ps1 export-fusion"
                )
                return

            self._fusion["tfidf"] = TFIDFVectorizer.load(self.paths["tfidf_fusion"])
            self._fusion["svd"] = _load_pickle(self.paths["svd_fusion"])
            self._fusion["svm"] = _load_pickle(self.paths["fusion_svm"])
            self._fusion["encoder"] = _load_pickle(self.paths["label_encoder_fusion"])

            model_clip, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model_clip.eval()
            self._fusion["clip_model"] = model_clip
            self._fusion["clip_preprocess"] = preprocess
            self._fusion["clip_tokenizer"] = open_clip.get_tokenizer("ViT-B-32")
            self.fusion_ready = True
            logger.info("Pipeline fusion prêt")
        except Exception as exc:  # noqa: BLE001
            self._errors["fusion"] = str(exc)
            logger.exception("Chargement fusion échoué")

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _preprocess_text(text: str) -> str:
        try:
            from src.preprocessing.pipeline import PreprocessingPipeline

            pipe = PreprocessingPipeline(combine_text=True, create_features=False)
            tmp = pd.DataFrame({"designation": [text], "description": [""]})
            tmp = pipe.fit_transform(tmp)
            return tmp["text_combined"].iloc[0]
        except Exception:  # noqa: BLE001
            return text.strip()

    @staticmethod
    def _decode(encoder, pred_interne) -> Any:
        if encoder is None:
            return pred_interne
        return encoder.inverse_transform([pred_interne])[0]

    def _result(self, code, modality: str) -> Dict[str, Any]:
        meta = self.meta[modality]
        return {
            "prdtypecode": int(code) if str(code).isdigit() else code,
            "category": category_name(code),
            "modality": modality,
            "model": meta["name"],
            "f1_macro": meta["f1_macro"],
        }

    # ------------------------------------------------------------------ predict
    def predict_text(self, text: str) -> Dict[str, Any]:
        if not self.text_ready:
            raise RuntimeError(self._errors.get("text", "Pipeline texte indisponible"))
        cleaned = self._preprocess_text(text)
        X = self._text["vectorizer"].transform(pd.Series([cleaned]))
        pred = self._text["model"].predict(X)[0]
        code = self._decode(self._text["encoder"], pred)
        return self._result(code, "text")

    def predict_image(self, image: Union[Image.Image, str, Path, bytes]) -> Dict[str, Any]:
        if not self.image_ready:
            raise RuntimeError(self._errors.get("image", "Pipeline image indisponible"))
        img = self._to_pil(image)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.convert("RGB").save(tmp.name)
            temp_path = tmp.name
        try:
            features = self._image["extractor"].extract_from_paths([temp_path])
            pred = self._image["model"].predict(features)[0]
            code = self._decode(self._image.get("encoder"), pred)
            return self._result(code, "image")
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def predict_fusion(
        self, text: str, image: Union[Image.Image, str, Path, bytes]
    ) -> Dict[str, Any]:
        if not self.fusion_ready:
            raise RuntimeError(self._errors.get("fusion", "Pipeline fusion indisponible"))

        import torch
        from sklearn.preprocessing import normalize as sk_normalize

        cleaned = self._preprocess_text(text)
        X_tfidf = self._fusion["tfidf"].transform(pd.Series([cleaned]))
        X_tfidf_svd = self._fusion["svd"].transform(X_tfidf)
        X_tfidf_norm = sk_normalize(X_tfidf_svd)

        img = self._to_pil(image).convert("RGB")
        img_tensor = self._fusion["clip_preprocess"](img).unsqueeze(0)
        with torch.no_grad():
            clip_img = self._fusion["clip_model"].encode_image(img_tensor)
        clip_img = (clip_img / clip_img.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

        tokens = self._fusion["clip_tokenizer"]([cleaned])
        with torch.no_grad():
            clip_txt = self._fusion["clip_model"].encode_text(tokens)
        clip_txt = (clip_txt / clip_txt.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

        X_fusion = sk_normalize(np.hstack([X_tfidf_norm, clip_img, clip_txt]))
        pred = self._fusion["svm"].predict(X_fusion)[0]
        code = self._decode(self._fusion["encoder"], pred)
        return self._result(code, "fusion")

    def predict(
        self,
        text: Optional[str] = None,
        image: Optional[Union[Image.Image, str, Path, bytes]] = None,
    ) -> Dict[str, Any]:
        """Routage intelligent selon les modalités fournies."""
        has_text = bool(text and str(text).strip())
        has_image = image is not None

        if has_text and has_image:
            if self.fusion_ready:
                return self.predict_fusion(text, image)
            # Fallback : texte seul si fusion absente
            if self.text_ready:
                out = self.predict_text(text)
                out["fallback"] = "fusion_unavailable_used_text"
                return out
            raise RuntimeError("Ni fusion ni texte disponibles")

        if has_text:
            return self.predict_text(text)

        if has_image:
            return self.predict_image(image)

        raise ValueError("Fournir au moins un texte ou une image")

    @staticmethod
    def _to_pil(image: Union[Image.Image, str, Path, bytes]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, Path)):
            return Image.open(image)
        if isinstance(image, bytes):
            return Image.open(BytesIO(image))
        raise TypeError(f"Type image non supporté: {type(image)}")

    def info(self) -> Dict[str, Any]:
        return {
            "models_dir": str(self.models_dir),
            "pipelines": {
                "text": {"ready": self.text_ready, "error": self._errors.get("text"), **self.meta["text"]},
                "image": {"ready": self.image_ready, "error": self._errors.get("image"), **self.meta["image"]},
                "fusion": {"ready": self.fusion_ready, "error": self._errors.get("fusion"), **self.meta["fusion"]},
            },
            "routing": {
                "text+image": "fusion (fallback text)",
                "text": "text",
                "image": "image",
            },
        }

    def health(self) -> Tuple[bool, Dict[str, Any]]:
        any_ready = self.text_ready or self.image_ready or self.fusion_ready
        return any_ready, {
            "status": "ok" if any_ready else "degraded",
            "text_ready": self.text_ready,
            "image_ready": self.image_ready,
            "fusion_ready": self.fusion_ready,
            "errors": self._errors,
        }
