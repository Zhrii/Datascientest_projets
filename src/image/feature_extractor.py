"""
Extracteur de features images pour la classification (ResNet50 pré-entraîné).
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import warnings
warnings.filterwarnings("ignore")


class ImageFeatureExtractor:
    """
    Extrait des features (2048-dim) à partir d'images via ResNet50 (backbone ImageNet).
    Les features sont utilisables avec les classificateurs sklearn (SVM, XGBoost, etc.).
    """

    def __init__(
        self,
        output_dim: int = 2048,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        output_dim : int
            Dimension des features (2048 pour ResNet50 backbone, ou 256 si projection).
        device : str, optional
            'cuda' ou 'cpu'. Par défaut : GPU si disponible.
        """
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except ImportError:
            raise ImportError("torch et torchvision requis: pip install torch torchvision")

        self.output_dim = output_dim
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._image_size = (224, 224)

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone.to(self._device)
        self.backbone.eval()

        if output_dim != 2048:
            self._projection = nn.Linear(2048, output_dim).to(self._device)
        else:
            self._projection = None

    def _load_image(
        self,
        image_path: Union[str, Path],
        base_dir: Optional[Path] = None,
    ) -> np.ndarray:
        """Charge une image en array (H, W, 3) normalisé [0,1]."""
        try:
            from src.multimodal.data_loader import load_image
        except ImportError:
            from ..multimodal.data_loader import load_image
        return load_image(str(image_path), size=self._image_size, base_dir=base_dir)

    def _preprocess_batch(self, images: np.ndarray) -> "torch.Tensor":
        """Convertit batch (N, H, W, 3) en tenseur (N, 3, H, W) avec normalisation ImageNet."""
        import torch
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        if images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
        x = torch.from_numpy(images).float().to(self._device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def extract_from_paths(
        self,
        image_paths: List[str],
        base_dir: Optional[Path] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extrait les features pour une liste de chemins d'images.

        Parameters
        ----------
        image_paths : List[str]
            Chemins vers les images (relatifs à base_dir si fourni)
        base_dir : Path, optional
            Dossier de base pour les chemins relatifs
        batch_size : int
            Taille des batches pour l'extraction
        show_progress : bool
            Afficher une barre de progression

        Returns
        -------
        np.ndarray
            Features de shape (N, output_dim)
        """
        import torch
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x

        all_features = []
        base_dir = Path(base_dir) if base_dir else None

        it = range(0, len(image_paths), batch_size)
        if show_progress:
            it = tqdm(it, desc="Extraction des features", unit="batch")

        for i in it:
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            for p in batch_paths:
                try:
                    img = self._load_image(p, base_dir)
                    batch_images.append(img)
                except Exception:
                    batch_images.append(np.zeros((*self._image_size, 3), dtype=np.float32))
            batch_arr = np.stack(batch_images)
            x = self._preprocess_batch(batch_arr)

            with torch.no_grad():
                feat = self.backbone(x)
                if self._projection is not None:
                    feat = self._projection(feat)
                all_features.append(feat.cpu().numpy())

        return np.vstack(all_features).astype(np.float32)

    def extract_from_array(self, images: np.ndarray) -> np.ndarray:
        """
        Extrait les features à partir d'un array d'images (N, H, W, 3) en [0,1].
        """
        import torch
        x = self._preprocess_batch(images)
        with torch.no_grad():
            feat = self.backbone(x)
            if self._projection is not None:
                feat = self._projection(feat)
        return feat.cpu().numpy().astype(np.float32)
