"""
Encoders texte et image pour le matching multimodal.
"""

from typing import List, Union, Optional
import numpy as np


class TextEncoder:
    """
    Encodeur de texte via sentence-transformers (multilingue).
    Sortie : vecteurs de dimension embedding_dim (projection optionnelle).
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers requis: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name, device=device)
        self._native_dim = self.model.get_sentence_embedding_dimension()
        self.embedding_dim = embedding_dim or self._native_dim

        self._projection = None
        if self.embedding_dim != self._native_dim:
            try:
                import torch
                self._projection = torch.nn.Linear(self._native_dim, self.embedding_dim)
                self._device = next(self.model.parameters()).device
            except ImportError:
                raise ImportError("PyTorch requis pour la projection: pip install torch")

    def encode(
        self,
        texts: Union[List[str], np.ndarray],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode une liste de textes en vecteurs (N, embedding_dim)."""
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        if self._projection is not None:
            import torch
            with torch.no_grad():
                x = torch.from_numpy(out).float().to(self._device)
                out = self._projection(x).cpu().numpy()
        return out.astype(np.float32)


class ImageEncoder:
    """
    Encodeur d'images : backbone ResNet50 (torchvision) + projection vers embedding_dim.
    Entrée : images (N, H, W, 3) ou tenseur (N, 3, H, W), normalisées [0,1] ou ImageNet.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = True,
        device: Optional[str] = None,
    ):
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except ImportError:
            raise ImportError("torch et torchvision requis: pip install torch torchvision")

        self.embedding_dim = embedding_dim
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone.to(self._device)
        self.projection = nn.Linear(2048, embedding_dim).to(self._device)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward PyTorch (pour entraînement avec gradients)."""
        import torch
        if x.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std
        feat = self.backbone(x)
        return self.projection(feat)

    def encode(self, images: np.ndarray) -> np.ndarray:
        """
        Encode des images (numpy in -> numpy out).
        images : (N, H, W, 3) en [0,1] ou déjà (N, 3, H, W).
        """
        import torch
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        if images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
        x = torch.from_numpy(images).float().to(self._device)
        with torch.no_grad():
            emb = self.forward(x)
        return emb.cpu().numpy().astype(np.float32)

    def eval(self) -> "ImageEncoder":
        """Mode évaluation (batch norm, etc.)."""
        self.backbone.eval()
        self.projection.eval()
        return self

    def train(self, mode: bool = True) -> "ImageEncoder":
        """Mode entraînement."""
        self.backbone.train(mode)
        self.projection.train(mode)
        return self
