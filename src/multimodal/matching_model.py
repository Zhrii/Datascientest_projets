"""
Modèle Dual Encoder et loss contrastive (InfoNCE) pour matching texte-image.
"""

from typing import Optional, List, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def contrastive_loss(
    text_emb: "torch.Tensor",
    image_emb: "torch.Tensor",
    temperature: float = 0.07,
) -> "torch.Tensor":
    """
    Loss InfoNCE : pour chaque texte_i, la paire positive est (texte_i, image_i).
    text_emb, image_emb : (batch_size, embedding_dim), normalisés ou non (on normalise ici).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch requis: pip install torch")
    text_emb = F.normalize(text_emb, dim=1)
    image_emb = F.normalize(image_emb, dim=1)
    logits = (text_emb @ image_emb.T) / temperature
    batch_size = text_emb.size(0)
    labels = torch.arange(batch_size, device=text_emb.device)
    return F.cross_entropy(logits, labels)


class DualEncoderModel:
    """
    Combine TextEncoder et ImageEncoder pour le matching.
    Utilisable pour entraînement (avec gradients sur les projections / backbones)
    et pour inférence (embeddings numpy).
    """

    def __init__(
        self,
        text_encoder,
        image_encoder,
        temperature: float = 0.07,
        device: Optional[str] = None,
    ):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self._device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

    def get_text_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        as_tensor: bool = False,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Encode les textes. Si as_tensor=True, retourne un tenseur (pour entraînement)."""
        emb = self.text_encoder.encode(texts, batch_size=batch_size)
        if as_tensor and TORCH_AVAILABLE:
            return torch.from_numpy(emb).float().to(self._device)
        return emb

    def get_image_embeddings(
        self,
        images: Union[np.ndarray, "torch.Tensor"],
        as_tensor: bool = False,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Encode les images. Si as_tensor=True, attend un tenseur et garde le graphe (entraînement)."""
        if TORCH_AVAILABLE and isinstance(images, torch.Tensor):
            out = self.image_encoder.forward(images)
            if not as_tensor:
                return out.detach().cpu().numpy().astype(np.float32)
            return out
        emb = self.image_encoder.encode(images)
        if as_tensor and TORCH_AVAILABLE:
            return torch.from_numpy(emb).float().to(self._device)
        return emb

    def compute_loss(
        self,
        text_emb: "torch.Tensor",
        image_emb: "torch.Tensor",
        temperature: Optional[float] = None,
    ) -> "torch.Tensor":
        """Loss InfoNCE à partir des embeddings (tenseurs)."""
        return contrastive_loss(text_emb, image_emb, temperature or self.temperature)

    def parameters(self):
        """Paramètres entraînables (projections, etc.)."""
        params = []
        if hasattr(self.image_encoder, "projection"):
            params += list(self.image_encoder.projection.parameters())
        if hasattr(self.image_encoder, "backbone"):
            params += list(self.image_encoder.backbone.parameters())
        if getattr(self.text_encoder, "_projection", None) is not None:
            params += list(self.text_encoder._projection.parameters())
        return params
