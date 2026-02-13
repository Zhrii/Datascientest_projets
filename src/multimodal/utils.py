"""
Utilitaires pour la recherche et la similarité (matching texte-image).
"""

from typing import List, Tuple, Optional
import numpy as np


def compute_similarity(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
) -> np.ndarray:
    """
    Similarité cosinus entre un vecteur texte et des vecteurs image.
    text_emb : (D,) ou (1, D)
    image_emb : (N, D)
    Retourne : (N,) ou (1, N)
    """
    if text_emb.ndim == 1:
        text_emb = text_emb.reshape(1, -1)
    n = np.linalg.norm(text_emb, axis=1, keepdims=True)
    n[n == 0] = 1
    text_n = text_emb / n
    ni = np.linalg.norm(image_emb, axis=1, keepdims=True)
    ni[ni == 0] = 1
    image_n = image_emb / ni
    return (text_n @ image_n.T).squeeze()


def find_matching_images(
    text: str,
    text_encoder,
    image_embeddings: np.ndarray,
    image_paths: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Pour un texte, retourne les top_k images les plus proches.
    image_embeddings : (N, D), image_paths : liste de N chemins.
    Retourne : [(path, score), ...]
    """
    text_emb = text_encoder.encode([text])
    sim = compute_similarity(text_emb, image_embeddings)
    if sim.ndim > 1:
        sim = sim.squeeze()
    idx = np.argsort(sim)[::-1][:top_k]
    return [(image_paths[i], float(sim[i])) for i in idx]


def find_matching_texts(
    image_emb: np.ndarray,
    text_embeddings: np.ndarray,
    text_list: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Pour une image (embedding), retourne les top_k textes les plus proches.
    image_emb : (D,), text_embeddings : (N, D), text_list : N textes.
    Retourne : [(texte, score), ...]
    """
    if image_emb.ndim == 1:
        image_emb = image_emb.reshape(1, -1)
    sim = compute_similarity(image_emb, text_embeddings)
    if sim.ndim > 1:
        sim = sim.squeeze()
    idx = np.argsort(sim)[::-1][:top_k]
    return [(text_list[i], float(sim[i])) for i in idx]


def recall_at_k(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    k_values: List[int] = [1, 5, 10],
) -> dict:
    """
    Recall@K : pour chaque requête, la vraie image a le même id (productid ou index).
    query_embeddings : (Q, D), gallery_embeddings : (G, D)
    query_ids, gallery_ids : identifiants (ex. productid) pour matcher.
    Retourne : {"recall@1": float, "recall@5": float, ...}
    """
    q_n = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    g_n = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    sim = q_n @ g_n.T
    results = {}
    for k in k_values:
        topk = np.argsort(-sim, axis=1)[:, :k]
        hits = 0
        for i in range(len(query_ids)):
            topk_ids = gallery_ids[topk[i]]
            if query_ids[i] in topk_ids:
                hits += 1
        results[f"recall@{k}"] = hits / len(query_ids) if query_ids.size else 0.0
    return results
