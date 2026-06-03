"""
embedder.py — Wrapper del modelo de embedding.

Responsabilidades:
  - Cargar multilingual-e5-base una sola vez (singleton)
  - Aplicar prefijos "passage: " y "query: " según el protocolo E5
  - Embeddear lotes de textos con barra de progreso
  - Normalizar vectores L2 (obligatorio para similitud coseno con FAISS IndexFlatIP)
  - Exponer dos métodos públicos:
      embed_documents(texts)  → np.ndarray shape (N, 768)  — para ingestión
      embed_query(text)       → np.ndarray shape (768,)    — para retrieval
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.settings import settings


# ---------------------------------------------------------------------------
# Singleton del modelo — se carga una sola vez al importar el módulo
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Cargando modelo '{settings.embedding_model}' en {settings.embedding_device}...")
        _model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        print(f"[Embedder] Modelo cargado. Dimensión de salida: {_model.get_sentence_embedding_dimension()}")
    return _model


# ---------------------------------------------------------------------------
# Normalización L2
# ---------------------------------------------------------------------------

def _normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normaliza cada vector a norma unitaria (L2).
    Requerido para que inner product en FAISS sea equivalente a coseno.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Evitar división por cero
    norms = np.where(norms == 0, 1e-10, norms)
    return vectors / norms


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def embed_documents(texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Embedea una lista de textos de documentos (para ingestión).
    Aplica el prefijo "passage: " requerido por el modelo E5.

    Args:
        texts:         Lista de strings a embeddear.
        batch_size:    Tamaño del lote para inferencia.
        show_progress: Mostrar barra de progreso tqdm.

    Returns:
        np.ndarray float32 de shape (len(texts), 768), normalizado L2.
    """
    model = _get_model()

    prefixed = [f"passage: {t}" for t in texts]

    all_vectors = []
    batches = range(0, len(prefixed), batch_size)

    if show_progress:
        batches = tqdm(batches, desc="Embedding documentos", unit="batch")

    for start in batches:
        batch = prefixed[start : start + batch_size]
        vecs = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,   # normalizamos manualmente abajo
        )
        all_vectors.append(vecs)

    matrix = np.vstack(all_vectors).astype(np.float32)
    return _normalize(matrix)


def embed_query(text: str) -> np.ndarray:
    """
    Embedea una query de usuario (para retrieval en tiempo real).
    Aplica el prefijo "query: " requerido por el modelo E5.

    Args:
        text: String de la query del usuario.

    Returns:
        np.ndarray float32 de shape (768,), normalizado L2.
    """
    model = _get_model()

    prefixed = f"query: {text}"
    vec = model.encode(
        prefixed,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    vec = vec.astype(np.float32).reshape(1, -1)
    return _normalize(vec)[0]


def get_embedding_dim() -> int:
    """Devuelve la dimensión del modelo cargado (768 para e5-base)."""
    return _get_model().get_sentence_embedding_dimension()
