"""
store.py — Constructor y lector del índice FAISS.

Responsabilidades:
  - Crear el índice FAISS (IndexFlatIP, 768 dims)
  - Agregar vectores + metadata en lotes
  - Persistir el índice y la metadata en disco
  - Cargar un índice existente desde disco
  - Ejecutar búsqueda por similitud coseno (top-k)
  - Exponer build_info para auditoría

Archivos que gestiona en faiss_store/:
  index.faiss      ← vectores binarios
  metadata.pkl     ← dict { int_id → Document.metadata + text }
  build_info.json  ← auditoría: modelo, fecha, total de vectores
"""

import os
import json
import pickle
import time
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

from src.ingestion.formatters import Document
from src.ingestion.embedder import embed_documents, get_embedding_dim
from src.settings import settings


# ---------------------------------------------------------------------------
# Rutas de los archivos en disco
# ---------------------------------------------------------------------------

def _index_path() -> Path:
    return settings.faiss_store_path / "index.faiss"

def _metadata_path() -> Path:
    return settings.faiss_store_path / "metadata.pkl"

def _build_info_path() -> Path:
    return settings.faiss_store_path / "build_info.json"


# ---------------------------------------------------------------------------
# FAISSStore — clase principal
# ---------------------------------------------------------------------------

class FAISSStore:
    """
    Encapsula el índice FAISS y su metadata asociada.

    Uso en ingestión:
        store = FAISSStore.create_empty()
        store.add_documents(chunks)
        store.save()

    Uso en retrieval:
        store = FAISSStore.load()
        results = store.search("¿Qué es la preeclampsia?", k=5)
    """

    def __init__(self, index: faiss.IndexFlatIP, metadata: Dict[int, dict]):
        self.index    = index
        self.metadata = metadata          # { faiss_id (int) → {text, chunk_id, source, ...} }
        self._dim     = index.d

    # ------------------------------------------------------------------
    # Constructores
    # ------------------------------------------------------------------

    @classmethod
    def create_empty(cls) -> "FAISSStore":
        """Crea un índice vacío listo para recibir vectores."""
        dim   = get_embedding_dim()
        index = faiss.IndexFlatIP(dim)
        print(f"[FAISSStore] Índice vacío creado. Dimensión: {dim}")
        return cls(index=index, metadata={})

    @classmethod
    def load(cls) -> "FAISSStore":
        """Carga el índice y la metadata desde disco."""
        idx_path  = _index_path()
        meta_path = _metadata_path()

        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No se encontró el índice FAISS en '{settings.faiss_store_path}'. "
                "Ejecuta primero el pipeline de ingestión."
            )

        index = faiss.read_index(str(idx_path))

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        print(f"[FAISSStore] Índice cargado: {index.ntotal:,} vectores, dim={index.d}")
        return cls(index=index, metadata=metadata)

    # ------------------------------------------------------------------
    # Ingestión
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document], batch_size: int = 64) -> int:
        """
        Embedea y agrega una lista de Documents al índice.

        Returns:
            Número de documentos agregados en esta llamada.
        """
        if not documents:
            return 0

        texts = [doc.text for doc in documents]

        # Embeddear en lotes
        vectors = embed_documents(texts, batch_size=batch_size, show_progress=True)

        # El ID en FAISS es la posición secuencial desde el total actual
        start_id = self.index.ntotal

        # Agregar vectores al índice
        self.index.add(vectors)

        # Guardar metadata por cada vector
        for i, doc in enumerate(documents):
            faiss_id = start_id + i
            self.metadata[faiss_id] = {
                **doc.metadata,
                "text": doc.text,
            }

        added = len(documents)
        print(f"[FAISSStore] +{added:,} documentos | Total en índice: {self.index.ntotal:,}")
        return added

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, embedding_model: str = None) -> None:
        """Persiste el índice y la metadata en faiss_store/."""
        store_path = settings.faiss_store_path
        store_path.mkdir(parents=True, exist_ok=True)

        # Guardar índice FAISS
        faiss.write_index(self.index, str(_index_path()))

        # Guardar metadata
        with open(_metadata_path(), "wb") as f:
            pickle.dump(self.metadata, f)

        # Guardar build_info
        build_info = {
            "embedding_model": embedding_model or settings.embedding_model,
            "faiss_index_type": "IndexFlatIP",
            "dimension":        self.index.d,
            "total_vectors":    self.index.ntotal,
            "saved_at":         datetime.utcnow().isoformat() + "Z",
        }
        with open(_build_info_path(), "w", encoding="utf-8") as f:
            json.dump(build_info, f, indent=2)

        print(f"[FAISSStore] Guardado en '{store_path}'")
        print(f"  index.faiss  : {_index_path().stat().st_size / 1e6:.1f} MB")
        print(f"  metadata.pkl : {_metadata_path().stat().st_size / 1e6:.1f} MB")
        print(f"  Total vectores: {self.index.ntotal:,}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Busca los k fragmentos más relevantes para una query.

        Args:
            query: Texto de la pregunta del usuario.
            k:     Número de resultados (default: settings.rag_top_k).

        Returns:
            Lista de dicts con keys: text, score, source_dataset,
            language, doc_id, chunk_id, y resto de metadata.
        """
        from src.ingestion.embedder import embed_query

        if k is None:
            k = settings.rag_top_k

        if self.index.ntotal == 0:
            return []

        # Embeddear la query
        q_vec = embed_query(query).reshape(1, -1)

        # Buscar en FAISS
        k_actual = min(k, self.index.ntotal)
        scores, ids = self.index.search(q_vec, k_actual)

        results = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:          # FAISS devuelve -1 si no hay suficientes vectores
                continue
            meta = self.metadata.get(int(faiss_id), {})
            results.append({
                "score":      float(score),
                **meta,
            })

        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return self.index.ntotal

    def build_info(self) -> dict:
        """Lee y devuelve el build_info.json si existe."""
        p = _build_info_path()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}
