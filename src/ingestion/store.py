"""
store.py — Constructor y lector del índice FAISS.

Responsabilidades:
  - Crear el índice FAISS (IndexFlatIP, 768 dims)
  - Agregar vectores + metadata en lotes
  - Persistir el índice y la metadata en disco
  - Cargar un índice existente desde disco
  - Ejecutar búsqueda por similitud coseno (top-k)
  - Activar/desactivar documentos sin necesidad de reconstruir el índice
  - Exponer build_info para auditoría

Archivos que gestiona en faiss_store/:
  index.faiss      ← vectores binarios
  metadata.pkl     ← dict { int_id → Document.metadata + text }
  build_info.json  ← auditoría: modelo, fecha, total de vectores
"""

import json
import logging
import pickle
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List

import faiss
import numpy as np
from pathlib import Path

from src.ingestion.formatters import Document
from src.ingestion.embedder import embed_documents, get_embedding_dim
from src.settings import settings

logger = logging.getLogger(__name__)


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
# Estado "activo" de un fragmento — un solo lugar donde vive el default
# ---------------------------------------------------------------------------

def is_active(meta: dict) -> bool:
    """Un fragmento participa en el RAG salvo que esté explícitamente desactivado.

    El pipeline de ingestión NO escribe la clave 'active'; su ausencia
    significa ACTIVO. Nunca usar meta["active"] ni meta.get("active") sin
    default: descartaría el 100% del índice existente y /chat seguiría
    respondiendo 200 con un fallback genérico ("no se encontraron
    fragmentos relevantes"), sin ningún error visible.
    """
    return bool(meta.get("active", True))


# Sobre-muestreo interno para el filtro de 'active' en search(): los
# fragmentos desactivados se descartan DESPUÉS de la búsqueda en FAISS,
# así que pedimos más candidatos de los necesarios para no devolver menos
# de k. Con IndexFlatIP el costo de un k mayor es despreciable: el barrido
# exhaustivo de los N vectores domina, k solo afecta al heap de top-k.
INACTIVE_OVERFETCH = 3
INACTIVE_OVERFETCH_MAX = 500


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

    Uso en administración (panel):
        with store.write_lock():
            store.update_document_status(doc_id, active=False)
            store.save_metadata()
    """

    def __init__(self, index: faiss.IndexFlatIP, metadata: Dict[int, dict]):
        self.index    = index
        self.metadata = metadata          # { faiss_id (int) → {text, chunk_id, source, ...} }

        # Contador de mutaciones: sube con cada add_documents() o
        # update_document_status() que cambie algo. Lo consume el cache de
        # /documents para invalidarse sin depender de un TTL.
        self._mutation_seq = 0

        # index.search/index.add no son seguros de forma concurrente entre
        # sí; write_lock serializa a los escritores (incluye el chequeo de
        # duplicados del upload) sin bloquear lecturas.
        self._index_lock = threading.RLock()
        self._write_lock = threading.RLock()

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

        if index.ntotal != len(metadata):
            logger.warning(
                "Desincronización detectada al cargar el índice: %s vectores "
                "vs %s entradas de metadata. El resultado de search() puede "
                "no corresponder a los IDs esperados.",
                index.ntotal, len(metadata),
            )

        print(f"[FAISSStore] Índice cargado: {index.ntotal:,} vectores, dim={index.d}")
        return cls(index=index, metadata=metadata)

    # ------------------------------------------------------------------
    # Ingestión
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> int:
        """
        Embedea y agrega una lista de Documents al índice.

        Returns:
            Número de documentos agregados en esta llamada.
        """
        if not documents:
            return 0

        texts = [doc.text for doc in documents]

        # Embeddear en lotes (fuera del index_lock: es la parte lenta y no
        # toca self.index en absoluto).
        vectors = embed_documents(texts, batch_size=batch_size, show_progress=show_progress)

        with self._index_lock:
            # El ID en FAISS es la posición secuencial desde el total actual
            start_id = self.index.ntotal
            self.index.add(vectors)

        # Guardar metadata por cada vector
        for i, doc in enumerate(documents):
            faiss_id = start_id + i
            self.metadata[faiss_id] = {
                **doc.metadata,
                "text": doc.text,
            }

        added = len(documents)
        self._mutation_seq += 1
        print(f"[FAISSStore] +{added:,} documentos | Total en índice: {self.index.ntotal:,}")
        return added

    def rollback_append(self, start_id: int, count: int) -> None:
        """Deshace un append fallido eliminando los ÚLTIMOS `count` vectores.

        ÚNICO uso sancionado de remove_ids en este proyecto. Es seguro
        exclusivamente porque elimina un sufijo contiguo: FAISS renumera
        los vectores posteriores al eliminado, y como aquí no hay
        posteriores, ningún faiss_id sobreviviente cambia de posición.
        NUNCA llamar remove_ids con ids arbitrarios: desincronizaría
        metadata y vectores sin lanzar excepción (texto equivocado con
        score equivocado, sin ningún error visible).

        Uso: cuando add_documents() tuvo éxito pero save() falló después,
        esto revierte el índice en memoria al estado consistente con disco.
        """
        if count <= 0:
            return
        if start_id + count != self.index.ntotal:
            raise RuntimeError(
                "rollback_append solo puede deshacer el último append "
                f"(start_id={start_id}, count={count}, ntotal={self.index.ntotal})"
            )
        with self._index_lock:
            self.index.remove_ids(np.arange(start_id, start_id + count, dtype=np.int64))
        for fid in range(start_id, start_id + count):
            self.metadata.pop(fid, None)
        self._mutation_seq += 1

    # ------------------------------------------------------------------
    # Administración — activar / desactivar documentos
    # ------------------------------------------------------------------

    @contextmanager
    def write_lock(self):
        """Serializa mutaciones (toggle, upload). No bloquea lecturas:
        /chat sigue respondiendo mientras se guarda en disco."""
        with self._write_lock:
            yield

    def update_document_status(self, doc_id: str, active: bool) -> int:
        """Activa o desactiva todos los fragmentos de un documento.

        La comparación de doc_id es case-insensitive; el valor canónico
        almacenado no se modifica. No persiste — el llamador decide cuándo
        escribir a disco (ver save_metadata()).

        Returns:
            Número de fragmentos afectados.
        """
        target  = doc_id.lower()
        changed = 0
        for entry in self.metadata.values():
            if str(entry.get("doc_id") or "").lower() == target:
                entry["active"] = active
                changed += 1
        if changed:
            self._mutation_seq += 1
        return changed

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _write_index_file(self) -> None:
        """Escribe index.faiss con tmp+rename atómico.

        Requiere ~2x el tamaño del índice en disco de forma transitoria.
        Sin esto, un fallo o una lectura concurrente a mitad de escritura
        deja un index.faiss corrupto que no vuelve a cargar.
        """
        tmp = _index_path().with_suffix(".faiss.tmp")
        faiss.write_index(self.index, str(tmp))
        tmp.replace(_index_path())

    def _write_metadata_file(self) -> None:
        """Escribe metadata.pkl con tmp+rename atómico."""
        tmp = _metadata_path().with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.metadata, f)
        tmp.replace(_metadata_path())

    def _write_build_info(self, embedding_model: str = None) -> None:
        """Escribe build_info.json con el estado actual del índice."""
        build_info = {
            "embedding_model": embedding_model or settings.embedding_model,
            "faiss_index_type": "IndexFlatIP",
            "dimension":        self.index.d,
            "total_vectors":    self.index.ntotal,
            "saved_at":         datetime.utcnow().isoformat() + "Z",
        }
        with open(_build_info_path(), "w", encoding="utf-8") as f:
            json.dump(build_info, f, indent=2)

    def save(self, embedding_model: str = None) -> None:
        """Persiste el índice, la metadata y build_info en faiss_store/."""
        store_path = settings.faiss_store_path
        store_path.mkdir(parents=True, exist_ok=True)

        self._write_index_file()
        self._write_metadata_file()
        self._write_build_info(embedding_model)

        print(f"[FAISSStore] Guardado en '{store_path}'")
        print(f"  index.faiss  : {_index_path().stat().st_size / 1e6:.1f} MB")
        print(f"  metadata.pkl : {_metadata_path().stat().st_size / 1e6:.1f} MB")
        print(f"  Total vectores: {self.index.ntotal:,}")

    def save_metadata(self, embedding_model: str = None) -> None:
        """Persiste únicamente metadata.pkl y build_info (sin index.faiss).

        Usado tras activar/desactivar documentos: no hace falta reescribir
        el índice, que no cambió.
        """
        store_path = settings.faiss_store_path
        store_path.mkdir(parents=True, exist_ok=True)

        self._write_metadata_file()
        self._write_build_info(embedding_model)

        print(f"[FAISSStore] metadata guardada en '{store_path}'")
        print(f"  metadata.pkl : {_metadata_path().stat().st_size / 1e6:.1f} MB")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Busca los k fragmentos más relevantes para una query, excluyendo
        los desactivados desde el panel de administración.

        Args:
            query: Texto de la pregunta del usuario.
            k:     Número de resultados (default: settings.rag_top_k).

        Returns:
            Lista de dicts con keys: text, score, source_dataset,
            language, doc_id, chunk_id, y resto de metadata. Como mucho
            k elementos.
        """
        from src.ingestion.embedder import embed_query

        if k is None:
            k = settings.rag_top_k

        if self.index.ntotal == 0:
            return []

        # Embeddear la query
        q_vec = embed_query(query).reshape(1, -1)

        # Sobre-muestreamos para poder filtrar 'active' después sin que el
        # resultado se encoja por debajo de k. Con nada desactivado, esto
        # es equivalente a pedir exactamente k: se cortan los primeros k.
        k_fetch  = min(k * INACTIVE_OVERFETCH, INACTIVE_OVERFETCH_MAX)
        k_actual = min(max(k_fetch, k), self.index.ntotal)

        with self._index_lock:
            scores, ids = self.index.search(q_vec, k_actual)

        results = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:          # FAISS devuelve -1 si no hay suficientes vectores
                continue
            meta = self.metadata.get(int(faiss_id), {})
            if not is_active(meta):
                continue
            # 'score' va al final a propósito: si algún día la metadata
            # trae una clave 'score' (no ocurre hoy), no debe pisar la
            # similitud real calculada por FAISS.
            results.append({**meta, "score": float(score)})
            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return self.index.ntotal

    @property
    def mutation_seq(self) -> int:
        """Contador que cambia con cada add_documents / update_document_status.
        Usado para invalidar cachés derivados de la metadata sin TTL."""
        return self._mutation_seq

    def build_info(self) -> dict:
        """Lee y devuelve el build_info.json si existe."""
        p = _build_info_path()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}
