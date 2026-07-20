"""
retriever.py — Capa de recuperación RAG híbrida.

Estrategia:
  - Búsqueda DENSA (FAISS): sobre textbook, medmcqa, medqa_*
    Semántica — captura sinónimos y paráfrasis.
  - Búsqueda LÉXICA (BM25): sobre multiclinsum_summary y multiclinsum_fulltext
    Exacta — solo retorna casos clínicos si hay coincidencia real de términos.
    Evita que casos irrelevantes contaminen el contexto del LLM.

Resultado final: top-5 densa + top-2 BM25 (solo si score >= umbral).
Los fragmentos se numeran en orden: primero los densos, luego los BM25.

Uso:
    from src.rag.retriever import retrieve
    docs = retrieve("síntomas de preeclampsia", k=5)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.store import FAISSStore
from src.settings import settings

logger = logging.getLogger(__name__)

# Fuentes que usa la búsqueda densa (excluye Multiclinsum)
DENSE_SOURCES = {"textbook", "medmcqa", "medqa_us", "medqa_taiwan", "medqa_mainland"}
MULTICLINSUM_SOURCES = {"multiclinsum_summary", "multiclinsum_fulltext"}

# ---------------------------------------------------------------------------
# Singleton FAISS
# ---------------------------------------------------------------------------

_store: FAISSStore | None = None


def _get_store() -> FAISSStore:
    global _store
    if _store is None:
        logger.info("[Retriever] Cargando índice FAISS...")
        _store = FAISSStore.load()
        logger.info(f"[Retriever] Índice listo: {_store.total:,} vectores")
    return _store


# ---------------------------------------------------------------------------
# Etiquetas legibles por dataset
# ---------------------------------------------------------------------------

SOURCE_LABELS = {
    "multiclinsum_summary":  "Caso clínico real en español",
    "multiclinsum_fulltext": "Caso clínico real en español",
    "medmcqa":               "Pregunta médica con explicación",
    "medqa_us":              "Pregunta de examen médico (inglés)",
    "medqa_taiwan":          "Pregunta de examen médico (chino tradicional)",
    "medqa_mainland":        "Pregunta de examen médico (chino simplificado)",
    "textbook":              "Textbook de medicina",
}


def source_label(source_dataset: str) -> str:
    return SOURCE_LABELS.get(source_dataset, f"Fuente: {source_dataset}")


# ---------------------------------------------------------------------------
# Búsqueda densa — FAISS solo sobre fuentes no-Multiclinsum
# ---------------------------------------------------------------------------

def _retrieve_dense(query: str, k: int) -> list[dict[str, Any]]:
    """
    Recupera los k fragmentos más relevantes usando FAISS.
    Filtra en post-proceso para excluir Multiclinsum.
    Pide k*4 a FAISS para tener margen de filtrado.
    """
    store = _get_store()
    # Pedimos bastante más para poder filtrar Multiclinsum y quedarnos con k.
    # Multiclinsum ocupa ~14% del índice; con k*10 hay margen suficiente
    # incluso en consultas donde Multiclinsum domina las primeras posiciones.
    candidates = store.search(query, k=k * 10)

    results = []
    for doc in candidates:
        src = doc.get("source_dataset", "")
        if src in DENSE_SOURCES:
            results.append({**doc, "retrieval": "dense"})
            if len(results) >= k:
                break

    logger.info(f"[Retriever:dense] {len(results)}/{k} fragmentos (fuentes: textbook/medqa/medmcqa)")
    return results


# ---------------------------------------------------------------------------
# Búsqueda léxica — BM25 solo sobre Multiclinsum
# ---------------------------------------------------------------------------

def _retrieve_bm25(query: str, k: int) -> list[dict[str, Any]]:
    """
    Busca en Multiclinsum usando BM25.
    Solo retorna fragmentos con coincidencia léxica real.
    """
    try:
        from src.rag.bm25_index import search_bm25
        results = search_bm25(query, k=k, min_score=0.5)
        logger.info(f"[Retriever:bm25] {len(results)} fragmentos de Multiclinsum con match léxico")
        return results
    except Exception as e:
        logger.warning(f"[Retriever:bm25] Error en búsqueda BM25: {e}")
        return []


# ---------------------------------------------------------------------------
# Función pública
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int | None = None,
    k_bm25: int = 2,
) -> list[dict[str, Any]]:
    """
    Recupera fragmentos relevantes usando búsqueda híbrida.

    Args:
        query:  Pregunta del usuario.
        k:      Fragmentos densos a recuperar (default: settings.rag_top_k = 5).
        k_bm25: Fragmentos BM25 de Multiclinsum (default: 2, solo si hay match).

    Returns:
        Lista combinada: primero fragmentos densos, luego BM25 si los hay.
        Máximo k + k_bm25 fragmentos.
    """
    if not query or not query.strip():
        return []

    if k is None:
        k = settings.rag_top_k

    dense_results = _retrieve_dense(query, k=k)
    bm25_results  = _retrieve_bm25(query, k=k_bm25)

    # Merge: densos primero, BM25 al final (máx k+k_bm25 total)
    combined = dense_results + bm25_results
    logger.info(
        f"[Retriever] Total: {len(dense_results)} densos + "
        f"{len(bm25_results)} BM25 = {len(combined)} fragmentos"
    )
    return combined


# ---------------------------------------------------------------------------
# Formateo del contexto para el LLM
# ---------------------------------------------------------------------------

def format_context(docs: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """
    Convierte la lista de docs en bloque de contexto para el prompt.
    Cada fragmento va numerado — el LLM puede citar [n] inline.
    """
    if not docs:
        return "No se encontraron fragmentos relevantes en la base de conocimiento."

    fragments: list[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "").strip()
        retrieval_type = doc.get("retrieval", "dense")
        tag = " [caso clínico]" if retrieval_type == "bm25" else ""
        fragment = f"--- Fragmento [{i}]{tag} ---\n{text}"

        if total_chars + len(fragment) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                fragments.append(fragment[:remaining] + "...")
            break

        fragments.append(fragment)
        total_chars += len(fragment)

    return "\n\n".join(fragments)
