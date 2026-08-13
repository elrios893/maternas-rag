"""
retriever_configD.py — CONFIG D: sin textbook ni multiclinsum (licencia).

Prueba de impacto antes de remover fisicamente estos dos datasets del
indice FAISS (foragents/qa_technical.md Q28): textbook (18 libros
medicos sin licencia identificable) y multiclinsum (CC-BY-4.0, casos
clinicos reales derivados de PMC-Patients, sin garantia documentada de
desidentificacion por parte del dataset original).

Diferencias respecto a Config C:
  - DENSE_SOURCES ya NO incluye "textbook"
  - La capa BM25 sobre Multiclinsum se elimina por completo (no se
    llama a _retrieve_bm25 / bm25_index en absoluto)
  - retrieve() devuelve unicamente resultados densos de:
    medmcqa + medqa_us + medqa_taiwan + medqa_mainland + maternaqaes_lm

Esto NO borra los vectores del indice FAISS todavia — solo los excluye
del retrieval, para medir el impacto en las metricas de Ragas antes de
decidir si vale la pena reconstruir el indice sin ellos.

PARA ACTIVAR CONFIG D (solo evaluacion, no produccion):
    copy src\\rag\\retriever_configD.py src\\rag\\retriever.py

PARA RESTAURAR CONFIG C (produccion):
    copy src\\rag\\retriever_configC.py src\\rag\\retriever.py

Ver foragents/retrieval_arquitecturas_configs.md, foragents/eval_runbook.md
y foragents/qa_technical.md Q28.
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

# Config D: sin textbook (licencia no identificable) y sin multiclinsum
# (ni denso ni BM25 — evaluando su remocion completa, no solo su demotion)
DENSE_SOURCES = {
    "medmcqa",
    "medqa_us",
    "medqa_taiwan",
    "medqa_mainland",
    "maternaqaes_lm",
}

# ---------------------------------------------------------------------------
# Singleton FAISS
# ---------------------------------------------------------------------------

_store: FAISSStore | None = None


def _get_store() -> FAISSStore:
    global _store
    if _store is None:
        logger.info("[Retriever] Cargando indice FAISS...")
        _store = FAISSStore.load()
        logger.info(f"[Retriever] Indice listo: {_store.total:,} vectores")
    return _store


# ---------------------------------------------------------------------------
# Etiquetas legibles por dataset
# ---------------------------------------------------------------------------

SOURCE_LABELS = {
    "medmcqa":               "Pregunta medica con explicacion",
    "medqa_us":               "Pregunta de examen medico (ingles)",
    "medqa_taiwan":            "Pregunta de examen medico (chino tradicional)",
    "medqa_mainland":          "Pregunta de examen medico (chino simplificado)",
    "maternaqaes_lm":          "Documento clinico obstetrico en espanol",
}


def source_label(source_dataset: str) -> str:
    return SOURCE_LABELS.get(source_dataset, f"Fuente: {source_dataset}")


def source_path(doc: dict[str, Any]) -> str:
    filename = doc.get("filename") or doc.get("source_pdf") or ""
    chunk_id = doc.get("chunk_id") or ""
    doc_id   = doc.get("doc_id") or ""

    if filename and chunk_id:
        return f"{filename} (chunk {chunk_id})"
    if filename:
        return filename
    if doc_id and chunk_id:
        return f"{doc_id}/{chunk_id}"
    if doc_id:
        return doc_id
    if chunk_id:
        return chunk_id
    return "desconocido"


# ---------------------------------------------------------------------------
# Busqueda densa — FAISS sobre DENSE_SOURCES (sin textbook)
# ---------------------------------------------------------------------------

def _retrieve_dense(query: str, k: int) -> list[dict[str, Any]]:
    store = _get_store()
    # k*10 para filtrar textbook y multiclinsum (~49% del indice combinado)
    candidates = store.search(query, k=k * 10)

    results = []
    for doc in candidates:
        src = doc.get("source_dataset", "")
        if src in DENSE_SOURCES:
            results.append({**doc, "retrieval": "dense"})
            if len(results) >= k:
                break

    logger.info(f"[Retriever:dense] {len(results)}/{k} fragmentos")
    return results


# ---------------------------------------------------------------------------
# Funcion publica
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int | None = None,
    k_bm25: int = 0,
) -> list[dict[str, Any]]:
    """
    Config D: top-k densa (medmcqa + medqa_* + maternaqaes_lm) unicamente.
    Sin textbook, sin multiclinsum (ni denso ni BM25).
    """
    if not query or not query.strip():
        return []
    if k is None:
        k = settings.rag_top_k

    dense_results = _retrieve_dense(query, k=k)

    logger.info(f"[Retriever] Total: {len(dense_results)} densos (config D)")
    return dense_results


# ---------------------------------------------------------------------------
# Formateo del contexto para el LLM
# ---------------------------------------------------------------------------

def format_context(docs: list[dict[str, Any]], max_chars: int = 4000) -> str:
    if not docs:
        return "No se encontraron fragmentos relevantes en la base de conocimiento."

    fragments: list[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "").strip()
        fragment = f"--- Fragmento [{i}] ---\n{text}"

        if total_chars + len(fragment) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                fragments.append(fragment[:remaining] + "...")
            break

        fragments.append(fragment)
        total_chars += len(fragment)

    return "\n\n".join(fragments)
