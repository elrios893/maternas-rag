"""
retriever_configC.py — CONFIG C: FAISS+BM25 + corpus MaternaQA-es LM.

Igual que Config B (hibrido FAISS+BM25) pero con maternaqaes_lm
incluido en la capa densa (2.223 chunks de obstetricia en espanol).

Diferencias respecto a Config B:
  - DENSE_SOURCES incluye "maternaqaes_lm"
  - Los chunks de obstetricia ES compiten en el ranking semantico
    junto con textbook/medmcqa/medqa_*
  - El retrieval puede ahora recuperar fragmentos exactos de las
    GPCs colombianas, manuales y articulos de obstetricia que
    son la fuente del dataset MaternaQA-es

PARA ACTIVAR CONFIG C:
    copy src\\rag\\retriever_configC.py src\\rag\\retriever.py

PARA RESTAURAR CONFIG B:
    copy src\\rag\\retriever_configB.py src\\rag\\retriever.py

Prerequisito: haber ejecutado la ingestion del corpus LM:
    python -m src.ingestion.ingest_maternaqaes_lm

Ver foragents/retrieval_arquitecturas_configs.md y qa_technical Q27.
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

# Config C: incluye maternaqaes_lm en la capa densa
DENSE_SOURCES = {
    "textbook",
    "medmcqa",
    "medqa_us",
    "medqa_taiwan",
    "medqa_mainland",
    "maternaqaes_lm",   # <-- nuevo en Config C
}
MULTICLINSUM_SOURCES = {"multiclinsum_summary", "multiclinsum_fulltext"}

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
    "multiclinsum_summary":  "Caso clinico real en espanol",
    "multiclinsum_fulltext": "Caso clinico real en espanol",
    "medmcqa":               "Pregunta medica con explicacion",
    "medqa_us":              "Pregunta de examen medico (ingles)",
    "medqa_taiwan":          "Pregunta de examen medico (chino tradicional)",
    "medqa_mainland":        "Pregunta de examen medico (chino simplificado)",
    "textbook":              "Textbook de medicina",
    "maternaqaes_lm":        "Documento clinico obstetrico en espanol",
}


def source_label(source_dataset: str) -> str:
    return SOURCE_LABELS.get(source_dataset, f"Fuente: {source_dataset}")


# ---------------------------------------------------------------------------
# Busqueda densa — FAISS sobre DENSE_SOURCES (incluye maternaqaes_lm)
# ---------------------------------------------------------------------------

def _retrieve_dense(query: str, k: int) -> list[dict[str, Any]]:
    store = _get_store()
    # k*10 para filtrar Multiclinsum (~14% del indice) y garantizar k utiles
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
# Busqueda lexica — BM25 solo sobre Multiclinsum
# ---------------------------------------------------------------------------

def _retrieve_bm25(query: str, k: int) -> list[dict[str, Any]]:
    try:
        from src.rag.bm25_index import search_bm25
        results = search_bm25(query, k=k, min_score=0.5)
        logger.info(f"[Retriever:bm25] {len(results)} fragmentos Multiclinsum")
        return results
    except Exception as e:
        logger.warning(f"[Retriever:bm25] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# Funcion publica
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int | None = None,
    k_bm25: int = 2,
) -> list[dict[str, Any]]:
    """
    Config C: top-k densa (textbook + medmcqa + medqa_* + maternaqaes_lm)
    + top-2 BM25 (Multiclinsum, solo si hay match lexico).
    """
    if not query or not query.strip():
        return []
    if k is None:
        k = settings.rag_top_k

    dense_results = _retrieve_dense(query, k=k)
    bm25_results  = _retrieve_bm25(query, k=k_bm25)

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
    if not docs:
        return "No se encontraron fragmentos relevantes en la base de conocimiento."

    fragments: list[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "").strip()
        retrieval_type = doc.get("retrieval", "dense")
        tag = " [caso clinico]" if retrieval_type == "bm25" else ""
        fragment = f"--- Fragmento [{i}]{tag} ---\n{text}"

        if total_chars + len(fragment) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                fragments.append(fragment[:remaining] + "...")
            break

        fragments.append(fragment)
        total_chars += len(fragment)

    return "\n\n".join(fragments)
