"""
retriever_configA.py — CONFIG A: FAISS puro (sin BM25, baseline).

Busqueda densa unica sobre el indice FAISS completo (375.392 vectores).
Sin distincion de fuente: multiclinsum, textbook, medmcqa y medqa_*
compiten en el mismo ranking por similitud coseno (IndexFlatIP).
Devuelve los top-k globales sin filtrar por dataset de origen.

PARA ACTIVAR CONFIG A:
    copy src\rag\retriever_configA.py src\rag\retriever.py

PARA RESTAURAR CONFIG B (actual):
    copy src\rag\retriever_configB.py src\rag\retriever.py

Ver foragents/retrieval_arquitecturas_configs.md para detalles completos.
Ver foragents/eval_setup_critico.md para instrucciones de evaluacion.
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

_store: FAISSStore | None = None


def _get_store() -> FAISSStore:
    global _store
    if _store is None:
        logger.info("[Retriever] Cargando indice FAISS...")
        _store = FAISSStore.load()
        logger.info(f"[Retriever] Indice listo: {_store.total:,} vectores")
    return _store


SOURCE_LABELS = {
    "multiclinsum_summary":  "Caso clinico real en espanol",
    "multiclinsum_fulltext": "Caso clinico real en espanol",
    "medmcqa":               "Pregunta medica con explicacion",
    "medqa_us":              "Pregunta de examen medico (ingles)",
    "medqa_taiwan":          "Pregunta de examen medico (chino tradicional)",
    "medqa_mainland":        "Pregunta de examen medico (chino simplificado)",
    "textbook":              "Textbook de medicina",
}


def source_label(source_dataset: str) -> str:
    return SOURCE_LABELS.get(source_dataset, f"Fuente: {source_dataset}")


def retrieve(
    query: str,
    k: int | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Config A: top-k global sobre todos los datasets sin distincion de fuente.
    Multiclinsum compite junto con textbook/medmcqa en el mismo ranking.
    """
    if not query or not query.strip():
        return []
    if k is None:
        k = settings.rag_top_k
    store   = _get_store()
    results = store.search(query, k=k)
    if min_score > 0.0:
        results = [r for r in results if r.get("score", 0.0) >= min_score]
    return results


def format_context(docs: list[dict[str, Any]], max_chars: int = 4000) -> str:
    if not docs:
        return "No se encontraron fragmentos relevantes en la base de conocimiento."
    fragments: list[str] = []
    total_chars = 0
    for i, doc in enumerate(docs, 1):
        text     = doc.get("text", "").strip()
        fragment = f"--- Fragmento [{i}] ---\n{text}"
        if total_chars + len(fragment) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                fragments.append(fragment[:remaining] + "...")
            break
        fragments.append(fragment)
        total_chars += len(fragment)
    return "\n\n".join(fragments)
