"""
retriever.py — Capa de recuperación RAG sobre el índice FAISS.

Responsabilidades:
  - Cargar el FAISSStore como singleton (una sola vez por proceso)
  - Ejecutar búsqueda top-k con el prefijo E5 correcto ("query: ...")
  - Filtrar y formatear los resultados para el LLM
  - Exponer retrieve() como función pública simple

Uso:
    from src.rag.retriever import retrieve
    docs = retrieve("¿Qué síntomas indican preeclampsia?", k=5)
    for d in docs:
        print(d["score"], d["source_dataset"], d["text"][:80])
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

# ---------------------------------------------------------------------------
# Singleton del índice FAISS
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
# Etiquetas legibles para cada tipo de fuente
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
    """Devuelve una etiqueta legible para humanos según el dataset de origen."""
    return SOURCE_LABELS.get(source_dataset, f"Fuente: {source_dataset}")


# ---------------------------------------------------------------------------
# Función pública
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Recupera los fragmentos más relevantes del índice FAISS para la query.

    El prefijo "query: " se aplica internamente en embed_query() —
    no hace falta incluirlo en el argumento query.

    Args:
        query:     Pregunta o texto del usuario.
        k:         Número de fragmentos a recuperar (default: settings.rag_top_k).
        min_score: Score mínimo de similitud coseno para incluir un resultado.
                   0.0 = no filtrar (todos los resultados).

    Returns:
        Lista de dicts ordenada por score descendente, cada uno con:
            text           — texto del fragmento
            score          — similitud coseno (0.0 – 1.0)
            source_dataset — "medmcqa" | "medqa_us" | "multiclinsum_summary" | etc.
            language       — "en" | "es" | "zh-hans" | "zh-hant"
            doc_id         — identificador del documento original
            chunk_id       — identificador único del chunk
            + resto de metadata del formatter original
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
    """
    Convierte la lista de docs recuperados en un bloque de contexto
    listo para incluir en el prompt del LLM.

    Cada fragmento numerado. El LLM puede citar [n] inline.
    """
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
