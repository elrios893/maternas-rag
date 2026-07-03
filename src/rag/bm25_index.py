"""
bm25_index.py — Índice BM25 sobre los fragmentos de Multiclinsum.

Solo Multiclinsum se indexa con BM25 (búsqueda léxica exacta).
El resto de fuentes (textbook, medmcqa, medqa_*) usan búsqueda densa FAISS.

El índice se construye en memoria la primera vez que se usa (singleton).
Multiclinsum tiene ~51,804 fragmentos — construcción ~10-20 s, ~150 MB RAM.

Uso:
    from src.rag.bm25_index import search_bm25
    results = search_bm25("preeclampsia hipertension proteinuria", k=2)
"""

from __future__ import annotations

import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.settings import settings

logger = logging.getLogger(__name__)

MULTICLINSUM_SOURCES = {"multiclinsum_summary", "multiclinsum_fulltext"}

# ---------------------------------------------------------------------------
# Tokenizador simple multilingüe (ES / EN)
# ---------------------------------------------------------------------------

_STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "que", "se", "los", "las", "un", "una",
    "por", "con", "para", "del", "al", "es", "su", "sus", "lo", "le", "les",
    "como", "pero", "si", "no", "fue", "una", "este", "esta", "esto",
}

_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "for",
    "on", "at", "by", "with", "from", "as", "be", "this", "that", "are",
    "were", "it", "he", "she", "we", "they", "has", "had", "have", "not",
}

_STOPWORDS = _STOPWORDS_ES | _STOPWORDS_EN


def _tokenize(text: str) -> list[str]:
    """Tokeniza texto en minúsculas, elimina stopwords y tokens cortos."""
    tokens = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Singleton BM25
# ---------------------------------------------------------------------------

_bm25_index = None          # instancia BM25Okapi
_bm25_docs: list[dict] = [] # lista de dicts con metadata + text


def _build_index() -> None:
    """Carga metadata.pkl, filtra Multiclinsum y construye el índice BM25."""
    global _bm25_index, _bm25_docs

    from rank_bm25 import BM25Okapi

    meta_path = settings.faiss_store_path / "metadata.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No se encontró metadata.pkl en '{settings.faiss_store_path}'. "
            "Ejecuta primero el pipeline de ingestión."
        )

    logger.info("[BM25] Cargando metadata.pkl y filtrando Multiclinsum...")
    with open(meta_path, "rb") as f:
        metadata: dict[int, dict] = pickle.load(f)

    # Filtrar solo fragmentos de Multiclinsum
    _bm25_docs = [
        doc for doc in metadata.values()
        if doc.get("source_dataset") in MULTICLINSUM_SOURCES
    ]
    logger.info(f"[BM25] {len(_bm25_docs):,} fragmentos de Multiclinsum cargados")

    # Tokenizar y construir índice
    corpus = [_tokenize(doc.get("text", "")) for doc in _bm25_docs]
    _bm25_index = BM25Okapi(corpus)
    logger.info("[BM25] Índice BM25 construido")


def _get_index():
    global _bm25_index
    if _bm25_index is None:
        _build_index()
    return _bm25_index


# ---------------------------------------------------------------------------
# Función pública
# ---------------------------------------------------------------------------

def search_bm25(
    query: str,
    k: int = 2,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Busca en el índice BM25 de Multiclinsum.

    Solo devuelve fragmentos con score BM25 >= min_score.
    Si no hay coincidencias léxicas suficientes, devuelve lista vacía —
    así Multiclinsum no contamina el contexto con casos irrelevantes.

    Args:
        query:     Texto de la query del usuario.
        k:         Máximo de resultados a devolver.
        min_score: Score BM25 mínimo (escala relativa, no absoluta).
                   0.5 es conservador — requiere al menos 1-2 términos exactos.

    Returns:
        Lista de dicts ordenada por score descendente con clave "bm25_score".
    """
    if not query or not query.strip():
        return []

    index = _get_index()
    tokens = _tokenize(query)

    if not tokens:
        return []

    scores = index.get_scores(tokens)

    # Emparejar scores con documentos y ordenar
    scored = sorted(
        zip(scores, _bm25_docs),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, doc in scored[:k * 3]:  # candidatos extra antes de filtrar
        if score < min_score:
            break
        results.append({**doc, "bm25_score": float(score), "retrieval": "bm25"})
        if len(results) >= k:
            break

    logger.info(
        f"[BM25] query='{query[:50]}' tokens={tokens[:6]} "
        f"→ {len(results)} resultados (top score={scores.max():.2f})"
    )
    return results
