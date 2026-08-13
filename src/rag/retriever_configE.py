"""
retriever_configE.py — CONFIG E: Config D + HyDE (Hypothetical Document Embeddings).

Prueba de si generar una respuesta hipotetica antes de embeder mejora el
retrieval frente a embeder la query cruda del usuario. Motivacion: las
usuarias de Telegram escriben corto/informal ("me duele la cabeza, es
normal?") mientras que maternaqaes_lm son guias clinicas en lenguaje
formal — el desajuste de vocabulario puede hacer que la busqueda densa
no encuentre el fragmento correcto aunque exista en el indice.

Diferencia respecto a Config D:
  - Antes de embeder, se le pide al LLM (Groq, mismo modelo que produccion)
    que escriba un parrafo corto de respuesta hipotetica a la pregunta.
    Ese parrafo (no la query original) es lo que se embede y se busca en FAISS.
  - Si la generacion HyDE falla (error de API, timeout), se usa la query
    original sin HyDE — no debe romper el retrieval.
  - DENSE_SOURCES identico a Config D (medmcqa + medqa_* + maternaqaes_lm,
    sin textbook/multiclinsum — ver qa_technical.md Q31)

Costo: +1 llamada Groq por turno (ademas de la generacion de la respuesta
final), suma latencia y consumo de cuota diaria de tokens.

PARA ACTIVAR CONFIG E (solo evaluacion, no produccion):
    copy src\\rag\\retriever_configE.py src\\rag\\retriever.py

PARA RESTAURAR CONFIG D (produccion):
    copy src\\rag\\retriever_configD.py src\\rag\\retriever.py

Ver foragents/qa_technical.md Q31 (por que Config D es produccion) y Q32
(resultados de este experimento).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from groq import Groq

from src.ingestion.store import FAISSStore
from src.settings import settings

logger = logging.getLogger(__name__)

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
# Cliente Groq (singleton) — mismo patron que chain.py
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# HyDE — genera un parrafo hipotetico de respuesta para embeder en vez
# de la query cruda
# ---------------------------------------------------------------------------

_HYDE_PROMPT = (
    "Eres un profesional clinico escribiendo un fragmento de una guia de "
    "practica clinica en obstetricia/ginecologia.\n\n"
    "Pregunta de una paciente: '{query}'\n\n"
    "Escribe un parrafo corto (3-4 oraciones) en espanol, con lenguaje "
    "clinico formal, que respondería esa pregunta como si fuera un extracto "
    "de una guia medica o un texto de referencia clinica. No uses markdown, "
    "no digas que eres una IA, no incluyas advertencias ni disclaimers — "
    "solo el contenido clinico."
)


def _generate_hyde(query: str) -> str:
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
            temperature=0.3,
            max_tokens=150,
        )
        hyde_text = resp.choices[0].message.content.strip()
        logger.info(f"[Retriever:HyDE] '{query[:40]}' -> '{hyde_text[:60]}...'")
        return hyde_text or query
    except Exception as e:
        logger.warning(f"[Retriever:HyDE] Error generando, usando query cruda: {e}")
        return query


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
# Busqueda densa — FAISS sobre DENSE_SOURCES, embediendo el texto HyDE
# ---------------------------------------------------------------------------

def _retrieve_dense(query: str, k: int) -> list[dict[str, Any]]:
    store = _get_store()
    hyde_text = _generate_hyde(query)

    # k*10 para filtrar fuentes fuera de DENSE_SOURCES con margen
    candidates = store.search(hyde_text, k=k * 10)

    results = []
    for doc in candidates:
        src = doc.get("source_dataset", "")
        if src in DENSE_SOURCES:
            results.append({**doc, "retrieval": "dense"})
            if len(results) >= k:
                break

    logger.info(f"[Retriever:dense] {len(results)}/{k} fragmentos (via HyDE)")
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
    Config E: Config D + HyDE. Genera una respuesta hipotetica con el LLM
    y embede eso (no la query cruda) para la busqueda densa.
    """
    if not query or not query.strip():
        return []
    if k is None:
        k = settings.rag_top_k

    dense_results = _retrieve_dense(query, k=k)

    logger.info(f"[Retriever] Total: {len(dense_results)} densos (config E, HyDE)")
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
