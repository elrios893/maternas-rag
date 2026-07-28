"""
main.py — API FastAPI del chatbot Maternas.

Endpoints:
    GET  /health        — estado del servicio, vectores cargados
    POST /chat          — turno completo del chatbot (intent + risk + RAG + LLM)
    POST /classify      — solo clasificación (intent + risk, sin generar respuesta)

Arrancar:
    uvicorn src.api.main:app --reload --port 8000

Docs interactivas:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkDetail,
    ClassifyRequest,
    ClassifyResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentStatsResponse,
    HealthResponse,
    SourceDoc,
    TYPED_CHUNK_FIELDS,
)
from src.classifiers.intent_classifier import classify_intent
from src.classifiers.risk_detector import detect_risk
from src.rag.chain import chat as rag_chat
from src.rag.retriever import _get_store
from src.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup: cargar FAISS al arrancar (no en el primer request)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando índice FAISS al arrancar...")
    try:
        store = _get_store()
        logger.info(f"FAISS listo: {store.total:,} vectores")
    except Exception as e:
        logger.error(f"Error cargando FAISS: {e}")
    yield
    logger.info("Apagando servidor.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Maternas API",
    description="Chatbot RAG de salud materna — clasificación de intención, detección de riesgo y respuestas basadas en evidencia.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción restringir a la URL de Streamlit
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: agrupar chunks por documento
# ---------------------------------------------------------------------------


def _group_documents(store) -> dict[str, dict[str, Any]]:
    """Agrupa todos los chunks del store por doc_id.

    Retorna { doc_id: { doc_id, source_dataset, language,
                        chunk_count, total_chars, has_chunks } }
    """
    groups: dict[str, dict[str, Any]] = {}
    for entry in store.metadata.values():
        doc_id = entry.get("doc_id", "")
        if not doc_id:
            continue
        if doc_id not in groups:
            groups[doc_id] = {
                "doc_id":         doc_id,
                "source_dataset": entry.get("source_dataset", ""),
                "language":       entry.get("language", ""),
                "chunk_count":    0,
                "total_chars":    0,
                "has_chunks":     entry.get("is_chunk", False),
            }
        groups[doc_id]["chunk_count"] += 1
        groups[doc_id]["total_chars"] += len(entry.get("text", ""))
    return groups


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["sistema"])
def health() -> HealthResponse:
    """Estado del servicio y métricas básicas."""
    try:
        store = _get_store()
        return HealthResponse(
            status="ok",
            model=settings.embedding_model,
            total_vectors=store.total,
            faiss_loaded=True,
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {str(e)[:80]}",
            model=settings.embedding_model,
            total_vectors=0,
            faiss_loaded=False,
        )


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["chatbot"])
def chat(request: ChatRequest) -> ChatResponse:
    """
    Turno completo del chatbot.

    Recibe el mensaje del usuario y el historial de la conversación.
    Retorna la respuesta generada junto con metadatos de clasificación y fuentes.

    El caller es responsable de mantener y pasar el historial entre turnos.
    """
    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        result = rag_chat(
            query=request.message,
            history=history,
            k=request.k,
        )
    except Exception as e:
        logger.error(f"[/chat] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)[:120]}")

    sources = []
    for s in result.sources:
        sources.append(SourceDoc(
            score=s.get("score", 0.0),
            source_dataset=s.get("source_dataset", ""),
            language=s.get("language", ""),
            doc_id=s.get("doc_id"),
            chunk_id=s.get("chunk_id"),
        ))

    return ChatResponse(
        answer=result.answer,
        intent=result.intent,
        risk_level=result.risk_level,
        action=result.action,
        risk_flags=result.risk_flags,
        sources=sources,
        reasoning=result.reasoning,
        tokens_used=result.tokens_used,
    )


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse, tags=["clasificadores"])
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Solo clasificación: intención + riesgo clínico sin generar respuesta.

    Útil para pruebas rápidas de los clasificadores o para pipelines
    donde la generación se hace por separado.
    """
    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        intent_result = classify_intent(request.message, conversation_history=history)
        risk_result   = detect_risk(request.message, intent=intent_result.intent)
    except Exception as e:
        logger.error(f"[/classify] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:120])

    return ClassifyResponse(
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        risk_level=risk_result.level,
        risk_action=risk_result.action,
        risk_flags=risk_result.flags,
        risk_reasoning=risk_result.reasoning,
        used_heuristic=risk_result.used_heuristic,
    )


# ---------------------------------------------------------------------------
# GET /documents/stats
# ---------------------------------------------------------------------------

@app.get("/documents/stats", response_model=DocumentStatsResponse, tags=["documentos"])
def documents_stats() -> DocumentStatsResponse:
    """Estadísticas agregadas de la base documental."""
    try:
        store = _get_store()
    except Exception:
        logger.exception("Error al acceder al índice FAISS para /documents/stats")
        raise HTTPException(status_code=503, detail="Servicio de documentos no disponible")

    groups = _group_documents(store)

    return DocumentStatsResponse(
        total_vectors=store.total,
        doc_count=len(groups),
        total_chars=sum(g["total_chars"] for g in groups.values()),
        faiss_loaded=True,
        model=settings.embedding_model,
        indexed_at=store.build_info().get("saved_at", ""),
    )


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

@app.get("/documents", response_model=DocumentListResponse, tags=["documentos"])
def list_documents(
    search: str = "",
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
) -> DocumentListResponse:
    """Lista documentos con búsqueda y paginación."""
    try:
        store = _get_store()
    except Exception:
        logger.exception("Error al acceder al índice FAISS para /documents")
        raise HTTPException(status_code=503, detail="Servicio de documentos no disponible")

    groups = _group_documents(store)
    docs = list(groups.values())

    if search:
        q = search.lower()
        docs = [d for d in docs if q in d["doc_id"].lower() or q in d["source_dataset"].lower()]

    docs.sort(key=lambda d: d["doc_id"])
    total = len(docs)
    start = (page - 1) * per_page

    return DocumentListResponse(
        documents=docs[start:start + per_page],
        total=total,
        page=page,
        per_page=per_page,
    )


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}
# ---------------------------------------------------------------------------

@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse, tags=["documentos"])
def get_document_detail(doc_id: str) -> DocumentDetailResponse:
    """Detalle de un documento con todos sus fragmentos."""
    try:
        store = _get_store()
    except Exception:
        logger.exception("Error al acceder al índice FAISS para /documents/{doc_id}")
        raise HTTPException(status_code=503, detail="Servicio de documentos no disponible")

    chunks: list[dict[str, Any]] = []
    total_chars = 0

    for entry in store.metadata.values():
        if entry.get("doc_id") != doc_id:
            continue
        total_chars += len(entry.get("text", ""))
        chunks.append(entry)

    if not chunks:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' no encontrado")

    chunks.sort(key=lambda c: c.get("chunk_index", 0))

    first = chunks[0]

    return DocumentDetailResponse(
        doc_id=doc_id,
        source_dataset=first.get("source_dataset", ""),
        language=first.get("language", ""),
        chunk_count=len(chunks),
        total_chars=total_chars,
        has_chunks=any(c.get("is_chunk", False) for c in chunks),
        chunks=[
            ChunkDetail(
                chunk_id=c.get("chunk_id", ""),
                chunk_index=c.get("chunk_index", 0),
                text=c.get("text", ""),
                text_length=len(c.get("text", "")),
                language=c.get("language", ""),
                source_dataset=c.get("source_dataset", ""),
                is_chunk=c.get("is_chunk", False),
                subject=c.get("subject"),
                topic=c.get("topic"),
                filename=c.get("filename"),
                metadata={k: v for k, v in c.items() if k not in TYPED_CHUNK_FIELDS},
            )
            for c in chunks
        ],
    )
