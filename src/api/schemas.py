"""
schemas.py — Modelos Pydantic para la API de Maternas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role:    str = Field(..., description="'user' o 'assistant'")
    content: str = Field(..., description="Texto del turno")


class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensaje actual del usuario", min_length=1)
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Historial de la conversación (turnos anteriores)",
    )
    k: Optional[int] = Field(
        default=None,
        ge=1, le=20,
        description="Número de fragmentos RAG a recuperar (default: settings.rag_top_k)",
    )


class SourceDoc(BaseModel):
    score:          float
    source_dataset: str
    language:       str
    doc_id:         Optional[str] = None
    chunk_id:       Optional[str] = None


class ChatResponse(BaseModel):
    answer:      str
    intent:      str
    risk_level:  str
    action:      str
    risk_flags:  list[str]
    sources:     list[SourceDoc]
    reasoning:   str
    tokens_used: int


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ClassifyResponse(BaseModel):
    intent:          str
    intent_confidence: float
    risk_level:      str
    risk_action:     str
    risk_flags:      list[str]
    risk_reasoning:  str
    used_heuristic:  bool


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:        str
    model:         str
    total_vectors: int
    faiss_loaded:  bool


# ---------------------------------------------------------------------------
# GET /documents, GET /documents/stats
# ---------------------------------------------------------------------------

class DocumentSummary(BaseModel):
    doc_id:         str
    source_dataset: str
    language:       str
    chunk_count:    int
    total_chars:    int
    has_chunks:     bool
    active:         bool = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total:     int
    page:      int
    per_page:  int


class DocumentStatsResponse(BaseModel):
    total_vectors: int
    doc_count:     int
    total_chars:   int
    faiss_loaded:  bool
    model:         str
    indexed_at:    str


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}
# ---------------------------------------------------------------------------

TYPED_CHUNK_FIELDS = {
    "chunk_id", "chunk_index", "text", "language",
    "source_dataset", "is_chunk", "subject", "topic", "filename",
}


class ChunkDetail(BaseModel):
    chunk_id:      str
    chunk_index:   int
    text:          str
    text_length:   int
    language:      str
    source_dataset: str
    is_chunk:      bool
    subject:       Optional[str] = None
    topic:         Optional[str] = None
    filename:      Optional[str] = None
    metadata:      dict = Field(default_factory=dict, description="Atributos adicionales del índice")


class DocumentDetailResponse(BaseModel):
    doc_id:         str
    source_dataset: str
    language:       str
    chunk_count:    int
    total_chars:    int
    has_chunks:     bool
    chunks:         list[ChunkDetail]


# ---------------------------------------------------------------------------
# PATCH /documents/{doc_id}
# ---------------------------------------------------------------------------

class ToggleDocumentStatus(BaseModel):
    active: bool


class ToggleDocumentResponse(BaseModel):
    doc_id:          str
    active:          bool
    affected_chunks: int


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------

class UploadDocumentResponse(BaseModel):
    filename:       str
    doc_id:         str
    chunks_created: int
    total_vectors:  int
