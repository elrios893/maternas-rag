"""
routes_documents.py — Endpoints del panel de administración de documentos.

Router separado (no crece main.py) para que la dependencia de auth se
declare UNA VEZ, en el router: un endpoint nuevo no puede quedar sin
proteger por descuido.

Todos los endpoints usan _get_store() de src.rag.retriever — el mismo
singleton que sirve /chat — nunca FAISSStore.load() por su cuenta.
Un store propio dejaría al servidor sirviendo vectores viejos mientras
un save() posterior revertiría en silencio cualquier alta hecha aquí.

Restricción operativa: el modelo de singleton + locks en proceso asume
un único worker. Arrancar con `uvicorn --workers N>1` deja a cada
worker con una copia divergente del índice; el último save() gana.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from src.api.auth import require_admin_token
from src.api.schemas import (
    ChunkDetail,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentStatsResponse,
    DocumentSummary,
    ToggleDocumentResponse,
    ToggleDocumentStatus,
    UploadDocumentResponse,
    TYPED_CHUNK_FIELDS,
)
from src.ingestion.chunkers import chunk_document
from src.ingestion.formatters import format_upload, safe_doc_id
from src.ingestion.store import is_active
from src.rag.retriever import _get_store
from src.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documentos"],
    dependencies=[Depends(require_admin_token)],
)

# Límites del upload. Acotamos cantidad de CHUNKS, no solo bytes: el
# embedding corre síncrono dentro del request sobre CPU, y lo que define
# el tiempo real es cuántos chunks hay que embeddear, no el tamaño del
# archivo. 400 chunks son ~640 KB de texto, del orden de un minuto en CPU.
MAX_UPLOAD_BYTES  = 2 * 1024 * 1024   # 2 MB
MAX_UPLOAD_CHUNKS = 400
MIN_UPLOAD_CHARS  = 50                # igual a MIN_PARAGRAPH_CHARS en chunkers.py

DETAIL_PER_PAGE_MAX = 50


# ---------------------------------------------------------------------------
# Vista cacheada del índice — un solo barrido O(N) por mutación, no por request
# ---------------------------------------------------------------------------
# store.metadata tiene ~375k entradas; _build_index_view lo recorre entero.
# La UI llama /documents y /documents/stats juntos en cada interacción, así
# que sin caché eso son dos barridos por click. Cacheamos por versión
# (mutation_seq), no por TTL: un TTL seguiría reportando "activo" un rato
# después de que el operador lo desactivó, el peor fallo para un control
# de compliance.

_view_cache_lock = threading.Lock()
_view_cache: dict[str, Any] = {"key": None, "value": None}


def _build_index_view(store) -> dict:
    """Un barrido O(N) sobre store.metadata que produce:
        groups    : { doc_id → resumen agregado del documento }
        canonical : { doc_id.lower() → doc_id tal como está almacenado }

    active y has_chunks se agregan sobre TODOS los chunks del documento
    (AND / OR), no se toman del primero que aparece — si no, un documento
    a medio activar/desactivar reportaría un estado arbitrario.
    """
    groups: dict[str, dict[str, Any]] = {}
    canonical: dict[str, str] = {}

    for entry in store.metadata.values():
        doc_id = str(entry.get("doc_id") or "")
        if not doc_id:
            continue
        key = doc_id.lower()
        if key not in groups:
            canonical[key] = doc_id
            groups[doc_id] = {
                "doc_id":         doc_id,
                "source_dataset": entry.get("source_dataset", ""),
                "language":       entry.get("language", ""),
                "chunk_count":    0,
                "total_chars":    0,
                "has_chunks":     False,
                "active":         True,
            }
        g = groups[doc_id]
        g["chunk_count"] += 1
        g["total_chars"] += len(entry.get("text", ""))
        g["has_chunks"]  = g["has_chunks"] or bool(entry.get("is_chunk", False))
        g["active"]      = g["active"] and is_active(entry)

    return {"groups": groups, "canonical": canonical}


def _index_view(store) -> dict:
    key = (id(store), store.total, store.mutation_seq)
    with _view_cache_lock:
        if _view_cache["key"] == key:
            return _view_cache["value"]
    view = _build_index_view(store)   # fuera del lock: el barrido es largo
    with _view_cache_lock:
        _view_cache["key"], _view_cache["value"] = key, view
    return view


def _get_store_or_503():
    try:
        return _get_store()
    except Exception:
        logger.exception("Error al acceder al índice FAISS")
        raise HTTPException(status_code=503, detail="Servicio de documentos no disponible")


# ---------------------------------------------------------------------------
# GET /documents/stats — declarado ANTES de /{doc_id}, o 'stats' matchea
# el path param y nunca llega acá.
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=DocumentStatsResponse)
def documents_stats() -> DocumentStatsResponse:
    """Estadísticas agregadas de la base documental."""
    store = _get_store_or_503()
    groups = _index_view(store)["groups"]

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

@router.get("", response_model=DocumentListResponse)
def list_documents(
    search: str = "",
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
) -> DocumentListResponse:
    """Lista documentos con búsqueda y paginación."""
    store = _get_store_or_503()
    groups = _index_view(store)["groups"]
    docs = list(groups.values())

    if search:
        q = search.lower()
        docs = [d for d in docs if q in d["doc_id"].lower() or q in d["source_dataset"].lower()]

    docs.sort(key=lambda d: d["doc_id"])
    total = len(docs)
    start = (page - 1) * per_page

    return DocumentListResponse(
        documents=[DocumentSummary(**d) for d in docs[start:start + per_page]],
        total=total,
        page=page,
        per_page=per_page,
    )


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}
# ---------------------------------------------------------------------------

@router.get("/{doc_id}", response_model=DocumentDetailResponse)
def document_detail(
    doc_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=DETAIL_PER_PAGE_MAX),
) -> DocumentDetailResponse:
    """Detalle de un documento con inspección de sus fragmentos, paginado.

    Paginado a propósito: un documento de textbook son miles de chunks con
    texto completo cada uno; devolverlos todos de una sería una respuesta
    de varios MB dentro de un único diálogo de la UI.
    """
    store = _get_store_or_503()
    target = doc_id.lower()

    all_chunks = [
        entry for entry in store.metadata.values()
        if str(entry.get("doc_id") or "").lower() == target
    ]
    if not all_chunks:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' no encontrado")

    all_chunks.sort(key=lambda c: c.get("chunk_index", 0))

    canonical_id  = str(all_chunks[0].get("doc_id") or doc_id)
    source_dataset = all_chunks[0].get("source_dataset", "")
    language       = all_chunks[0].get("language", "")
    total_chars    = sum(len(c.get("text", "")) for c in all_chunks)
    has_chunks     = any(c.get("is_chunk", False) for c in all_chunks)
    active         = all(is_active(c) for c in all_chunks)

    start = (page - 1) * per_page
    page_chunks = all_chunks[start:start + per_page]

    chunks = [
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
        for c in page_chunks
    ]

    return DocumentDetailResponse(
        doc_id=canonical_id,
        source_dataset=source_dataset,
        language=language,
        chunk_count=len(all_chunks),
        total_chars=total_chars,
        has_chunks=has_chunks,
        active=active,
        page=page,
        per_page=per_page,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# PATCH /documents/{doc_id}
# ---------------------------------------------------------------------------

@router.patch("/{doc_id}", response_model=ToggleDocumentResponse)
def toggle_document(doc_id: str, body: ToggleDocumentStatus) -> ToggleDocumentResponse:
    """Activa o desactiva todos los fragmentos de un documento."""
    store = _get_store_or_503()

    view = _index_view(store)
    canonical = view["canonical"].get(doc_id.lower())
    if canonical is None:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' no encontrado")

    with store.write_lock():
        affected = store.update_document_status(canonical, active=body.active)
        try:
            store.save_metadata()
        except Exception:
            # Revertir en memoria para no divergir de lo que quedó en disco.
            store.update_document_status(canonical, active=not body.active)
            logger.exception("Error al persistir metadata tras PATCH /documents/%s", canonical)
            raise HTTPException(status_code=500, detail="No se pudo persistir el cambio de estado")

    return ToggleDocumentResponse(doc_id=canonical, active=body.active, affected_chunks=affected)


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadDocumentResponse)
def upload_document(file: UploadFile = File(...)) -> UploadDocumentResponse:
    """Sube un documento .txt, lo chunkea y lo indexa en el FAISS en vivo.

    Solo cargar documentos cuya licencia permita su uso e indexación: el
    proyecto removió textbook y multiclinsum del índice por licenciamiento
    no defendible (ver src/rag/retriever.py, DENSE_SOURCES); un upload
    abierto reintroduce ese riesgo por otra puerta. El token admin y la
    etiqueta 'upload' visible en cada fuente citada son las mitigaciones.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".txt") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .txt sin ruta en el nombre")

    doc_id = safe_doc_id(filename)
    if not doc_id:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande: máximo {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        # latin-1 mapea los 256 valores de byte y nunca lanza: es un
        # fallback total, no un intento que pueda fallar (a diferencia de
        # lo que sugeriría un segundo except aquí).
        text = content.decode("latin-1")

    if len(text.strip()) < MIN_UPLOAD_CHARS:
        raise HTTPException(status_code=400, detail="El archivo está vacío o es demasiado corto")

    store = _get_store_or_503()

    doc = format_upload(filename, text)
    chunks = chunk_document(doc)

    if len(chunks) > MAX_UPLOAD_CHUNKS:
        raise HTTPException(
            status_code=413,
            detail=f"El documento genera demasiados fragmentos ({len(chunks)} > {MAX_UPLOAD_CHUNKS})",
        )

    with store.write_lock():
        if doc_id.lower() in _index_view(store)["canonical"]:   # dentro del lock: sin TOCTOU
            raise HTTPException(status_code=409, detail=f"Ya existe un documento con doc_id '{doc_id}'")

        start_id = store.total
        added = store.add_documents(chunks, show_progress=False)
        try:
            store.save()
        except Exception:
            store.rollback_append(start_id, added)
            logger.exception("Error al persistir el índice tras la subida de '%s'", filename)
            raise HTTPException(status_code=500, detail="Error al indexar el documento")

    return UploadDocumentResponse(
        filename=filename,
        doc_id=doc_id,
        chunks_created=added,
        total_vectors=store.total,
    )
