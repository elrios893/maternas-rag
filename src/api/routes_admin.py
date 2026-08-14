"""
routes_admin.py — Métricas de evaluación y configuración del sistema,
solo lectura, para el panel de administración.

No recomputa nada: lee los reportes que src/evaluation/eval_pipeline.py
ya generó en evaluation_reports/, y expone la configuración efectiva del
proceso (con los secretos redactados). Editar settings en caliente no es
un caso soportado: desincronizaría .env, el proceso en memoria y el
FAISSStore singleton.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import require_admin_token
from src.api.schemas import (
    AdminConfigResponse,
    EvaluationDetailResponse,
    EvaluationListResponse,
    EvaluationSummary,
)
from src.rag.retriever import DENSE_SOURCES, _get_store
from src.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["administracion"], dependencies=[Depends(require_admin_token)])

REPORTS_DIR = Path("evaluation_reports")   # igual que src/evaluation/eval_pipeline.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_run_path(run_id: str) -> Path:
    """Resuelve run_id -> evaluation_reports/eval_results_<run_id>.json,
    verificando que el resultado siga dentro de REPORTS_DIR (sin esto,
    un run_id tipo '../../secrets' podría leer fuera del directorio)."""
    reports_dir = REPORTS_DIR.resolve()
    candidate = (reports_dir / f"eval_results_{run_id}.json").resolve()
    if reports_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="run_id inválido")
    return candidate


def _list_run_files() -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(REPORTS_DIR.glob("eval_results_*.json"))


# ---------------------------------------------------------------------------
# GET /admin/evaluations
# ---------------------------------------------------------------------------

@router.get("/evaluations", response_model=EvaluationListResponse)
def list_evaluations() -> EvaluationListResponse:
    """Lista las corridas de evaluación ya generadas, más recientes primero."""
    runs = []
    for path in _list_run_files():
        run_id = path.stem.removeprefix("eval_results_")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("No se pudo leer el reporte de evaluación '%s'", path)
            continue

        runs.append(EvaluationSummary(
            run_id=run_id,
            config=data.get("config", ""),
            timestamp=data.get("timestamp", ""),
            dataset=data.get("dataset", ""),
            n_evaluated=data.get("n_evaluated", 0),
            n_failed=data.get("n_failed", 0),
            metrics_global=data.get("metrics_global", {}),
        ))

    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return EvaluationListResponse(runs=runs)


# ---------------------------------------------------------------------------
# GET /admin/evaluations/{run_id}
# ---------------------------------------------------------------------------

@router.get("/evaluations/{run_id}", response_model=EvaluationDetailResponse)
def evaluation_detail(run_id: str, include_rows: bool = Query(False)) -> EvaluationDetailResponse:
    """Detalle completo de una corrida. 'rows' (por-pregunta) se omite por
    defecto: es el grueso del archivo y solo hace falta para inspección fina."""
    path = _resolve_run_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Corrida '{run_id}' no encontrada")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Error al leer el reporte de evaluación '%s'", path)
        raise HTTPException(status_code=500, detail="No se pudo leer el reporte de evaluación")

    return EvaluationDetailResponse(
        run_id=run_id,
        config=data.get("config", ""),
        timestamp=data.get("timestamp", ""),
        dataset=data.get("dataset", ""),
        n_sample=data.get("n_sample", 0),
        n_evaluated=data.get("n_evaluated", 0),
        n_failed=data.get("n_failed", 0),
        metrics_global=data.get("metrics_global", {}),
        metrics_by_tipo=data.get("metrics_by_tipo", {}),
        metrics_by_dificultad=data.get("metrics_by_dificultad", {}),
        rows=data.get("rows") if include_rows else None,
    )


# ---------------------------------------------------------------------------
# GET /admin/config
# ---------------------------------------------------------------------------

@router.get("/config", response_model=AdminConfigResponse)
def admin_config() -> AdminConfigResponse:
    """Configuración efectiva del proceso, con secretos redactados a un
    booleano 'configurado'. Nunca se expone el valor de un secreto."""
    try:
        store = _get_store()
        build_info = store.build_info()
    except Exception:
        build_info = {}

    return AdminConfigResponse(
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        rag_top_k=settings.rag_top_k,
        dense_sources=sorted(DENSE_SOURCES),
        groq_model=settings.groq_model,
        index_build_info=build_info,
        secrets_configured={
            "groq_api_key":              bool(settings.groq_api_key),
            "groq_api_key_2":            bool(settings.groq_api_key_2),
            "telegram_bot_token":        bool(settings.telegram_bot_token),
            "admin_api_token":           bool(settings.admin_api_token),
            "openrouter_key":            bool(settings.openrouter_key),
            "cerebras_key":              bool(settings.cerebras_key),
            "active_users_encryption_key": bool(settings.active_users_encryption_key),
            "notifier_smtp_password":    bool(settings.notifier_smtp_password),
        },
    )
