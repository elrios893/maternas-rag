"""
client.py — Cliente HTTP para la API del backend Maternas.

Sin dependencias de Streamlit. Las excepciones de httpx se propagan
hacia la capa de presentación (vistas), que decide cómo mostrarlas.
"""

from urllib.parse import quote

import httpx

from src.settings import settings

API_URL     = settings.api_url
API_TIMEOUT = 60


def _admin_headers() -> dict:
    """Header de autenticación para /documents* y /admin*.

    No se envía en /health ni /chat, que son públicos. El token vive en
    el proceso de Streamlit (settings.admin_api_token, leído de .env) —
    nunca llega al navegador, porque Streamlit renderiza del lado
    servidor.
    """
    token = settings.admin_api_token
    return {"X-Admin-Token": token} if token else {}


# ---------------------------------------------------------------------------
# Chat (público)
# ---------------------------------------------------------------------------

def check_health() -> dict:
    r = httpx.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def call_chat(message: str, history: list) -> dict:
    payload = {"message": message, "history": history, "k": 5}
    r = httpx.post(f"{API_URL}/chat", json=payload, timeout=API_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Documentos (requieren X-Admin-Token)
# ---------------------------------------------------------------------------

def list_documents(search: str = "", page: int = 1, per_page: int = 20) -> dict:
    r = httpx.get(
        f"{API_URL}/documents",
        params={"search": search, "page": page, "per_page": per_page},
        headers=_admin_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_document_stats() -> dict:
    r = httpx.get(f"{API_URL}/documents/stats", headers=_admin_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def get_document_detail(doc_id: str, page: int = 1, per_page: int = 20) -> dict:
    r = httpx.get(
        f"{API_URL}/documents/{quote(doc_id, safe='')}",
        params={"page": page, "per_page": per_page},
        headers=_admin_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def toggle_document_status(doc_id: str, active: bool) -> dict:
    r = httpx.patch(
        f"{API_URL}/documents/{quote(doc_id, safe='')}",
        json={"active": active},
        headers=_admin_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def upload_document(filename: str, content: bytes) -> dict:
    r = httpx.post(
        f"{API_URL}/documents/upload",
        files={"file": (filename, content)},
        headers=_admin_headers(),
        # El embedding corre síncrono en el backend (ver
        # src/api/routes_documents.py, MAX_UPLOAD_CHUNKS); 300s da margen
        # incluso para una subida cerca del límite corriendo en CPU.
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Administración (requieren X-Admin-Token)
# ---------------------------------------------------------------------------

def list_evaluations() -> dict:
    r = httpx.get(f"{API_URL}/admin/evaluations", headers=_admin_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def get_evaluation_detail(run_id: str) -> dict:
    r = httpx.get(
        f"{API_URL}/admin/evaluations/{quote(run_id, safe='')}",
        headers=_admin_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_admin_config() -> dict:
    r = httpx.get(f"{API_URL}/admin/config", headers=_admin_headers(), timeout=10)
    r.raise_for_status()
    return r.json()
