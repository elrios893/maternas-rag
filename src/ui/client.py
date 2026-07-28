"""
client.py — Cliente HTTP para la API del backend Maternas.

Sin dependencias de Streamlit. Las excepciones de httpx se propagan
hacia la capa de presentación.
"""

import httpx
from urllib.parse import quote

from src.settings import settings

API_URL     = settings.api_url
API_TIMEOUT = 60


def check_health() -> dict:
    r = httpx.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def call_chat(message: str, history: list) -> dict:
    payload = {"message": message, "history": history, "k": 5}
    r = httpx.post(f"{API_URL}/chat", json=payload, timeout=API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def list_documents(search: str = "", page: int = 1, per_page: int = 20) -> dict:
    r = httpx.get(
        f"{API_URL}/documents",
        params={"search": search, "page": page, "per_page": per_page},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_document_stats() -> dict:
    r = httpx.get(f"{API_URL}/documents/stats", timeout=10)
    r.raise_for_status()
    return r.json()


def get_document_detail(doc_id: str) -> dict:
    r = httpx.get(f"{API_URL}/documents/{quote(doc_id, safe='')}", timeout=10)
    r.raise_for_status()
    return r.json()
