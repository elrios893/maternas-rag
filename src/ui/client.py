"""
client.py — Cliente HTTP para la API del backend Maternas.

Sin dependencias de Streamlit. Las excepciones de httpx se propagan
hacia la capa de presentación.
"""

import httpx

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
