"""
auth.py — Autenticación del panel de administración.

Token compartido en la cabecera X-Admin-Token, validado contra
settings.admin_api_token. Sin dependencias nuevas.

Protege TODOS los endpoints /documents* y /admin*, incluidas las lecturas:
GET /documents/{doc_id} devuelve el texto completo del corpus y CORS es
abierto (allow_origins=["*"]), así que un listado sin auth sería un volcado
completo y scriptable del material — el mismo cuyo licenciamiento el
proyecto ya tuvo que limpiar una vez (ver retriever.py, DENSE_SOURCES).

Fail-closed: sin ADMIN_API_TOKEN configurado, el panel responde 503 en
vez de quedar abierto. Estos endpoints mutan un índice de producción
compartido con /chat; "alguien olvidó una env var" no debe equivaler
en silencio a "cualquiera que alcance el puerto puede apagar el retrieval".
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from src.settings import settings


def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    expected = (settings.admin_api_token or "").strip()

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Panel de administración deshabilitado: configura ADMIN_API_TOKEN en .env",
        )

    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de administración inválido o ausente",
        )
