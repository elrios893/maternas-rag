"""
active_users.py — Registro compartido de usuarios activos del bot.

Persiste los chat_id de Telegram que han interactuado con el bot
en un archivo JSON, para que el scheduler pueda enviarles mensajes.

Uso:
    from src.bot.active_users import register, get_all

    register(123456789)           # al recibir un mensaje
    users = get_all()             # para enviar a todos
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "active_users.json"
_lock = threading.Lock()


def _load() -> set[int]:
    if not _REGISTRY_PATH.exists():
        return set()
    try:
        with _lock, open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Error leyendo active_users.json: %s", e)
        return set()


def _save(users: set[int]) -> None:
    try:
        with _lock, open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(users), f, ensure_ascii=False)
    except OSError as e:
        logger.error("Error escribiendo active_users.json: %s", e)


def register(chat_id: int) -> None:
    users = _load()
    if chat_id in users:
        return
    users.add(chat_id)
    _save(users)
    logger.debug("Usuario %s registrado como activo", chat_id)


def get_all() -> set[int]:
    return _load()


def remove(chat_id: int) -> None:
    users = _load()
    users.discard(chat_id)
    _save(users)


def clear() -> None:
    _save(set())
    logger.info("Registro de usuarios activos limpiado")
