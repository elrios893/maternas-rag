"""
active_users.py — Registro compartido de usuarios activos del bot.

Persiste los chat_id de Telegram que han interactuado con el bot
en un archivo JSON, junto con metadatos de riesgo para scheduler.

Cada usuario almacena:
    - risk_points:       puntaje acumulado de riesgo
    - latest_risk_level: último nivel ("low"/"medium"/"high")
    - latest_risk_flags: banderas clínicas del último mensaje
    - last_activity:     timestamp ISO del último mensaje
    - last_check_sent:   timestamp ISO del último check enviado

Uso:
    register(chat_id, risk_level="medium")
    users = get_all()
    update_check_sent(chat_id)
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "active_users.json"
_lock = threading.Lock()

_RISK_POINTS = {"low": 0, "medium": 3, "high": 10}
_RISK_DECAY_PER_HOUR = 1
_RISK_MAX_POINTS = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_user_data() -> dict:
    return {
        "risk_points": 0,
        "latest_risk_level": "low",
        "latest_risk_flags": [],
        "last_activity": "",
        "last_check_sent": "",
    }


def _load() -> dict[str, dict]:
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        with _lock, open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            migrated = {str(cid): _default_user_data() for cid in data}
            _save(migrated)
            logger.info("active_users.json migrado de lista a dict")
            return migrated
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Error leyendo active_users.json: %s", e)
        return {}


def _save(users: dict[str, dict]) -> None:
    try:
        with _lock, open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error("Error escribiendo active_users.json: %s", e)


def _apply_decay(user: dict) -> None:
    if not user.get("last_activity") or user["risk_points"] <= 0:
        return
    try:
        last = datetime.fromisoformat(user["last_activity"])
        hours_idle = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        decay = int(hours_idle * _RISK_DECAY_PER_HOUR)
        if decay > 0:
            user["risk_points"] = max(0, user["risk_points"] - decay)
    except (ValueError, TypeError):
        pass


def register(
    chat_id: int,
    risk_level: str = "low",
    risk_flags: list[str] | None = None,
) -> None:
    key = str(chat_id)
    users = _load()
    now = _now_iso()
    points = _RISK_POINTS.get(risk_level, 0)

    if key in users:
        user = users[key]
        _apply_decay(user)
        user["risk_points"] = min(_RISK_MAX_POINTS, user["risk_points"] + points)
        user["latest_risk_level"] = risk_level
        if risk_flags:
            user["latest_risk_flags"] = risk_flags
    else:
        users[key] = {
            "risk_points": points,
            "latest_risk_level": risk_level,
            "latest_risk_flags": risk_flags or [],
            "last_activity": now,
            "last_check_sent": "",
        }

    users[key]["last_activity"] = now
    _save(users)


def update_check_sent(chat_id: int) -> None:
    key = str(chat_id)
    users = _load()
    if key in users:
        users[key]["last_check_sent"] = _now_iso()
        _save(users)


def get_all() -> dict[str, dict]:
    return _load()


def remove(chat_id: int) -> None:
    key = str(chat_id)
    users = _load()
    users.pop(key, None)
    _save(users)


def clear() -> None:
    _save({})
    logger.info("Registro de usuarios activos limpiado")
