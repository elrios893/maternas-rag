"""
status_scheduler.py — Envío periódico de check de estado vía Telegram.

Usa APScheduler con un job independiente por usuario, cada uno con su
propio intervalo según riesgo acumulado. Los jobs se ejecutan en paralelo
a través del thread pool de APScheduler.

Un job de sync periódico detecta usuarios nuevos, cambios de riesgo
o usuarios eliminados y ajusta los jobs automáticamente.

Arrancar:
    python src/bot/status_scheduler.py

Variables de entorno (ver .env.example):
    TELEGRAM_BOT_TOKEN                   — token del bot (requerido)
    STATUS_CHECK_BASE_INTERVAL_MINUTES   — intervalo base en min (default: 60)
    STATUS_CHECK_MIN_INTERVAL_MINUTES    — intervalo mínimo en min (default: 5)
    STATUS_CHECK_MESSAGE                 — texto del mensaje en Markdown
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.base import JobLookupError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bot.active_users import get_all, update_check_sent  # noqa: E402
from src.settings import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

TOKEN = settings.telegram_bot_token
BASE_INTERVAL = settings.status_check_base_interval_minutes
MIN_INTERVAL = settings.status_check_min_interval_minutes
MESSAGE = settings.status_check_message

# chat_id -> {"job_id": str, "last_risk_points": int}
_user_jobs: dict[str, dict] = {}

scheduler = BlockingScheduler()


def _user_interval(risk_points: int) -> float:
    return max(MIN_INTERVAL, BASE_INTERVAL / (1 + risk_points / 10))


def _send_to_user(chat_id: str) -> None:
    if not TOKEN:
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": int(chat_id),
        "text": MESSAGE,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            update_check_sent(int(chat_id))
            logger.debug("Check enviado a %s", chat_id)
        else:
            logger.error("Error al enviar a %s: HTTP %s", chat_id, resp.status_code)
    except requests.RequestException as e:
        logger.error("Error de conexión al enviar a %s: %s", chat_id, e)


def _add_user_job(chat_id: str, risk_points: int) -> None:
    interval = _user_interval(risk_points)
    job_id = f"user_{chat_id}"
    try:
        scheduler.add_job(
            _send_to_user,
            "interval",
            minutes=interval,
            id=job_id,
            args=[chat_id],
            replace_existing=True,
            name=f"Check usuario {chat_id} (riesgo {risk_points})",
        )
        _user_jobs[chat_id] = {"job_id": job_id, "last_risk_points": risk_points}
        logger.debug("Job creado para %s cada %.1f min", chat_id, interval)
    except Exception as e:
        logger.error("Error creando job para %s: %s", chat_id, e)


def _remove_user_job(chat_id: str) -> None:
    info = _user_jobs.pop(chat_id, None)
    if info:
        try:
            scheduler.remove_job(info["job_id"])
            logger.debug("Job eliminado para %s", chat_id)
        except JobLookupError:
            pass


def _sync_users() -> None:
    users = get_all()
    active = set(users.keys())
    existing = set(_user_jobs.keys())

    # Eliminar jobs de usuarios que ya no existen
    for chat_id in existing - active:
        _remove_user_job(chat_id)

    # Crear o actualizar jobs
    for chat_id, user_data in users.items():
        risk_points = user_data.get("risk_points", 0)

        if chat_id not in existing:
            _add_user_job(chat_id, risk_points)
        else:
            old_risk = _user_jobs[chat_id].get("last_risk_points", 0)
            if old_risk != risk_points:
                _remove_user_job(chat_id)
                _add_user_job(chat_id, risk_points)


def main() -> None:
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN no configurado en .env")
        sys.exit(1)

    # Sincronización inicial
    _sync_users()

    # Sync periódico cada 15s para detectar nuevos usuarios o cambios de riesgo
    scheduler.add_job(
        _sync_users,
        "interval",
        seconds=15,
        id="sync_users",
        name="Sincronización de usuarios activos",
    )

    logger.info(
        "Scheduler iniciado — base=%.1f min, min=%.1f min, %d usuario(s)",
        BASE_INTERVAL, MIN_INTERVAL, len(_user_jobs),
    )
    logger.info("Presiona Ctrl+C para detener.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler detenido por el usuario.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
