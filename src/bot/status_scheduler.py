"""
status_scheduler.py — Envío periódico de check de estado vía Telegram.

Usa APScheduler para enviar un mensaje configurable a TODOS los usuarios
que han interactuado con el bot, en intervalos regulares.

El registro de usuarios activos se mantiene en active_users.json,
actualizado por el bot cuando alguien le escribe.

Arrancar:
    python src/bot/status_scheduler.py

Variables de entorno (ver .env.example):
    TELEGRAM_BOT_TOKEN            — token del bot (requerido)
    STATUS_CHECK_INTERVAL_MINUTES — intervalo en minutos (default: 10, usar 0.5 = 30 s)
    STATUS_CHECK_MESSAGE          — texto del mensaje en Markdown (default: ver abajo)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bot.active_users import get_all  # noqa: E402
from src.settings import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKEN = settings.telegram_bot_token
INTERVAL_MINUTES = settings.status_check_interval_minutes
MESSAGE = settings.status_check_message

# ---------------------------------------------------------------------------
# Envío del mensaje a todos los usuarios activos
# ---------------------------------------------------------------------------


def send_status_check() -> None:
    if not TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN no configurado")
        return

    users = get_all()
    if not users:
        logger.info("No hay usuarios activos registrados aún")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    success = 0
    failed = 0

    for chat_id in users:
        payload = {
            "chat_id": chat_id,
            "text": MESSAGE,
            "parse_mode": "Markdown",
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                success += 1
            else:
                failed += 1
                logger.error(
                    "Error al enviar a %s: HTTP %s — %s",
                    chat_id,
                    resp.status_code,
                    resp.text[:200],
                )
        except requests.RequestException as e:
            failed += 1
            logger.error("Error de conexión al enviar a %s: %s", chat_id, e)

    logger.info(
        "Check de estado enviado a %d usuario(s): %d ok, %d error(es)",
        len(users),
        success,
        failed,
    )


# ---------------------------------------------------------------------------
# Inicio
# ---------------------------------------------------------------------------


def main() -> None:
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN no configurado en .env")
        sys.exit(1)

    scheduler = BlockingScheduler()

    scheduler.add_job(
        send_status_check,
        "interval",
        minutes=INTERVAL_MINUTES,
        id="status_check",
        name="Envío periódico de check de estado",
    )

    logger.info(
        "Scheduler iniciado — enviando cada %s minuto(s) a %d usuario(s) registrado(s)",
        INTERVAL_MINUTES,
        len(get_all()),
    )
    logger.info("Presiona Ctrl+C para detener.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler detenido por el usuario.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
