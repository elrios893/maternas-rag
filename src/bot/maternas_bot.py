"""
maternas_bot.py — Bot de Telegram para Maternas.

Conecta el chatbot RAG con Telegram usando polling.
Requiere la API FastAPI corriendo en localhost:8080.

Arrancar:
    python src/bot/maternas_bot.py

Comandos:
    /start  — mensaje de bienvenida
    /help   — instrucciones de uso
    /reset  — reinicia la conversacion (borra historial)
    /stats  — info del sistema (vectores indexados)
"""

from __future__ import annotations

import httpx
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, Job
from src.settings import settings

from src.bot.active_users import (
    register as register_active_user,
    get_all as get_active_users,
    update_check_sent,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL      = "http://localhost:8080"
API_TIMEOUT  = 60
TOKEN        = settings.telegram_bot_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Historial conversacional en memoria: { user_id: [{"role": ..., "content": ...}] }
# Se pierde al reiniciar el bot — suficiente para MVP.
# ---------------------------------------------------------------------------

histories: dict[int, list[dict]] = {}

# ---------------------------------------------------------------------------
# Flag de salud de la API (para el scheduler)
# ---------------------------------------------------------------------------
# Se actualiza en cada handle_message(). Si la API no responde, el scheduler
# no enviará status checks (congruencia: no mandar check si el bot está caído).
# ---------------------------------------------------------------------------

_api_healthy: bool = False


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_chat(message: str, user_id: int) -> dict | None:
    payload = {
        "message": message,
        "history": histories.get(user_id, []),
        "k": 5,
    }
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            r = await client.post(f"{API_URL}/chat", json=payload)
            if r.status_code == 200:
                return r.json()
            logger.error(f"API error {r.status_code}: {r.text[:200]}")
    except httpx.ConnectError:
        logger.error("No se pudo conectar con la API en localhost:8080")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
    return None

# ---------------------------------------------------------------------------
# Risk indicators
# ---------------------------------------------------------------------------

def risk_emoji(level: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(level, "⚪")

def risk_label(level: str) -> str:
    return {"low": "Riesgo Bajo", "medium": "Riesgo Medio", "high": "🚨 RIESGO ALTO 🚨"}.get(level, level)

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        f"¡Hola {user.first_name}! 🤰\n\n"
        "Soy Maternas, un asistente de salud para madres gestantes.\n\n"
        "Puedes preguntarme sobre:\n"
        "• Síntomas del embarazo\n"
        "• Nutrición y ejercicios\n"
        "• Medicamentos y suplementos\n"
        "• Lactancia y postparto\n"
        "• Salud mental perinatal\n\n"
        "Comandos:\n"
        "/help  — más información\n"
        "/reset — reiniciar conversación\n"
        "/stats — estado del sistema\n\n"
        "⚠️ No reemplazo a un médico — si tienes una emergencia, busca atención profesional."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤰 **Maternas** — asistente de salud materna\n\n"
        "Escribe tu pregunta en lenguaje natural. El sistema:\n"
        "1. Clasifica tu intención (síntomas, nutrición, alarma, etc.)\n"
        "2. Evalúa el riesgo clínico\n"
        "3. Busca información médica relevante\n"
        "4. Genera una respuesta fundamentada\n\n"
        "Ejemplos:\n"
        "• \"¿Es normal tener náuseas a las 10 semanas?\"\n"
        "• \"¿Qué alimentos debo evitar?\"\n"
        "• \"Tengo dolor de cabeza intenso y veo manchas\"\n\n"
        "Si tienas una urgencia, por favor contacta a tu médico de inmediato.",
        parse_mode=constants.ParseMode.MARKDOWN,
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in histories:
        del histories[user_id]
    await update.message.reply_text("Conversación reiniciada. ¿En qué puedo ayudarte?")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{API_URL}/health")
            if r.status_code == 200:
                data = r.json()
                await update.message.reply_text(
                    f"📊 **Estado del sistema**\n\n"
                    f"• Fragmentos médicos: {data.get('total_vectors', 0):,}\n"
                    f"• Modelo embedding: {data.get('model', '?').split('/')[-1]}\n"
                    f"• FAISS cargado: {'✅' if data.get('faiss_loaded') else '❌'}\n"
                    f"• Usuarios activos en este turno: {len(histories)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text("❌ No se pudo conectar con la API.")
    except Exception:
        await update.message.reply_text("❌ API no disponible. Asegúrate de que el servidor esté corriendo en localhost:8080.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global _api_healthy

    user_id = update.effective_user.id
    text    = update.message.text.strip()

    if not text:
        return

    # Indicador de escritura
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    result = await call_chat(text, user_id)

    if result is None:
        _api_healthy = False
        await update.message.reply_text(
            "❌ No pude conectarme con el sistema. Verifica que la API esté funcionando."
        )
        return

    _api_healthy = True

    # Guardar en historial
    if user_id not in histories:
        histories[user_id] = []
    histories[user_id].append({"role": "user", "content": text})
    histories[user_id].append({"role": "assistant", "content": result.get("answer", "")})

    # Armar respuesta
    answer = result.get("answer", "No se pudo generar una respuesta.")
    level  = result.get("risk_level", "low")
    flags  = result.get("risk_flags", [])
    register_active_user(user_id, risk_level=level, risk_flags=flags)  # scheduler integration
    intent = result.get("intent", "")
    action = result.get("action", "")

    # Header con parse_mode HTML (controlado, no puede fallar)
    if level == "high":
        header = (
            f"🚨 <b>🚨 RIESGO ALTO — BUSCA ATENCIÓN MÉDICA INMEDIATA 🚨</b>\n\n"
            f"<b>Señales detectadas:</b> {', '.join(flags)}\n"
            f"<b>Acción recomendada:</b> {action}\n\n"
        )
    elif level == "medium":
        header = f"{risk_emoji(level)} <b>{risk_label(level)}</b>\n"
        if flags:
            header += f"<b>Detectado:</b> {', '.join(flags)}\n"
        header += "<b>Recomendación:</b> consulta a tu médico en los próximos días.\n\n"
    else:
        header = f"{risk_emoji(level)} <b>{risk_label(level)}</b>\n\n"

    # Enviar header con formato HTML
    await update.message.reply_text(header, parse_mode=constants.ParseMode.HTML)

    # Enviar respuesta como texto plano (sin parse_mode para evitar errores del LLM)
    max_chars = 4000
    await update.message.reply_text(answer[:max_chars])

async def handle_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Solo entiendo mensajes de texto. Por favor escribe tu pregunta."
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Error procesando update {update}: {context.error}")

# ---------------------------------------------------------------------------
# Status Check Scheduler (integrado — reemplaza status_scheduler.py)
# ---------------------------------------------------------------------------
# Cada usuario activo recibe un mensaje periódico de "check de estado".
# La frecuencia depende del nivel de riesgo:
#   LOW    → cada STATUS_CHECK_INTERVAL_LOW_SECONDS    (default: 60s)
#   MEDIUM → cada STATUS_CHECK_INTERVAL_MEDIUM_SECONDS (default: 45s)
#   HIGH   → cada STATUS_CHECK_INTERVAL_HIGH_SECONDS   (default: 30s)
#
# El scheduler solo envía checks si _api_healthy == True. Esto asegura
# congruencia: si el bot no puede hablar con la API, no se mandan
# mensajes como si todo funcionara.
# ---------------------------------------------------------------------------

_user_status_jobs: dict[str, Job] = {}
_user_risk_levels: dict[str, str] = {}


def _check_interval(risk_level: str) -> float:
    return {
        "low":    settings.status_check_interval_low_seconds,
        "medium": settings.status_check_interval_medium_seconds,
        "high":   settings.status_check_interval_high_seconds,
    }.get(risk_level, settings.status_check_interval_low_seconds)


async def _send_status_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envía el mensaje de check de estado a un usuario (solo si API responde)."""
    if not _api_healthy:
        logger.debug("API no disponible — status check omitido")
        return

    chat_id = context.job.data
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=settings.status_check_message,
            parse_mode=constants.ParseMode.MARKDOWN,
        )
        update_check_sent(chat_id)
        logger.debug("Status check enviado a %s", chat_id)
    except Exception as e:
        logger.warning("Error enviando status check a %s: %s", chat_id, e)


async def _sync_user_jobs(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sincroniza los jobs de status check con active_users.json."""
    users = get_active_users()
    active = set(users.keys())
    existing = set(_user_status_jobs.keys())

    # Eliminar jobs de usuarios que ya no existen
    for chat_id in existing - active:
        _user_status_jobs.pop(chat_id, None).schedule_removal()
        _user_risk_levels.pop(chat_id, None)
        logger.debug("Job eliminado para chat %s", chat_id)

    # Crear o actualizar jobs según riesgo
    for chat_id, user_data in users.items():
        risk_level = user_data.get("latest_risk_level", "low")
        interval = _check_interval(risk_level)

        if chat_id not in existing:
            job = context.job_queue.run_repeating(
                _send_status_check,
                interval=interval,
                first=interval,
                data=int(chat_id),
                name=f"status_check_{chat_id}",
            )
            _user_status_jobs[chat_id] = job
            _user_risk_levels[chat_id] = risk_level
            logger.debug(
                "Job creado para %s — riesgo=%s, cada %.0fs",
                chat_id, risk_level, interval,
            )
        else:
            prev = _user_risk_levels.get(chat_id)
            if prev != risk_level:
                _user_status_jobs[chat_id].schedule_removal()
                job = context.job_queue.run_repeating(
                    _send_status_check,
                    interval=interval,
                    first=interval,
                    data=int(chat_id),
                    name=f"status_check_{chat_id}",
                )
                _user_status_jobs[chat_id] = job
                _user_risk_levels[chat_id] = risk_level
                logger.debug(
                    "Job re-programado para %s — riesgo %s→%s, cada %.0fs",
                    chat_id, prev, risk_level, interval,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN no configurado en .env")
        sys.exit(1)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(~filters.TEXT, handle_non_text))
    app.add_error_handler(error_handler)

    # ── Status Check Scheduler (integrado) ──
    app.job_queue.run_repeating(
        _sync_user_jobs,
        interval=15,
        first=5,
        name="sync_user_jobs",
    )
    logger.info("Scheduler de status check integrado — sync cada 15s")

    logger.info("Bot Maternas iniciado. Presiona Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
