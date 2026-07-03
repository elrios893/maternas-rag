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

import asyncio
import httpx
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.settings import settings

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
    user_id = update.effective_user.id
    text    = update.message.text.strip()

    if not text:
        return

    # Indicador de escritura
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    result = await call_chat(text, user_id)

    if result is None:
        await update.message.reply_text(
            "❌ No pude conectarme con el sistema. Verifica que la API esté funcionando."
        )
        return

    answer             = result.get("answer", "No se pudo generar una respuesta.")
    level              = result.get("risk_level", "low")
    intent             = result.get("intent", "")
    action             = result.get("action", "")
    flags              = result.get("risk_flags", [])
    needs_clarification = result.get("needs_clarification", False)

    # Si el sistema pide clarificación, no guardamos en historial todavía
    # — esperamos la respuesta del usuario al siguiente mensaje
    if not needs_clarification:
        if user_id not in histories:
            histories[user_id] = []
        histories[user_id].append({"role": "user",      "content": text})
        histories[user_id].append({"role": "assistant", "content": answer})

    # Caso clarificación — mensaje especial sin header de riesgo
    if needs_clarification:
        await update.message.reply_text(
            f"💬 {answer}"
        )
        return

    # Armar header de riesgo
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

    await update.message.reply_text(header, parse_mode=constants.ParseMode.HTML)
    await update.message.reply_text(answer[:4000])

async def handle_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Solo entiendo mensajes de texto. Por favor escribe tu pregunta."
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Error procesando update {update}: {context.error}")

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

    logger.info("Bot Maternas iniciado. Presiona Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
