"""
routes_bot.py — Control del proceso del bot de Telegram desde el panel
de administración: estado, arranque, detención, reinicio y logs.

Router propio en vez de sumarlo a routes_admin.py: mismo motivo que
routes_documents.py — la dependencia de auth se declara una vez acá y
el diff de main.py queda en dos líneas. Toda la lógica de proceso vive
en bot_supervisor.py; este archivo es solo el mapeo a HTTP.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api import bot_supervisor
from src.api.auth import require_admin_token
from src.api.schemas import BotLogsResponse, BotStatusResponse

router = APIRouter(prefix="/admin/bot", tags=["bot"], dependencies=[Depends(require_admin_token)])


@router.get("/status", response_model=BotStatusResponse)
def bot_status() -> BotStatusResponse:
    return BotStatusResponse(**bot_supervisor.status())


@router.post("/start", response_model=BotStatusResponse)
def bot_start() -> BotStatusResponse:
    return BotStatusResponse(**bot_supervisor.start_bot())


@router.post("/stop", response_model=BotStatusResponse)
def bot_stop() -> BotStatusResponse:
    return BotStatusResponse(**bot_supervisor.stop_bot())


@router.post("/restart", response_model=BotStatusResponse)
def bot_restart() -> BotStatusResponse:
    return BotStatusResponse(**bot_supervisor.restart_bot())


@router.get("/logs", response_model=BotLogsResponse)
def bot_logs(limit: int = Query(200, ge=1, le=1000)) -> BotLogsResponse:
    return BotLogsResponse(lines=bot_supervisor.logs(limit))
