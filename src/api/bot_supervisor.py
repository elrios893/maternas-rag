"""
bot_supervisor.py — Arranca/detiene src/bot/maternas_bot.py como
subproceso hijo del proceso de la API, para poder controlarlo desde el
panel de administración sin depender de una terminal aparte.

Usa el mismo comando que ya funciona manualmente (`python
src/bot/maternas_bot.py`, ver el docstring de ese archivo) — no `-m`,
para no arriesgar un comportamiento distinto al documentado.

Todo el estado (proceso, buffer de logs) es un singleton de módulo
protegido por un lock, igual que _store/_index_lock en
src/ingestion/store.py — dos clicks de "Reiniciar" no deben poder
lanzar dos procesos del bot a la vez.

Nota Windows: Popen.terminate() en Windows llama a TerminateProcess, no
hay señal SIGTERM real — es un corte duro, no un apagado ordenado.
Aceptable acá: el bot no tiene estado en vuelo que deba flushear entre
turnos, cada update de Telegram es corto y stateless a nivel de proceso.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOT_SCRIPT = PROJECT_ROOT / "src" / "bot" / "maternas_bot.py"

_LOG_MAXLEN = 1000

_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_started_at: Optional[datetime] = None
_logs: deque[str] = deque(maxlen=_LOG_MAXLEN)

# Distingue "lo detuvo un admin" de "se murió solo": en Windows,
# Popen.terminate() llama a TerminateProcess, que deja un exit_code no
# nulo (típicamente 1) igual que un sys.exit(1) por token inválido — sin
# esta bandera, apretar "Detener" se vería en la consola idéntico a un
# crash real.
_manual_stop = False


def _reader(proc: subprocess.Popen) -> None:
    """Corre en un hilo aparte: vuelca stdout+stderr del subproceso al
    buffer de logs línea por línea, hasta que el proceso termina."""
    assert proc.stdout is not None
    for line in proc.stdout:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        _logs.append(f"[{ts}] {line.rstrip()}")
    _logs.append(f"[proceso terminado, code={proc.poll()}]")


def start_bot() -> dict:
    global _proc, _started_at, _manual_stop
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return status()   # ya corriendo — no-op, no clona un segundo proceso

        _logs.clear()
        _manual_stop = False
        _proc = subprocess.Popen(
            [sys.executable, str(BOT_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _started_at = datetime.now(timezone.utc)
        threading.Thread(target=_reader, args=(_proc,), daemon=True).start()
        logger.info("[BotSupervisor] Bot iniciado, pid=%s", _proc.pid)
        return status()


def stop_bot(timeout: float = 10.0) -> dict:
    global _proc, _manual_stop
    with _lock:
        if _proc is None or _proc.poll() is not None:
            return status()   # ya detenido — no-op

        _manual_stop = True
        _proc.terminate()
        try:
            _proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _proc.kill()
            _proc.wait(timeout=5)
        logger.info("[BotSupervisor] Bot detenido")
        return status()


def restart_bot() -> dict:
    stop_bot()
    return start_bot()


def status() -> dict:
    running = _proc is not None and _proc.poll() is None
    uptime = None
    if running and _started_at is not None:
        uptime = (datetime.now(timezone.utc) - _started_at).total_seconds()
    return {
        "running": running,
        "pid": _proc.pid if running else None,
        "started_at": _started_at.isoformat() if (running and _started_at) else None,
        "uptime_seconds": uptime,
        "exit_code": (_proc.poll() if (_proc is not None and not running) else None),
        # True solo si terminó por su cuenta (p. ej. TELEGRAM_BOT_TOKEN
        # inválido → sys.exit(1) inmediato) — False si lo detuvo un admin
        # con stop_bot()/restart_bot(), aunque el exit_code sea el mismo.
        "crashed": (not running and _proc is not None and not _manual_stop),
    }


def logs(limit: int = 200) -> list[str]:
    return list(_logs)[-limit:]


def shutdown() -> None:
    """Llamado desde el lifespan de la API al apagarse — evita dejar el
    subproceso del bot huérfano cuando se detiene uvicorn."""
    if _proc is not None and _proc.poll() is None:
        stop_bot(timeout=5)
