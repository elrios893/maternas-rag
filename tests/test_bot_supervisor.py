"""
test_bot_supervisor.py — Ciclo de vida start/stop/restart del
supervisor del bot de Telegram (src/api/bot_supervisor.py).

Usa un script Python trivial en vez del bot real: a bot_supervisor solo
le importa poder lanzar `[sys.executable, <script>]` y capturar su
stdout/stderr, así que cualquier script sirve para probar el ciclo de
vida del proceso sin depender de un TELEGRAM_BOT_TOKEN válido ni de red.
"""

from __future__ import annotations

import time

import pytest

from src.api import bot_supervisor


@pytest.fixture(autouse=True)
def reset_supervisor_state():
    """El estado de bot_supervisor es un singleton de módulo — hay que
    partir siempre de 'sin proceso' y no dejar nada corriendo entre tests."""
    bot_supervisor.stop_bot()
    bot_supervisor._proc = None
    bot_supervisor._started_at = None
    bot_supervisor._logs.clear()
    bot_supervisor._manual_stop = False
    yield
    bot_supervisor.stop_bot()
    bot_supervisor._proc = None
    bot_supervisor._started_at = None


def _write_script(tmp_path, body: str):
    script = tmp_path / "fake_bot.py"
    script.write_text(body, encoding="utf-8")
    return script


SLEEPY_SCRIPT = (
    "import time\n"
    "print('hola', flush=True)\n"
    "time.sleep(30)\n"
)

FAILING_SCRIPT = (
    "import sys\n"
    "print('arrancando', flush=True)\n"
    "sys.exit(1)\n"
)


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestLifecycle:
    def test_status_inicial_no_corriendo(self):
        assert bot_supervisor.status()["running"] is False
        assert bot_supervisor.status()["pid"] is None

    def test_start_bot_queda_corriendo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot_supervisor, "BOT_SCRIPT", _write_script(tmp_path, SLEEPY_SCRIPT))

        status = bot_supervisor.start_bot()
        assert status["running"] is True
        assert status["pid"] is not None
        assert _wait_for(lambda: any("hola" in line for line in bot_supervisor.logs()))

    def test_start_dos_veces_no_clona_proceso(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot_supervisor, "BOT_SCRIPT", _write_script(tmp_path, SLEEPY_SCRIPT))

        first = bot_supervisor.start_bot()
        second = bot_supervisor.start_bot()
        assert first["pid"] == second["pid"]

    def test_stop_bot_lo_detiene(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot_supervisor, "BOT_SCRIPT", _write_script(tmp_path, SLEEPY_SCRIPT))

        bot_supervisor.start_bot()
        status = bot_supervisor.stop_bot()
        assert status["running"] is False

    def test_stop_sin_proceso_es_no_op(self):
        status = bot_supervisor.stop_bot()
        assert status["running"] is False

    def test_restart_cambia_el_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot_supervisor, "BOT_SCRIPT", _write_script(tmp_path, SLEEPY_SCRIPT))

        first = bot_supervisor.start_bot()
        second = bot_supervisor.restart_bot()
        assert second["running"] is True
        assert second["pid"] != first["pid"]

    def test_proceso_que_muere_solo_expone_exit_code(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot_supervisor, "BOT_SCRIPT", _write_script(tmp_path, FAILING_SCRIPT))

        bot_supervisor.start_bot()
        assert _wait_for(lambda: bot_supervisor.status()["running"] is False)
        status = bot_supervisor.status()
        assert status["exit_code"] == 1
        assert status["crashed"] is True

    def test_detener_a_proposito_no_se_marca_como_crashed(self, tmp_path, monkeypatch):
        # En Windows, terminate() deja el mismo exit_code (1) que un
        # crash real — 'crashed' es lo único que distingue "lo detuvo un
        # admin" de "se murió solo", y es justo lo que lee la consola.
        monkeypatch.setattr(bot_supervisor, "BOT_SCRIPT", _write_script(tmp_path, SLEEPY_SCRIPT))

        bot_supervisor.start_bot()
        status = bot_supervisor.stop_bot()
        assert status["running"] is False
        assert status["crashed"] is False

    def test_shutdown_detiene_un_bot_corriendo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot_supervisor, "BOT_SCRIPT", _write_script(tmp_path, SLEEPY_SCRIPT))

        bot_supervisor.start_bot()
        bot_supervisor.shutdown()
        assert bot_supervisor.status()["running"] is False

    def test_shutdown_sin_proceso_no_lanza(self):
        bot_supervisor.shutdown()   # no debe lanzar excepción
