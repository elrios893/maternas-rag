"""
console_view.py — Consola de administración: estado y logs de la API y
del bot de Telegram, más control de arranque del bot.

Refresco manual (botón "Actualizar"), sin auto-poll — consistente con
que el resto del panel tampoco se refresca solo. El bot corre como
subproceso hijo de la API (src/api/bot_supervisor.py); la API no puede
reiniciarse a sí misma desde acá — ver la nota de alcance en
src/api/main.py.
"""

import httpx
import streamlit as st

from src.ui.client import (
    get_admin_logs,
    get_bot_logs,
    get_bot_status,
    restart_bot,
    start_bot,
    stop_bot,
)


def _fmt_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def render_console() -> None:
    if not st.session_state.get("is_admin"):
        st.error("Acceso restringido a administradores.")
        st.stop()

    st.title("🖥️ Consola")
    st.caption("Estado y logs de los procesos de Maternas.")
    st.divider()

    if not st.session_state.get("api_ok", False):
        st.warning("La API no está disponible.")
        return

    token = st.session_state.admin_token
    col_api, col_bot = st.columns(2)

    with col_api:
        _render_api_panel(token)

    with col_bot:
        _render_bot_panel(token)


def _render_api_panel(token: str) -> None:
    with st.container(border=True):
        st.subheader("🟢 API Maternas")
        st.caption("Este panel solo reporta su estado — reiniciarla es cosa de la terminal/proceso que la arrancó.")

        try:
            data = get_admin_logs(token, limit=200)
        except httpx.HTTPError:
            st.error("No se pudieron cargar los logs de la API.")
            return

        st.write(f"**Uptime:** {_fmt_uptime(data.get('uptime_seconds'))}")
        st.caption(f"Arrancó: {data.get('started_at', '—')}")

        with st.expander(f"Logs recientes ({len(data.get('lines', []))})"):
            if st.button("Actualizar", key="refresh_api_logs"):
                st.rerun()
            lines = data.get("lines", [])
            st.code("\n".join(lines) if lines else "Sin líneas todavía.", language="log")


def _render_bot_panel(token: str) -> None:
    with st.container(border=True):
        st.subheader("🤖 Bot de Telegram")

        try:
            status = get_bot_status(token)
        except httpx.HTTPError:
            st.error("No se pudo consultar el estado del bot.")
            return

        if status.get("running"):
            st.success(f"🟢 Corriendo — PID {status.get('pid')}")
            st.caption(f"Uptime: {_fmt_uptime(status.get('uptime_seconds'))} · arrancó: {status.get('started_at', '—')}")
        elif status.get("crashed"):
            st.error(f"⚠️ Se cerró solo (código {status.get('exit_code')}) — revisa TELEGRAM_BOT_TOKEN en Configuración.")
        else:
            st.info("🔴 Detenido")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Iniciar", use_container_width=True, disabled=bool(status.get("running"))):
                _run_action(start_bot, token, "Bot iniciado.")
        with col2:
            if st.button("⏹️ Detener", use_container_width=True, disabled=not status.get("running")):
                _run_action(stop_bot, token, "Bot detenido.")
        with col3:
            if st.button("🔄 Reiniciar", use_container_width=True):
                _run_action(restart_bot, token, "Bot reiniciado.")

        try:
            logs_data = get_bot_logs(token, limit=200)
        except httpx.HTTPError:
            st.error("No se pudieron cargar los logs del bot.")
            return

        with st.expander(f"Logs recientes ({len(logs_data.get('lines', []))})"):
            if st.button("Actualizar", key="refresh_bot_logs"):
                st.rerun()
            lines = logs_data.get("lines", [])
            st.code("\n".join(lines) if lines else "Sin líneas todavía.", language="log")


def _run_action(fn, token: str, success_msg: str) -> None:
    try:
        fn(token)
    except httpx.HTTPError as e:
        st.error(f"No se pudo completar la acción: {e}")
    else:
        st.success(success_msg)
        st.rerun()
