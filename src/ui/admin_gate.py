"""
admin_gate.py — Desbloqueo del panel administrativo dentro de la sesión.

Sin esto, cualquier persona que abriera la app vería Documentos,
Métricas y Configuración en la barra de navegación: st.navigation solo
construye el menú a partir de la lista de páginas que le pasamos, así
que ocultarlas ahí (ver app.py) es lo que las hace inalcanzables, no
solo invisibles — una sesión sin is_admin nunca recibe esas páginas en
la lista, así que st.navigation no puede resolverlas ni por URL directa.

El token se valida contra la API real (GET /admin/config) y no contra
settings.admin_api_token leído en el proceso de Streamlit: así una
sesión que ingresa el token equivocado nunca queda marcada como admin,
incluso si UI y API llegaran a correr con .env distintos, y reutiliza
la única fuente de verdad que ya existe (src/api/auth.py) en vez de
duplicar la comparación acá.

st.session_state.admin_token vive solo en esta sesión de navegador —
nunca en una variable de módulo, que en modo servidor de Streamlit se
comparte entre las sesiones concurrentes de otros usuarios.
"""

import httpx
import streamlit as st

from src.ui.client import get_admin_config


def is_admin() -> bool:
    return bool(st.session_state.get("is_admin"))


def _try_login(token: str) -> None:
    token = token.strip()
    if not token:
        st.session_state.admin_login_error = "Ingresa un token."
        return
    try:
        get_admin_config(token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            st.session_state.admin_login_error = "Panel deshabilitado: falta ADMIN_API_TOKEN en el backend."
        else:
            st.session_state.admin_login_error = "Token inválido."
        return
    except httpx.HTTPError:
        st.session_state.admin_login_error = "No se pudo conectar con la API."
        return

    st.session_state.is_admin = True
    st.session_state.admin_token = token
    st.session_state.admin_login_error = None


def render_admin_gate() -> None:
    """Widget de sidebar: entrar o salir del modo administrador."""
    st.divider()

    if is_admin():
        st.success("🔓 Modo administrador")
        if st.button("Cerrar sesión admin", use_container_width=True):
            st.session_state.is_admin = False
            st.session_state.admin_token = ""
            st.rerun()
        return

    st.caption("🔐 Acceso administrador")
    token_input = st.text_input(
        "Token de administración",
        type="password",
        key="admin_token_input",
        label_visibility="collapsed",
        placeholder="Token de administración",
    )
    if st.button("Entrar", use_container_width=True, key="admin_login_btn"):
        _try_login(token_input)
        st.rerun()

    error = st.session_state.get("admin_login_error")
    if error:
        st.error(error)
