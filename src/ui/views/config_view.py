"""
config_view.py — Vista de Configuración del panel administrativo.

Solo lectura: muestra la configuración efectiva del backend (GET
/admin/config) con los secretos redactados a un booleano. Editar
settings desde acá no es un caso soportado — desincronizaría .env, el
proceso en memoria y el FAISSStore singleton.
"""

import httpx
import streamlit as st

from src.ui.client import check_health, get_admin_config

SECRET_LABELS = {
    "groq_api_key": "Groq API Key",
    "groq_api_key_2": "Groq API Key (secundaria, juez de Ragas)",
    "telegram_bot_token": "Token del bot de Telegram",
    "admin_api_token": "Token de administración",
    "openrouter_key": "OpenRouter",
    "cerebras_key": "Cerebras",
    "active_users_encryption_key": "Cifrado de usuarios activos del bot",
    "notifier_smtp_password": "Contraseña SMTP del notificador",
}


def render_config() -> None:
    st.title("⚙ Configuración")
    st.caption("Configuración efectiva del backend. Solo lectura — los secretos se muestran redactados.")
    st.divider()

    if not st.session_state.get("api_ok", False):
        st.warning("La API no está disponible. Inicia el servidor para ver la configuración.")
        return

    try:
        cfg = get_admin_config()
    except httpx.HTTPError:
        st.error("No se pudo cargar la configuración del backend.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.subheader("Embedding")
            st.write(f"**Modelo:** {cfg.get('embedding_model', '—')}")
            st.write(f"**Dispositivo:** {cfg.get('embedding_device', '—')}")

        with st.container(border=True):
            st.subheader("Retrieval")
            st.write(f"**Fragmentos por consulta (k):** {cfg.get('rag_top_k', '—')}")
            st.write("**Fuentes activas en el índice denso:**")
            for src in cfg.get("dense_sources", []):
                st.caption(f"• {src}")

    with col_right:
        with st.container(border=True):
            st.subheader("LLM")
            st.write(f"**Modelo Groq:** {cfg.get('groq_model', '—')}")

        with st.container(border=True):
            st.subheader("Índice FAISS")
            build_info = cfg.get("index_build_info", {})
            if build_info:
                st.write(f"**Vectores:** {build_info.get('total_vectors', 0):,}")
                st.write(f"**Dimensión:** {build_info.get('dimension', '—')}")
                st.write(f"**Guardado:** {build_info.get('saved_at', '—')}")
            else:
                st.write("Sin información de build disponible.")

    st.write("")

    with st.container(border=True):
        st.subheader("Secretos configurados")
        st.caption("Solo se muestra si están definidos — nunca su valor.")
        secrets = cfg.get("secrets_configured", {})
        cols = st.columns(2)
        for i, (key, configured) in enumerate(secrets.items()):
            label = SECRET_LABELS.get(key, key)
            icon = "✅" if configured else "⬜"
            with cols[i % 2]:
                st.write(f"{icon} {label}")

    st.write("")

    if st.button("🔌 Probar conexión con la API"):
        try:
            health = check_health()
            st.success(f"OK — {health.get('total_vectors', 0):,} vectores, modelo {health.get('model', '—')}")
        except httpx.HTTPError:
            st.error("No se pudo conectar con la API.")
