"""
dashboard_view.py — Vista Dashboard del panel administrativo.

Solo lee st.session_state (poblado por app.py sidebar).
No importa client.py ni httpx.
"""

import streamlit as st


def _render_stat_card(title: str, value: str) -> None:
    with st.container(border=True):
        st.caption(title)
        st.write(value)


def render_dashboard() -> None:
    health = st.session_state.get("health", {})
    api_ok = st.session_state.get("api_ok", False)
    msg_count = len(st.session_state.get("messages", []))

    st.title("🏠 Dashboard")
    st.caption("Resumen del estado del sistema y de la sesión actual.")

    st.divider()

    # ------------------------------------------------------------------
    # KPI — variables independientes para facilitar crecimiento futuro
    # ------------------------------------------------------------------
    kpi_api = "Conectada" if api_ok else "Desconectada"
    kpi_model = health.get("model", "N/A").split("/")[-1]
    kpi_vectors = f"{health.get('total_vectors', 0):,}"
    kpi_conversation = f"{msg_count} mensajes"

    cols = st.columns(4)

    with cols[0]:
        _render_stat_card("🟢 Estado API", kpi_api)

    with cols[1]:
        _render_stat_card("🤖 Modelo IA", kpi_model)

    with cols[2]:
        _render_stat_card("📚 Fragmentos indexados", kpi_vectors)

    with cols[3]:
        _render_stat_card("💬 Conversación", kpi_conversation)

    st.write("")

    # ------------------------------------------------------------------
    # Fila inferior: Estado del sistema | Resumen de sesión
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("Estado del sistema")
            if api_ok:
                st.write("✅ API disponible")
                st.write("✅ Backend operativo")
                st.write("✅ Índice FAISS cargado")
                st.write("✅ Modelo listo")
            else:
                st.write("❌ API no disponible")
                st.write("❌ Backend no operativo")
                st.write("❌ Índice FAISS no disponible")
                st.write("❌ Modelo no disponible")

    with col_right:
        with st.container(border=True):
            st.subheader("Resumen de sesión")
            st.write(f"**Conversación activa:** {'Sí' if msg_count > 0 else 'No'}")
            st.write(f"**Mensajes:** {msg_count}")
            st.write("**Última actualización:** Ahora")
