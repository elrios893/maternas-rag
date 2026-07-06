"""
app.py — Interfaz Streamlit del chatbot Maternas.

Conecta al backend FastAPI en http://localhost:8080
Arranca con: streamlit run src/ui/app.py
"""

import streamlit as st
import httpx

from src.ui.client import check_health
from src.ui.views.chat_view import render_chat
from src.ui.views.dashboard_view import render_dashboard
from src.ui.views.documents_view import render_documents

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAGE_LABELS = {
    "dashboard": "🏠 Dashboard",
    "chat":      "💬 Chat",
    "metrics":   "📊 Métricas",
    "documents": "📁 Documentos",
    "settings":  "⚙ Configuración",
}

st.set_page_config(
    page_title="Maternas — Asistente de Salud",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Burbuja usuario */
.msg-user {
    background: #e8f4fd;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    color: #1a1a2e;
}
/* Burbuja asistente */
.msg-assistant {
    background: #f0f7f0;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    color: #1a1a2e;
}
/* Badge de riesgo */
.badge-low    { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }
.badge-medium { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }
.badge-high   { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }
/* Pill de fuente */
.source-pill  { background:#e9ecef; color:#495057; padding:2px 8px; border-radius:8px; font-size:0.78em; margin:2px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []        # [{role, content}]
if "meta" not in st.session_state:
    st.session_state.meta = []            # metadata del último turno
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤰 Maternas")
    st.caption("Asistente de salud para madres gestantes")

    st.radio(
        "Navegación",
        options=list(PAGE_LABELS.keys()),
        format_func=lambda key: PAGE_LABELS[key],
        key="current_page",
        label_visibility="collapsed",
    )
    st.divider()

    # Estado de la API
    try:
        health = check_health()
        api_ok = bool(health.get("faiss_loaded"))
    except httpx.HTTPError:
        health = {}
        api_ok = False

    st.session_state.health = health
    st.session_state.api_ok = api_ok

    if api_ok:
        st.success(f"API conectada")
    else:
        st.error("API no disponible")
        st.caption("Arranca el servidor con:\n```\npython -m uvicorn src.api.main:app --port 8080\n```")

# ---------------------------------------------------------------------------
# Funciones de renderizado de vistas
# ---------------------------------------------------------------------------


def _render_metrics_placeholder() -> None:
    st.title("📊 Métricas")
    st.write("Contenido de Métricas (placeholder)")


def _render_settings_placeholder() -> None:
    st.title("⚙ Configuración")
    st.write("Contenido de Configuración (placeholder)")

# ---------------------------------------------------------------------------
# Dispatcher de vistas
# ---------------------------------------------------------------------------

if st.session_state.current_page == "chat":
    render_chat()
elif st.session_state.current_page == "dashboard":
    render_dashboard()
elif st.session_state.current_page == "metrics":
    _render_metrics_placeholder()
elif st.session_state.current_page == "documents":
    render_documents()
elif st.session_state.current_page == "settings":
    _render_settings_placeholder()
