"""
app.py — Interfaz Streamlit del chatbot Maternas.

Conecta al backend FastAPI en http://localhost:8080
Arranca con: streamlit run src/ui/app.py
"""

import streamlit as st
import httpx

from src.ui.client import check_health
from src.ui.views.chat_view import render_chat

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
# Helpers
# ---------------------------------------------------------------------------


def risk_badge(level: str) -> str:
    labels = {"low": "🟢 Bajo", "medium": "🟡 Medio", "high": "🔴 ALTO"}
    css    = {"low": "badge-low", "medium": "badge-medium", "high": "badge-high"}
    label  = labels.get(level, level)
    klass  = css.get(level, "badge-low")
    return f'<span class="{klass}">{label}</span>'


def intent_label(intent: str) -> str:
    labels = {
        "control_prenatal":       "📅 Control prenatal",
        "signos_de_alarma":       "🚨 Signos de alarma",
        "sintomas_embarazo":      "🤰 Síntomas embarazo",
        "postparto":              "👶 Postparto",
        "lactancia":              "🍼 Lactancia",
        "salud_mental_perinatal": "💙 Salud mental",
        "medicamentos":           "💊 Medicamentos",
        "nutricion":              "🥗 Nutrición",
        "actividad_fisica":       "🏃 Actividad física",
        "planificacion_familiar": "📋 Planificación familiar",
        "consulta_administrativa":"📂 Administrativa",
        "pregunta_fuera_de_alcance": "❓ Fuera de alcance",
    }
    return labels.get(intent, intent)


def source_dataset_label(ds: str) -> str:
    labels = {
        "medmcqa":               "MedMCQA",
        "medqa_us":              "MedQA-US",
        "medqa_taiwan":          "MedQA-TW",
        "medqa_mainland":        "MedQA-ML",
        "multiclinsum_summary":  "Caso clínico (resumen)",
        "multiclinsum_fulltext": "Caso clínico (texto)",
        "textbook":              "Textbook médico",
    }
    return labels.get(ds, ds)


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

    if api_ok:
        st.success(f"API conectada")
        st.caption(f"📚 {health['total_vectors']:,} fragmentos médicos indexados")
        st.caption(f"🧠 {health['model'].split('/')[-1]}")
        st.session_state.api_ok = True
    else:
        st.error("API no disponible")
        st.caption("Arranca el servidor con:\n```\npython -m uvicorn src.api.main:app --port 8080\n```")
        st.session_state.api_ok = False

    st.divider()

    # Metadata del último turno
    if st.session_state.meta:
        m = st.session_state.meta[-1]
        st.subheader("Último turno")

        st.markdown(f"**Intención:** {intent_label(m.get('intent',''))}", unsafe_allow_html=True)
        st.markdown(f"**Riesgo:** {risk_badge(m.get('risk_level','low'))}", unsafe_allow_html=True)
        st.markdown(f"**Acción:** `{m.get('action','')}`")

        if m.get("risk_flags"):
            st.markdown("**Señales:**")
            for flag in m["risk_flags"]:
                st.markdown(f"- `{flag}`")

        if m.get("sources"):
            st.markdown("**Fuentes recuperadas:**")
            for s in m["sources"]:
                label = source_dataset_label(s.get("source_dataset",""))
                score = s.get("score", 0)
                st.markdown(
                    f'<span class="source-pill">{label} · {score:.3f}</span>',
                    unsafe_allow_html=True,
                )

        if m.get("tokens_used"):
            st.caption(f"Tokens usados: {m['tokens_used']:,}")

    st.divider()

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.meta     = []
        st.rerun()

# ---------------------------------------------------------------------------
# Funciones de renderizado de vistas
# ---------------------------------------------------------------------------


def _render_dashboard_placeholder() -> None:
    st.title("🏠 Dashboard")
    st.write("Contenido del Dashboard (placeholder)")


def _render_metrics_placeholder() -> None:
    st.title("📊 Métricas")
    st.write("Contenido de Métricas (placeholder)")


def _render_documents_placeholder() -> None:
    st.title("📁 Administración Documental del RAG")
    st.write("Contenido de Administración Documental (placeholder)")


def _render_settings_placeholder() -> None:
    st.title("⚙ Configuración")
    st.write("Contenido de Configuración (placeholder)")


# ---------------------------------------------------------------------------
# Dispatcher de vistas
# ---------------------------------------------------------------------------

if st.session_state.current_page == "chat":
    render_chat()
elif st.session_state.current_page == "dashboard":
    _render_dashboard_placeholder()
elif st.session_state.current_page == "metrics":
    _render_metrics_placeholder()
elif st.session_state.current_page == "documents":
    _render_documents_placeholder()
elif st.session_state.current_page == "settings":
    _render_settings_placeholder()
