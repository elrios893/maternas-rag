"""
app.py — Interfaz Streamlit del chatbot Maternas.

Conecta al backend FastAPI en http://localhost:8080
Arranca con: streamlit run src/ui/app.py
"""

import streamlit as st
import httpx
import json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL     = "http://localhost:8080"
API_TIMEOUT = 60

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_health() -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def call_chat(message: str, history: list) -> dict | None:
    payload = {"message": message, "history": history, "k": 5}
    try:
        r = httpx.post(f"{API_URL}/chat", json=payload, timeout=API_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text[:200]}")
    except httpx.ConnectError:
        st.error("No se puede conectar con la API. ¿Está corriendo en el puerto 8080?")
    except httpx.TimeoutException:
        st.error("La API tardó demasiado. Intenta de nuevo.")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
    return None


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
    st.divider()

    # Estado de la API
    health = check_health()
    if health and health.get("faiss_loaded"):
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
# Área principal — historial de chat
# ---------------------------------------------------------------------------

st.title("Maternas — Asistente de Salud Materna")
st.caption("Respondo preguntas sobre embarazo, parto, postparto y lactancia basándome en literatura médica.")

# Mostrar historial
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">👤 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-assistant">🤰 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Input del usuario
# ---------------------------------------------------------------------------

if not st.session_state.api_ok:
    st.warning("La API no está disponible. Inicia el servidor para continuar.")
else:
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            user_input = st.text_input(
                "Tu mensaje",
                placeholder="Ej: ¿Es normal tener náuseas a las 10 semanas?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Enviar", use_container_width=True)

    if submitted and user_input.strip():
        # Agregar mensaje del usuario al historial visual
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        # Llamar a la API
        with st.spinner("Consultando base de conocimiento médico..."):
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]   # sin el mensaje actual
            ]
            result = call_chat(user_input.strip(), history_payload)

        if result:
            answer = result.get("answer", "Sin respuesta")

            # Alerta visual para riesgo alto
            if result.get("risk_level") == "high":
                st.error("⚠️ Se detectaron señales de alarma. Busca atención médica de inmediato.")

            # Agregar respuesta al historial
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Guardar metadata del turno
            st.session_state.meta.append(result)

            st.rerun()
