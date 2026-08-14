"""
consent_gate.py — Aviso de tratamiento de datos, obligatorio antes de
usar cualquier página del panel.

A diferencia de la versión anterior de app.py (que abría el diálogo pero
seguía renderizando el chat detrás, bloqueando solo el formulario de
envío), acá el gate detiene el script con st.stop(): en un shell
multipágina, seguir adelante dejaría Documentos, Métricas y
Configuración accesibles detrás del modal.
"""

import streamlit as st

from src.consent import CONSENT_TEXT, FAREWELL_TEXT


@st.dialog("Aviso sobre tratamiento de información")
def _consent_dialog() -> None:
    st.text(CONSENT_TEXT)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Acepto", use_container_width=True):
            st.session_state.consent_status = "accepted"
            st.rerun()
    with col_b:
        if st.button("❌ No acepto", use_container_width=True):
            st.session_state.consent_status = "rejected"
            st.session_state.messages = []
            st.session_state.meta = []
            st.rerun()


def enforce_consent() -> None:
    """Bloquea toda la aplicación hasta que se acepte el aviso de datos."""
    if st.session_state.consent_status == "accepted":
        return

    _consent_dialog()

    if st.session_state.consent_status == "rejected":
        st.warning(FAREWELL_TEXT)
    else:
        st.info("Debes aceptar el aviso de tratamiento de información para poder chatear.")

    with st.form("chat_form_locked", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            locked_input = st.text_input(
                "Tu mensaje",
                placeholder="Acepta el aviso de datos para poder escribirme",
                label_visibility="collapsed",
            )
        with col2:
            locked_submitted = st.form_submit_button("Enviar", use_container_width=True)

    if locked_submitted and locked_input.strip():
        # Cualquier intento de enviar un mensaje sin aceptación vigente
        # vuelve a mostrar el aviso, en vez de procesar el mensaje.
        st.session_state.consent_status = None
        st.rerun()

    st.stop()
