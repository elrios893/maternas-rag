"""
chat_view.py — Vista del chat Maternas.

Solo contiene lógica de renderizado Streamlit.
La comunicación HTTP delega en src.ui.client.
"""

import streamlit as st
import httpx

from src.ui.client import call_chat


def render_chat() -> None:
    st.title("Maternas — Asistente de Salud Materna")
    st.caption("Respondo preguntas sobre embarazo, parto, postparto y lactancia basándome en literatura médica.")

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
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})

            with st.spinner("Consultando base de conocimiento médico..."):
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                try:
                    result = call_chat(user_input.strip(), history_payload)
                except httpx.ConnectError:
                    st.error("No se puede conectar con la API. ¿Está corriendo en el puerto 8080?")
                    return
                except httpx.TimeoutException:
                    st.error("La API tardó demasiado. Intenta de nuevo.")
                    return
                except httpx.HTTPStatusError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
                    return
                except Exception as e:
                    st.error(f"Error inesperado: {e}")
                    return

            answer = result.get("answer", "Sin respuesta")

            if result.get("risk_level") == "high":
                st.error("⚠️ Se detectaron señales de alarma. Busca atención médica de inmediato.")

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.meta.append(result)

            st.rerun()
