"""
config_view.py — Vista de Configuración del panel administrativo.

La mayor parte es de solo lectura: configuración efectiva del backend
(GET /admin/config) con los secretos redactados a un booleano — nunca
su valor. Un subconjunto acotado de variables SÍ es editable desde el
formulario al final (PATCH /admin/config, ver src/api/routes_admin.py):
groq_model/groq_api_key, notifier_* y los 3 intervalos + token del bot
de Telegram. El resto de settings (rutas de datasets, embedding, etc.)
sigue sin ser editable desde acá — desincronizaría el proceso en
memoria con el FAISSStore singleton.
"""

import httpx
import streamlit as st

from src.ui.client import check_health, get_admin_config, restart_bot, update_admin_config

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
    if not st.session_state.get("is_admin"):
        st.error("Acceso restringido a administradores.")
        st.stop()

    st.title("⚙ Configuración")
    st.caption("Configuración efectiva del backend. La mayoría es solo lectura; el formulario "
               "al final permite editar un subconjunto acotado de variables.")
    st.divider()

    if not st.session_state.get("api_ok", False):
        st.warning("La API no está disponible. Inicia el servidor para ver la configuración.")
        return

    try:
        cfg = get_admin_config(st.session_state.admin_token)
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
            health = check_health()  # /health es público, sin token
            st.success(f"OK — {health.get('total_vectors', 0):,} vectores, modelo {health.get('model', '—')}")
        except httpx.HTTPError:
            st.error("No se pudo conectar con la API.")

    st.write("")
    st.divider()
    _render_edit_form(cfg)


def _render_edit_form(cfg: dict) -> None:
    """Formulario de edición de un subconjunto acotado de variables.

    Solo se manda al backend lo que realmente cambió respecto al valor
    cargado en `cfg` — así guardar un cambio de Groq no dispara de
    arrastre el aviso de "reinicia el bot" solo porque los intervalos
    viajaron con su mismo valor de siempre. Los campos password de
    secretos SIEMPRE arrancan vacíos (nunca se prellenan, ni enmascarados)
    y un envío vacío significa "sin cambios" — no borra el secreto
    existente.
    """
    st.subheader("Editar configuración")
    st.caption(
        "Los cambios de Groq y notificaciones aplican de inmediato. Los del bot de "
        "Telegram requieren reiniciarlo — el botón para hacerlo aparece abajo si aplica."
    )

    editable = cfg.get("editable", {})

    with st.form("edit_config_form"):
        st.markdown("**Groq**")
        groq_model = st.text_input("Modelo", value=editable.get("groq_model", ""))
        groq_api_key = st.text_input(
            "API Key", type="password", placeholder="Dejar en blanco para no cambiar",
        )

        st.markdown("**Notificaciones por email**")
        notifier_enabled = st.checkbox("Habilitadas", value=bool(editable.get("notifier_enabled", True)))
        notifier_email_to = st.text_input("Destinatario", value=editable.get("notifier_email_to", ""))
        notifier_smtp_user = st.text_input("Usuario SMTP", value=editable.get("notifier_smtp_user", ""))
        notifier_smtp_password = st.text_input(
            "Contraseña SMTP", type="password", placeholder="Dejar en blanco para no cambiar",
        )

        st.markdown("**Bot de Telegram**")
        telegram_bot_token = st.text_input(
            "Token del bot", type="password", placeholder="Dejar en blanco para no cambiar",
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            interval_low = st.number_input(
                "Intervalo check — riesgo bajo (s)", min_value=1.0,
                value=float(editable.get("status_check_interval_low_seconds", 60.0)),
            )
        with col_b:
            interval_medium = st.number_input(
                "Intervalo check — riesgo medio (s)", min_value=1.0,
                value=float(editable.get("status_check_interval_medium_seconds", 45.0)),
            )
        with col_c:
            interval_high = st.number_input(
                "Intervalo check — riesgo alto (s)", min_value=1.0,
                value=float(editable.get("status_check_interval_high_seconds", 30.0)),
            )

        submitted = st.form_submit_button("Guardar cambios")

    if not submitted:
        return

    payload = {}
    if groq_model.strip() and groq_model != editable.get("groq_model"):
        payload["groq_model"] = groq_model.strip()
    if groq_api_key:
        payload["groq_api_key"] = groq_api_key
    if notifier_enabled != bool(editable.get("notifier_enabled", True)):
        payload["notifier_enabled"] = notifier_enabled
    if notifier_email_to != editable.get("notifier_email_to"):
        payload["notifier_email_to"] = notifier_email_to
    if notifier_smtp_user != editable.get("notifier_smtp_user"):
        payload["notifier_smtp_user"] = notifier_smtp_user
    if notifier_smtp_password:
        payload["notifier_smtp_password"] = notifier_smtp_password
    if telegram_bot_token:
        payload["telegram_bot_token"] = telegram_bot_token
    if interval_low != editable.get("status_check_interval_low_seconds"):
        payload["status_check_interval_low_seconds"] = interval_low
    if interval_medium != editable.get("status_check_interval_medium_seconds"):
        payload["status_check_interval_medium_seconds"] = interval_medium
    if interval_high != editable.get("status_check_interval_high_seconds"):
        payload["status_check_interval_high_seconds"] = interval_high

    if not payload:
        st.info("No hay cambios para guardar.")
        return

    try:
        result = update_admin_config(st.session_state.admin_token, **payload)
    except httpx.HTTPStatusError as e:
        st.error(f"No se pudo guardar: {e.response.status_code} — {e.response.text[:200]}")
        return
    except httpx.HTTPError:
        st.error("No se pudo conectar con la API.")
        return

    st.success(f"Actualizado: {', '.join(result['updated'])}")

    if result.get("requires_bot_restart"):
        st.warning("Estos cambios requieren reiniciar el bot de Telegram para aplicarse.")
        if st.button("Reiniciar bot ahora"):
            try:
                restart_bot(st.session_state.admin_token)
            except httpx.HTTPError:
                st.error("No se pudo reiniciar el bot. Revisa la página Consola.")
            else:
                st.success("Bot reiniciado.")
                st.rerun()
    else:
        st.rerun()
