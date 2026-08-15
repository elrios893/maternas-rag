"""
documents_view.py — Vista del Centro de Gestión Documental del RAG.

Portado desde la rama feature/cristian-admin-panel con correcciones:
  - PER_PAGE / DETAIL_PER_PAGE como constantes, no repetidas.
  - _load_document_data solo marca "cargado" en éxito, para poder
    reintentar tras un fallo transitorio (la rama lo marcaba antes).
  - La búsqueda vuelve a página 1 (la rama re-consultaba la página
    actual, que podía quedar fuera de rango).
  - _upload_error_message cubre 401/503 (token de administración).
  - El detalle usa return en vez de st.stop() dentro del diálogo.
  - El detalle pagina los fragmentos (backend: GET /documents/{id}
    ahora acepta page/per_page).
  - Ya no hace falta filtrar 'doc_id' de chunk['metadata']: el backend
    (TYPED_CHUNK_FIELDS) lo excluye en el origen.
"""

import math

import httpx
import streamlit as st

from src.ui.client import (
    get_document_detail,
    get_document_stats,
    list_documents,
    toggle_document_status,
    upload_document,
)

PER_PAGE = 20
DETAIL_PER_PAGE = 20


def _render_stat_card(title: str, value: str) -> None:
    with st.container(border=True):
        st.caption(title)
        st.write(value)


def _load_document_data() -> None:
    if st.session_state.get("doc_loaded"):
        return
    token = st.session_state.admin_token
    try:
        with st.spinner("Cargando base documental..."):
            stats = get_document_stats(token)
            resp = list_documents(token, search="", page=1, per_page=PER_PAGE)
        st.session_state.doc_stats = stats
        st.session_state.documents = resp.get("documents", [])
        st.session_state.doc_total = resp.get("total", 0)
        st.session_state.doc_page = 1
        st.session_state.doc_search_active = ""
        st.session_state.doc_loaded = True
        st.session_state.doc_load_error = False
    except httpx.HTTPError:
        # No se marca doc_loaded: un fallo transitorio (API reiniciando,
        # red) permite reintentar en el próximo rerun en vez de quedar
        # con una lista vacía para siempre.
        st.session_state.doc_stats = {}
        st.session_state.documents = []
        st.session_state.doc_total = 0
        st.session_state.doc_page = 1
        st.session_state.doc_load_error = True


def _fetch_documents(search: str = "", page: int = 1) -> None:
    token = st.session_state.admin_token
    with st.spinner("Buscando documentos..."):
        try:
            stats = get_document_stats(token)
            resp = list_documents(token, search=search, page=page, per_page=PER_PAGE)
            st.session_state.doc_stats = stats
            st.session_state.documents = resp.get("documents", [])
            st.session_state.doc_total = resp.get("total", 0)
            st.session_state.doc_page = page
            st.session_state.doc_search_active = search
        except httpx.HTTPError:
            st.session_state.documents = []
            st.session_state.doc_total = 0
            st.session_state.doc_page = 1


def _on_search_change() -> None:
    """Una búsqueda nueva siempre vuelve a página 1: quedarse en la
    página actual podría dejarla fuera de rango para el nuevo filtro."""
    _fetch_documents(search=st.session_state.get("doc_search", ""), page=1)


def _upload_error_message(exc: httpx.HTTPStatusError) -> str:
    """Convierte un error HTTP del upload en un mensaje legible."""
    try:
        detail = exc.response.json().get("detail", "")
    except ValueError:
        detail = ""
    status = exc.response.status_code

    if status == 400:
        return f"Archivo inválido: {detail or 'extensión, codificación o contenido no permitidos'}"
    if status == 401:
        return "Token de administración inválido. Revisa ADMIN_API_TOKEN en .env."
    if status == 409:
        return f"Documento duplicado: {detail or 'ya existe un documento con ese nombre'}"
    if status == 413:
        return f"Archivo demasiado grande o con demasiados fragmentos: {detail or 'reduce el archivo'}"
    if status == 500:
        return f"Error al indexar el documento: {detail or 'intenta de nuevo'}"
    if status == 503:
        return "Panel de administración deshabilitado: configura ADMIN_API_TOKEN en .env del backend."
    return f"Error del servidor ({status}): {detail or 'respuesta desconocida'}"


def _render_document_card(doc: dict) -> None:
    doc_id = doc.get("doc_id", "—")
    source = doc.get("source_dataset", "—")
    chunk_count = doc.get("chunk_count", 0)
    total_chars = doc.get("total_chars", 0)
    size_str = f"{total_chars:,}" if total_chars else "—"
    is_active = doc.get("active", True)
    status_text = "✅ Indexado" if is_active else "⏸️ Inactivo"

    with st.container(border=True):
        st.write(f"📄 **{doc_id}**")
        st.caption(f"{source} · {chunk_count} fragmentos · {size_str} caracteres")
        st.caption(f"Estado: {status_text}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ver detalles", key=f"detail_{doc_id}", use_container_width=True):
                st.session_state.doc_detail_page = 1
                _show_document_detail(doc_id)
        with col2:
            btn_label = "Activar" if not is_active else "Desactivar"
            if st.button(btn_label, key=f"toggle_{doc_id}", use_container_width=True):
                if is_active:
                    _confirm_deactivate(doc_id)
                else:
                    _toggle_doc_status(doc_id, active=True)


def _fmt_bool(v):
    if v is True:
        return "✅ Sí"
    if v is False:
        return "❌ No"
    return str(v)


_MARKERS = frozenset({
    "[QUESTION]", "[ANSWER]", "[EXPLANATION]",
    "[SOURCE]", "[OPTIONS]", "[SUBJECT]", "[TOPIC]",
})


def _render_chunk_text(text: str) -> None:
    if not any(m in text for m in _MARKERS):
        st.write(text)
        return
    formatted = text
    for m in _MARKERS:
        formatted = formatted.replace(m, f"**{m}**")
    st.write(formatted)


def _toggle_doc_status(doc_id: str, active: bool) -> None:
    try:
        toggle_document_status(st.session_state.admin_token, doc_id, active=active)
        st.session_state.doc_loaded = False
        st.rerun()
    except httpx.HTTPError:
        st.error("Error al cambiar el estado del documento")


@st.dialog("Confirmar desactivación")
def _confirm_deactivate(doc_id: str) -> None:
    st.write(f"¿Desactivar el documento **{doc_id}**?")
    st.write("Los fragmentos de este documento dejarán de participar en las respuestas del asistente.")
    st.write("Puedes reactivarlo en cualquier momento desde el panel.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancelar", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Desactivar", use_container_width=True, type="primary"):
            _toggle_doc_status(doc_id, active=False)


@st.dialog("Detalle del documento")
def _show_document_detail(doc_id: str) -> None:
    page = st.session_state.get("doc_detail_page", 1)

    with st.spinner("Cargando detalle..."):
        try:
            detail = get_document_detail(st.session_state.admin_token, doc_id, page=page, per_page=DETAIL_PER_PAGE)
        except httpx.HTTPError:
            st.error("No se pudo cargar el detalle del documento.")
            return

    st.write(f"**{detail['doc_id']}**")
    estado = "✅ Indexado" if detail.get("active", True) else "⏸️ Inactivo"
    st.caption(
        f"{detail['source_dataset']} · {detail['language']} · "
        f"{detail['chunk_count']} fragmentos · "
        f"{detail['total_chars']:,} caracteres · {estado}"
    )
    st.divider()

    total_chunks = detail.get("chunk_count", 0)
    chunks = detail.get("chunks", [])
    for chunk in chunks:
        chunk_idx = chunk.get("chunk_index", 0)
        text_len = chunk.get("text_length", 0)

        label = f"Fragmento {chunk_idx + 1} de {total_chunks} — {text_len:,} caracteres"

        with st.container(border=True):
            with st.expander(label):
                if chunk.get("subject"):
                    st.caption(f"Tema: {chunk['subject']}")
                if chunk.get("topic"):
                    st.caption(f"Tópico: {chunk['topic']}")
                if chunk.get("filename"):
                    st.caption(f"Archivo: {chunk['filename']}")

                st.divider()
                _render_chunk_text(chunk["text"])
                st.divider()

                extra = chunk.get("metadata", {})
                if extra:
                    st.caption("Metadatos técnicos")
                    for k, v in extra.items():
                        st.caption(f"• {k}: {_fmt_bool(v)}")

    total_pages = max(1, math.ceil(total_chunks / DETAIL_PER_PAGE))
    if total_pages > 1:
        st.divider()
        col_prev, col_cur, col_next = st.columns(3)
        with col_prev:
            if st.button("◀ Anterior", disabled=page <= 1, key="detail_prev", use_container_width=True):
                st.session_state.doc_detail_page = page - 1
                st.rerun()
        with col_cur:
            st.caption(f"Página {page} de {total_pages}")
        with col_next:
            if st.button("Siguiente ▶", disabled=page >= total_pages, key="detail_next", use_container_width=True):
                st.session_state.doc_detail_page = page + 1
                st.rerun()


def render_documents() -> None:
    if not st.session_state.get("is_admin"):
        st.error("Acceso restringido a administradores.")
        st.stop()

    st.title("📁 Documentos")
    st.caption("Gestiona la base documental utilizada por el asistente RAG.")
    st.divider()

    _load_document_data()

    stats = st.session_state.get("doc_stats", {})
    api_ok = st.session_state.get("api_ok", False)

    # ------------------------------------------------------------------
    # KPIs
    # ------------------------------------------------------------------
    kpi_documents = str(stats.get("doc_count", "—"))
    kpi_fragments = f"{stats.get('total_vectors', 0):,}"
    total_chars = stats.get("total_chars", 0)
    size_mb = total_chars / 1_048_576 if total_chars else 0
    kpi_size = f"≈ {size_mb:.0f} MB de texto" if size_mb > 0 else "—"
    kpi_status = "✅ Índice actualizado" if stats.get("faiss_loaded") else "❌ No disponible"

    cols = st.columns(4)

    with cols[0]:
        _render_stat_card("📄 Documentos", kpi_documents)

    with cols[1]:
        _render_stat_card("📚 Fragmentos", kpi_fragments)

    with cols[2]:
        _render_stat_card("💾 Texto indexado", kpi_size)

    with cols[3]:
        _render_stat_card("✅ Estado", kpi_status)

    st.write("")

    # ------------------------------------------------------------------
    # Carga de documentos
    # ------------------------------------------------------------------
    with st.expander("📤 Cargar nuevo documento", expanded=False):
        st.caption("Solo cargar documentos cuya licencia permita su uso e indexación: "
                    "pasan a ser recuperables por el chat como cualquier otro fragmento.")
        uploaded_file = st.file_uploader(
            "Seleccionar archivo .txt",
            type=["txt"],
            key="doc_uploader",
        )
        if uploaded_file is not None:
            if st.button("Subir e indexar", use_container_width=True):
                with st.spinner("Procesando documento..."):
                    try:
                        resp = upload_document(
                            st.session_state.admin_token, uploaded_file.name, uploaded_file.read()
                        )
                    except httpx.HTTPStatusError as exc:
                        st.error(_upload_error_message(exc))
                    except httpx.HTTPError:
                        st.error("No se pudo conectar con la API. Verifica que el servidor esté disponible.")
                    else:
                        st.success(
                            f"Documento indexado: **{resp.get('doc_id')}** — "
                            f"{resp.get('chunks_created')} fragmentos creados."
                        )
                        st.session_state.doc_loaded = False
                        st.rerun()

    st.write("")

    # ------------------------------------------------------------------
    # Barra de herramientas: búsqueda + actualizar
    # ------------------------------------------------------------------
    col_search, col_btn = st.columns([4, 1])

    with col_search:
        st.text_input(
            "Buscar",
            placeholder="Buscar documento por nombre o fuente...",
            label_visibility="collapsed",
            key="doc_search",
            on_change=_on_search_change,
        )

    with col_btn:
        if st.button("🔄 Actualizar", use_container_width=True):
            _fetch_documents(
                search=st.session_state.get("doc_search", ""),
                page=st.session_state.get("doc_page", 1),
            )

    st.write("")

    # ------------------------------------------------------------------
    # Contenido
    # ------------------------------------------------------------------
    if not api_ok:
        with st.container(border=True):
            st.subheader("Servicio no disponible")
            st.write("La API del backend no está disponible.")
            st.write("Inicia el servidor y vuelve a intentarlo.")
    elif st.session_state.get("doc_load_error"):
        with st.container(border=True):
            st.subheader("No se pudo cargar la base documental")
            st.write("Puede ser un problema transitorio de red o del backend.")
            if st.button("Reintentar", use_container_width=True):
                st.session_state.doc_loaded = False
                st.rerun()
    elif stats.get("doc_count", 0) == 0:
        with st.container(border=True):
            st.subheader("No hay documentos indexados")
            st.write("Aún no se ha indexado ningún documento en la base de conocimiento.")
            st.write("Ejecuta el pipeline de ingestión para comenzar.")
    else:
        docs = st.session_state.get("documents", [])

        if not docs:
            with st.container(border=True):
                st.subheader("No se encontraron documentos")
                st.write("Prueba otro término de búsqueda.")
        else:
            for doc in docs:
                _render_document_card(doc)

            st.write("")

            doc_page = st.session_state.get("doc_page", 1)
            doc_total = st.session_state.get("doc_total", 0)
            col_prev, col_cur, col_next = st.columns(3)

            with col_prev:
                if st.button("◀ Anterior", disabled=doc_page <= 1, use_container_width=True):
                    _fetch_documents(
                        search=st.session_state.get("doc_search", ""),
                        page=doc_page - 1,
                    )

            with col_cur:
                total_pages = max(1, math.ceil(doc_total / PER_PAGE))
                st.caption(f"Página {doc_page} de {total_pages}")

            with col_next:
                if st.button("Siguiente ▶", disabled=doc_page * PER_PAGE >= doc_total, use_container_width=True):
                    _fetch_documents(
                        search=st.session_state.get("doc_search", ""),
                        page=doc_page + 1,
                    )
