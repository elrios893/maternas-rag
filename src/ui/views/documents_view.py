"""
documents_view.py — Vista del Centro de Gestión Documental del RAG.
"""

import streamlit as st
import httpx

from src.ui.client import (
    get_document_detail,
    get_document_stats,
    list_documents,
    toggle_document_status,
)


def _render_stat_card(title: str, value: str) -> None:
    with st.container(border=True):
        st.caption(title)
        st.write(value)


def _load_document_data() -> None:
    if st.session_state.get("doc_loaded"):
        return
    st.session_state.doc_loaded = True
    try:
        with st.spinner("Cargando base documental..."):
            stats = get_document_stats()
            resp = list_documents(search="", page=1, per_page=20)
        st.session_state.doc_stats = stats
        st.session_state.documents = resp.get("documents", [])
        st.session_state.doc_total = resp.get("total", 0)
        st.session_state.doc_page = 1
        st.session_state.doc_search_active = ""
    except httpx.HTTPError:
        st.session_state.doc_stats = {}
        st.session_state.documents = []
        st.session_state.doc_total = 0
        st.session_state.doc_page = 1


def _fetch_documents(search: str = "", page: int = 1) -> None:
    with st.spinner("Buscando documentos..."):
        try:
            resp = list_documents(search=search, page=page, per_page=20)
            st.session_state.documents = resp.get("documents", [])
            st.session_state.doc_total = resp.get("total", 0)
            st.session_state.doc_page = page
            st.session_state.doc_search_active = search
        except httpx.HTTPError:
            st.session_state.documents = []
            st.session_state.doc_total = 0
            st.session_state.doc_page = 1


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
        toggle_document_status(doc_id, active=active)
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
    with st.spinner("Cargando detalle..."):
        try:
            detail = get_document_detail(doc_id)
        except httpx.HTTPError:
            st.error("No se pudo cargar el detalle del documento.")
            st.stop()

    st.write(f"**{detail['doc_id']}**")
    st.caption(
        f"{detail['source_dataset']} · {detail['language']} · "
        f"{detail['chunk_count']} fragmentos · "
        f"{detail['total_chars']:,} caracteres"
    )
    st.divider()

    chunks = detail.get("chunks", [])
    total_chunks = len(chunks)
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
                extra = {k: v for k, v in extra.items() if k != "doc_id"}
                if extra:
                    st.caption("Metadatos técnicos")
                    for k, v in extra.items():
                        st.caption(f"• {k}: {_fmt_bool(v)}")


def render_documents() -> None:
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
    kpi_size = f"{size_mb:.0f} MB" if size_mb > 0 else "—"
    kpi_status = "✅ Índice actualizado" if stats.get("faiss_loaded") else "❌ No disponible"

    cols = st.columns(4)

    with cols[0]:
        _render_stat_card("📄 Documentos", kpi_documents)

    with cols[1]:
        _render_stat_card("📚 Fragmentos", kpi_fragments)

    with cols[2]:
        _render_stat_card("💾 Tamaño total", kpi_size)

    with cols[3]:
        _render_stat_card("✅ Estado", kpi_status)

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
                st.caption(f"Página {doc_page}")

            with col_next:
                if st.button("Siguiente ▶", disabled=doc_page * 20 >= doc_total, use_container_width=True):
                    _fetch_documents(
                        search=st.session_state.get("doc_search", ""),
                        page=doc_page + 1,
                    )
