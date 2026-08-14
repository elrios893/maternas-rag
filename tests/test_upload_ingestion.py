"""
test_upload_ingestion.py — format_upload / safe_doc_id / dispatch de
chunking para documentos cargados desde el panel de administración.
"""

from __future__ import annotations

from src.ingestion.chunkers import chunk_document, chunk_recursive_split
from src.ingestion.formatters import Document, format_upload, safe_doc_id


class TestSafeDocId:
    def test_quita_extension(self):
        assert safe_doc_id("informe.txt") == "informe"

    def test_quita_componentes_de_ruta(self):
        assert safe_doc_id("/etc/passwd") == "passwd"
        assert safe_doc_id("..\\..\\secreto.txt") == "secreto"

    def test_reemplaza_caracteres_inseguros(self):
        assert safe_doc_id("informe final (v2)!!.txt") == "informe_final_v2"

    def test_conserva_puntos_guiones_y_guion_bajo(self):
        assert safe_doc_id("guia-clinica_2026.v1.txt") == "guia-clinica_2026.v1"


class TestFormatUpload:
    def test_metadata_shape(self):
        doc = format_upload("guia.txt", "Contenido del documento.")
        assert doc.metadata["source_dataset"] == "upload"
        assert doc.metadata["doc_id"] == "guia"
        assert doc.metadata["language"] == "es"
        assert doc.metadata["filename"] == "guia.txt"
        assert doc.metadata["uploaded_at"].endswith("Z")

    def test_limpia_el_texto(self):
        doc = format_upload("x.txt", "  con espacios al borde  \n")
        assert doc.text == "con espacios al borde"

    def test_nombre_con_ruta_se_reduce_al_archivo(self):
        doc = format_upload("carpeta/subcarpeta/archivo raro!.txt", "contenido")
        assert doc.metadata["filename"] == "archivo raro!.txt"
        assert doc.metadata["doc_id"] == "archivo_raro"


class TestChunkDispatch:
    def test_upload_despacha_a_recursive_split(self):
        text = ("Parrafo de contenido variado para forzar el chunking del texto subido. " * 60)
        doc = Document(text=text, metadata={"source_dataset": "upload", "doc_id": "d1"})
        chunks = chunk_document(doc)
        assert len(chunks) > 1
        assert all(c.metadata["is_chunk"] for c in chunks)
        assert all(c.metadata["source_dataset"] == "upload" for c in chunks)

    def test_upload_corto_no_pierde_marcadores_de_chunk(self):
        """Caso borde: si el texto es tan corto que ningún split del
        recursive splitter supera MIN_PARAGRAPH_CHARS, antes se devolvía
        el Document original SIN chunk_index/is_chunk. Ahora debe pasar
        por chunk_passthrough y traerlos igual."""
        doc = Document(text="corto", metadata={"source_dataset": "upload", "doc_id": "d2"})
        chunks = chunk_recursive_split(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["is_chunk"] is False
