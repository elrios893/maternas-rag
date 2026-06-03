"""
chunkers.py — Estrategias de chunking por tipo de fuente.

Recibe un Document (de formatters.py) y devuelve una lista de Documents.
Si el documento no necesita chunking, devuelve una lista de un solo elemento.

Estrategias implementadas:
  - passthrough        : sin chunking (MedMCQA, MedQA, summaries)
  - paragraph_grouping : agrupa párrafos hasta ~400 tokens (multiclinsum fulltexts)
  - recursive_split    : recursive character splitter (textbooks)

La decisión de qué estrategia aplicar se toma en base al campo
metadata["source_dataset"] del Document entrante.
"""

import uuid
import copy
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.formatters import Document


# ---------------------------------------------------------------------------
# Constantes — alineadas con el plan técnico
# ---------------------------------------------------------------------------

CHUNK_SIZE_TOKENS   = 400   # aprox. tokens por chunk (1 token ≈ 4 chars)
CHUNK_OVERLAP_TOKENS = 80   # overlap entre chunks de textbooks
CHARS_PER_TOKEN     = 4     # estimación conservadora

CHUNK_SIZE_CHARS    = CHUNK_SIZE_TOKENS  * CHARS_PER_TOKEN   # 1600
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN  # 320

# Para paragraph grouping (fulltexts)
PARAGRAPH_TARGET_CHARS = 350 * CHARS_PER_TOKEN  # ~1400 chars ≈ 350 tokens
MIN_PARAGRAPH_CHARS    = 50                      # ignorar párrafos vacíos


# ---------------------------------------------------------------------------
# Datasets que NO necesitan chunking
# ---------------------------------------------------------------------------

NO_CHUNK_SOURCES = {
    "medmcqa",
    "medqa_us",
    "medqa_taiwan",
    "medqa_mainland",
    "multiclinsum_summary",
}


# ---------------------------------------------------------------------------
# Helper: asignar chunk_id único copiando metadata del padre
# ---------------------------------------------------------------------------

def _make_chunk(text: str, parent_metadata: dict, chunk_index: int) -> Document:
    meta = copy.deepcopy(parent_metadata)
    meta["chunk_id"]    = str(uuid.uuid4())
    meta["chunk_index"] = chunk_index
    meta["is_chunk"]    = True
    return Document(text=text.strip(), metadata=meta)


# ---------------------------------------------------------------------------
# Estrategia 1: Passthrough — sin chunking
# ---------------------------------------------------------------------------

def chunk_passthrough(doc: Document) -> List[Document]:
    """
    Devuelve el documento tal cual, marcado como chunk único.
    Usado para MedMCQA, MedQA y multiclinsum summaries.
    """
    doc.metadata["chunk_index"] = 0
    doc.metadata["is_chunk"]    = False  # es el documento completo, no un fragmento
    return [doc]


# ---------------------------------------------------------------------------
# Estrategia 2: Paragraph grouping — para multiclinsum fulltexts
# ---------------------------------------------------------------------------
# Divide el texto por párrafos (\n\n) y los agrupa hasta alcanzar
# ~350 tokens por chunk. El overlap es de 1 párrafo completo entre
# chunks contiguos para preservar el hilo narrativo clínico.

def chunk_paragraph_grouping(doc: Document) -> List[Document]:
    """
    Agrupa párrafos naturales (\n\n) hasta ~350 tokens por chunk.
    Overlap de 1 párrafo entre chunks consecutivos.
    """
    raw_paragraphs = doc.text.split("\n\n")
    paragraphs = [p.strip() for p in raw_paragraphs if len(p.strip()) >= MIN_PARAGRAPH_CHARS]

    if not paragraphs:
        return [doc]

    chunks: List[Document] = []
    current_paragraphs: List[str] = []
    current_chars = 0
    chunk_index = 0

    for para in paragraphs:
        para_chars = len(para)

        # Si agregar este párrafo supera el target Y ya hay contenido → cerrar chunk
        if current_chars + para_chars > PARAGRAPH_TARGET_CHARS and current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(_make_chunk(chunk_text, doc.metadata, chunk_index))
            chunk_index += 1

            # Overlap: el último párrafo del chunk anterior inicia el siguiente
            current_paragraphs = [current_paragraphs[-1], para]
            current_chars = len(current_paragraphs[0]) + para_chars
        else:
            current_paragraphs.append(para)
            current_chars += para_chars

    # Último chunk con lo que quede
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunks.append(_make_chunk(chunk_text, doc.metadata, chunk_index))

    return chunks


# ---------------------------------------------------------------------------
# Estrategia 3: Recursive character split — para textbooks
# ---------------------------------------------------------------------------
# 400 tokens / 80 tokens overlap.
# Separadores en orden de prioridad: párrafo > línea > oración > espacio.

_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    separators=["\n\n", "\n", ". ", " "],
    length_function=len,
)


def chunk_recursive_split(doc: Document) -> List[Document]:
    """
    Aplica recursive character splitting al texto completo del documento.
    Usado para los 18 textbooks EN.
    """
    raw_chunks = _recursive_splitter.split_text(doc.text)

    # Filtrar chunks demasiado cortos (ruido, encabezados, etc.)
    valid_chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= MIN_PARAGRAPH_CHARS]

    if not valid_chunks:
        return [doc]

    return [
        _make_chunk(text, doc.metadata, idx)
        for idx, text in enumerate(valid_chunks)
    ]


# ---------------------------------------------------------------------------
# Dispatcher principal — punto de entrada único
# ---------------------------------------------------------------------------

def chunk_document(doc: Document) -> List[Document]:
    """
    Decide qué estrategia aplicar según el source_dataset del documento
    y devuelve la lista de chunks resultante.

    Uso:
        chunks = chunk_document(doc)
        # chunks es siempre List[Document], mínimo 1 elemento
    """
    source = doc.metadata.get("source_dataset", "")

    if source in NO_CHUNK_SOURCES:
        return chunk_passthrough(doc)

    if source == "multiclinsum_fulltext":
        return chunk_paragraph_grouping(doc)

    if source == "textbook":
        return chunk_recursive_split(doc)

    # Fallback: si llega un source desconocido, no chunking
    return chunk_passthrough(doc)
