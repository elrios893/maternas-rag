"""
formatters.py — Convierte registros crudos de cada dataset
en un Document estandarizado listo para embeddear e indexar.

Cada formatter recibe el registro original del dataset y devuelve
un Document con:
  - text     : string formateado, lo que se embeddeará
  - metadata : dict con fuente, idioma, subject, doc_id, etc.

No hay chunking aquí — ese es trabajo de chunkers.py.
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid


# ---------------------------------------------------------------------------
# Estructura de salida unificada
# ---------------------------------------------------------------------------

@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Asignar chunk_id único si no viene en metadata
        if "chunk_id" not in self.metadata:
            self.metadata["chunk_id"] = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _clean(text: Optional[str]) -> str:
    """Devuelve string limpio o vacío si es None."""
    if not text:
        return ""
    return text.strip()


def _correct_option_text(record: dict) -> str:
    """
    Devuelve el texto de la opción correcta de un registro MedMCQA.
    cop: 1=opa, 2=opb, 3=opc, 4=opd
    """
    mapping = {1: "opa", 2: "opb", 3: "opc", 4: "opd"}
    key = mapping.get(record.get("cop"))
    return _clean(record.get(key, "")) if key else ""


# ---------------------------------------------------------------------------
# Formatter 1: MedMCQA
# ---------------------------------------------------------------------------
# Formato exp-first: la explicación lidera el texto porque contiene
# el conocimiento médico denso que queremos recuperar.
# Cuando exp es null (11.8% de registros) se omite sin descartar.

def format_medmcqa(record: dict) -> Document:
    exp      = _clean(record.get("exp"))
    question = _clean(record.get("question"))
    answer   = _correct_option_text(record)
    subject  = _clean(record.get("subject_name"))
    topic    = _clean(record.get("topic_name"))

    parts = []
    if exp:
        parts.append(f"[EXPLANATION] {exp}")
    parts.append(f"[QUESTION] {question}")
    if answer:
        parts.append(f"[ANSWER] {answer}")
    if subject:
        parts.append(f"[SUBJECT] {subject}")
    if topic:
        parts.append(f"[TOPIC] {topic}")

    text = "\n".join(parts)

    metadata = {
        "source_dataset": "medmcqa",
        "doc_id":         _clean(record.get("id")),
        "subject":        subject,
        "topic":          topic,
        "language":       "en",
        "choice_type":    _clean(record.get("choice_type")),
        "has_explanation": bool(exp),
    }

    return Document(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# Formatter 2: MedQA — US (inglés, 5 opciones A-E)
# ---------------------------------------------------------------------------

def format_medqa_us(record: dict) -> Document:
    question   = _clean(record.get("question"))
    answer     = _clean(record.get("answer"))
    meta_info  = _clean(record.get("meta_info"))
    answer_idx = _clean(record.get("answer_idx"))
    options    = record.get("options", {})

    opts_text = "  ".join(
        f"{k}: {_clean(v)}" for k, v in sorted(options.items())
    )

    parts = [
        f"[QUESTION] {question}",
        f"[OPTIONS] {opts_text}",
        f"[ANSWER] {answer_idx}. {answer}",
    ]
    if meta_info:
        parts.append(f"[SOURCE] {meta_info}")

    text = "\n".join(parts)

    metadata = {
        "source_dataset": "medqa_us",
        "doc_id":         str(uuid.uuid4()),  # MedQA US no tiene ID propio
        "subject":        meta_info,
        "language":       "en",
    }

    return Document(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# Formatter 3: MedQA — Taiwan (chino tradicional, 4 opciones A-D)
# ---------------------------------------------------------------------------

def format_medqa_taiwan(record: dict) -> Document:
    question   = _clean(record.get("question"))
    answer     = _clean(record.get("answer"))
    answer_idx = _clean(record.get("answer_idx"))
    options    = record.get("options", {})

    opts_text = "  ".join(
        f"{k}: {_clean(v)}" for k, v in sorted(options.items())
    )

    parts = [
        f"[QUESTION] {question}",
        f"[OPTIONS] {opts_text}",
        f"[ANSWER] {answer_idx}. {answer}",
    ]

    text = "\n".join(parts)

    metadata = {
        "source_dataset": "medqa_taiwan",
        "doc_id":         str(uuid.uuid4()),
        "subject":        _clean(record.get("meta_info")),
        "language":       "zh-hant",
    }

    return Document(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# Formatter 4: MedQA — Mainland (chino simplificado, 5 opciones A-E)
# ---------------------------------------------------------------------------

def format_medqa_mainland(record: dict) -> Document:
    question   = _clean(record.get("question"))
    answer     = _clean(record.get("answer"))
    answer_idx = _clean(record.get("answer_idx", ""))
    meta_info  = _clean(record.get("meta_info"))
    options    = record.get("options", {})

    opts_text = "  ".join(
        f"{k}: {_clean(v)}" for k, v in sorted(options.items())
    )

    parts = [
        f"[QUESTION] {question}",
        f"[OPTIONS] {opts_text}",
        f"[ANSWER] {answer_idx}. {answer}".strip(". "),
    ]
    if meta_info:
        parts.append(f"[SOURCE] {meta_info}")

    text = "\n".join(parts)

    metadata = {
        "source_dataset": "medqa_mainland",
        "doc_id":         str(uuid.uuid4()),
        "subject":        meta_info,
        "language":       "zh-hans",
    }

    return Document(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# Formatter 5: Multiclinsum — Summary
# ---------------------------------------------------------------------------
# Los summaries son documentos primarios: texto denso y corto.
# Se indexan tal cual, sin ninguna transformación.

def format_multiclinsum_summary(filename: str, text: str) -> Document:
    # Extraer número de caso del nombre de archivo
    # Ejemplo: multiclinsum_ls_es_1234_sum.txt -> 1234
    case_id = filename.replace("multiclinsum_ls_es_", "").replace("_sum.txt", "")

    metadata = {
        "source_dataset": "multiclinsum_summary",
        "doc_id":         case_id,
        "subject":        "clinical_case",
        "language":       "es",
        "filename":       filename,
    }

    return Document(text=_clean(text), metadata=metadata)


# ---------------------------------------------------------------------------
# Formatter 6: Multiclinsum — Fulltext
# ---------------------------------------------------------------------------
# Los fulltexts se formatean aquí pero SIN chunking.
# chunkers.py se encargará de partirlos.
# Este formatter devuelve el texto limpio + metadata del caso.

def format_multiclinsum_fulltext(filename: str, text: str) -> Document:
    case_id = filename.replace("multiclinsum_ls_es_", "").replace(".txt", "")

    metadata = {
        "source_dataset": "multiclinsum_fulltext",
        "doc_id":         case_id,
        "subject":        "clinical_case",
        "language":       "es",
        "filename":       filename,
    }

    return Document(text=_clean(text), metadata=metadata)


# ---------------------------------------------------------------------------
# Formatter 7: Textbook (texto plano, cualquier libro)
# ---------------------------------------------------------------------------
# Los textbooks se formatean como texto plano con el nombre del libro
# en metadata. El chunking lo hace chunkers.py.

def format_textbook(filename: str, text: str) -> Document:
    book_name = filename.replace(".txt", "")

    metadata = {
        "source_dataset": "textbook",
        "doc_id":         book_name,
        "subject":        book_name,
        "language":       "en",
        "filename":       filename,
    }

    return Document(text=_clean(text), metadata=metadata)
