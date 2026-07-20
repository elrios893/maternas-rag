"""
ingest_maternaqaes_lm.py — Ingesta el corpus LM de MaternaQA-es al indice FAISS.

Descarga los 3 splits del corpus LM directamente desde GitHub raw:
  - train_lm.jsonl    (1.744 chunks, 52 PDFs, split train)
  - validation_lm.jsonl (101 chunks, 2 PDFs, split validation)
  - test_lm.jsonl     (108 chunks, 3 PDFs, split test — mismos del benchmark QA)

Total: 1.953 chunks de obstetricia en espanol, promedio 879 tokens/chunk.

Por que JSONL y no PDFs crudos:
  - Ya procesados, limpios y auditados por el equipo MaternaQA-es
  - Metadatos ricos: clinical_score, topics, section_type, content_role
  - Sin necesidad de OCR ni chunking manual
  - Ver foragents/qa_technical.md Q27 para decision completa

Los chunks se agregan con source_dataset="maternaqaes_lm" al indice FAISS existente
(incremental — no reconstruye los 375k vectores actuales).

Uso:
    # Ingestar los 3 splits (para Config C — upper bound con corpus completo)
    python -m src.ingestion.ingest_maternaqaes_lm

    # Solo train+val (para Config D — sin contaminacion del test set)
    python -m src.ingestion.ingest_maternaqaes_lm --no-test

    # Solo verificar cuantos chunks tendria, sin ingestar
    python -m src.ingestion.ingest_maternaqaes_lm --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.formatters import Document
from src.ingestion.store import FAISSStore
from src.settings import settings

# ---------------------------------------------------------------------------
# URLs raw de GitHub
# ---------------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/minciencias-maternas/MaternaQA-es/master/datasets/obstetrics/lm"

SPLITS = {
    "train":      f"{BASE_URL}/train_lm.jsonl",
    "validation": f"{BASE_URL}/validation_lm.jsonl",
    "test":       f"{BASE_URL}/test_lm.jsonl",
}

DATASET_ID = "maternaqaes_lm"
BATCH_SIZE  = 128


# ---------------------------------------------------------------------------
# Descarga y parseo
# ---------------------------------------------------------------------------

def _download_jsonl(url: str, split_name: str) -> list[dict]:
    """Descarga un JSONL de GitHub raw y lo parsea linea a linea."""
    print(f"  Descargando {split_name}: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            content = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error descargando {url}: {e}")

    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"    [WARN] Linea JSON invalida, omitida: {e}")
    print(f"  {split_name}: {len(records)} chunks descargados")
    return records


def _to_document(record: dict) -> Document | None:
    """Convierte un registro del LM dataset al formato Document del proyecto."""
    text = record.get("text", "").strip()
    if not text or len(text) < 50:
        return None

    meta = record.get("metadata", {})

    return Document(
        text=text,
        metadata={
            "source_dataset": DATASET_ID,
            "language":       "es",
            "doc_id":         meta.get("pdf_id", "unknown"),
            "chunk_id":       meta.get("chunk_id", ""),
            "source_pdf":     meta.get("source_pdf", ""),
            "section_type":   meta.get("section_type", ""),
            "content_role":   meta.get("content_role", ""),
            "topics":         meta.get("topics", []),
            "clinical_score": meta.get("clinical_score", 0),
            "token_estimate": meta.get("token_estimate", 0),
            "lm_split":       meta.get("split", ""),
            "pages":          meta.get("pages", []),
        },
    )


# ---------------------------------------------------------------------------
# Verificacion de chunk_ids ya ingestados (evitar duplicados)
# ---------------------------------------------------------------------------

def _get_existing_chunk_ids(store: FAISSStore) -> set[str]:
    """Extrae todos los chunk_ids del dataset maternaqaes_lm ya en el indice."""
    existing = set()
    for meta in store.metadata.values():
        if meta.get("source_dataset") == DATASET_ID:
            cid = meta.get("chunk_id", "")
            if cid:
                existing.add(cid)
    return existing


# ---------------------------------------------------------------------------
# Ingestion principal
# ---------------------------------------------------------------------------

def ingest(include_test: bool = True, dry_run: bool = False) -> None:
    sep = "=" * 62
    print(sep)
    print("  Ingestando MaternaQA-es LM corpus al indice FAISS")
    print(f"  Splits: train + validation" + (" + test (LEAKAGE WARNING)" if include_test else " (sin test — correcto para evaluacion)"))
    if dry_run:
        print("  MODO DRY-RUN: no se modificara el indice")
    print(sep)

    # 1. Descargar JSONL
    all_records: list[dict] = []
    # Por defecto descarga solo train+validation (sin leakage)
    splits_to_use = ["train", "validation"] + (["test"] if include_test else [])

    for split in splits_to_use:
        records = _download_jsonl(SPLITS[split], split)
        all_records.extend(records)

    print(f"\n  Total registros descargados: {len(all_records)}")

    # 2. Convertir a Documents
    documents: list[Document] = []
    skipped = 0
    for rec in all_records:
        doc = _to_document(rec)
        if doc:
            documents.append(doc)
        else:
            skipped += 1

    print(f"  Documents validos: {len(documents)} | Omitidos (muy cortos): {skipped}")

    if dry_run:
        print(f"\n  [DRY-RUN] Se ingrestarian {len(documents)} chunks.")
        print(f"  Tokens estimados totales: {sum(d.metadata.get('token_estimate',0) for d in documents):,}")
        _print_breakdown(documents)
        return

    # 3. Cargar indice existente
    store = FAISSStore.load()
    total_antes = store.total
    print(f"\n  Indice actual: {total_antes:,} vectores")

    # 4. Verificar duplicados
    existing_ids = _get_existing_chunk_ids(store)
    if existing_ids:
        print(f"  Chunks ya ingestados de {DATASET_ID}: {len(existing_ids)}")
        documents = [d for d in documents if d.metadata.get("chunk_id") not in existing_ids]
        print(f"  Chunks nuevos a ingestar: {len(documents)}")

    if not documents:
        print("  Nada que ingestar — todos los chunks ya estan en el indice.")
        return

    # 5. Ingestar en lotes
    print(f"\n  Ingestando {len(documents)} chunks en lotes de {BATCH_SIZE}...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        store.add_documents(batch, batch_size=BATCH_SIZE)

    # 6. Guardar
    store.save()
    total_despues = store.total
    added = total_despues - total_antes

    print(f"\n  Vectores antes : {total_antes:,}")
    print(f"  Vectores despues: {total_despues:,}")
    print(f"  Agregados       : {added:,}")
    _print_breakdown(documents)
    print(f"\n  Indice guardado en: {settings.faiss_store_path}")
    print(sep)


def _print_breakdown(documents: list[Document]) -> None:
    """Muestra distribucion por split y por source_pdf."""
    from collections import Counter

    splits = Counter(d.metadata.get("lm_split", "?") for d in documents)
    pdfs   = Counter(d.metadata.get("source_pdf", "?") for d in documents)

    print("\n  Por split:")
    for split, n in sorted(splits.items()):
        print(f"    {split:<12} {n:>5} chunks")

    print(f"\n  PDFs fuente ({len(pdfs)} unicos) — top 10 por chunks:")
    for pdf, n in pdfs.most_common(10):
        print(f"    {n:>4}  {pdf}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingesta MaternaQA-es LM corpus al indice FAISS",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Por defecto NO ingestar el split test — evita data leakage en evaluacion.
    # El split test contiene los chunks exactos de los 3 PDFs que generan los
    # 328 pares QA del benchmark MaternaQA-es. Si se ingestan, context_recall
    # y context_precision suben artificialmente (el sistema "ya vio" esos docs).
    # Usar --include-test solo para medir el upper bound teorico.
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Incluir split test (CUIDADO: data leakage en evaluacion con MaternaQA-es)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar estadisticas, sin modificar el indice",
    )
    args = parser.parse_args()

    ingest(include_test=args.include_test, dry_run=args.dry_run)
