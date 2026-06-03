"""
ingest_multiclinsum.py — Ingesta el dataset Multiclinsum completo.

Procesa:
  - 25,902 summaries  → passthrough (1 doc = 1 vector)
  - 25,902 fulltexts  → paragraph grouping chunking

Checkpoint: guarda el índice FAISS en disco al finalizar cada fase
(summaries primero, fulltexts después) para poder retomar si falla.

Uso:
    python -m src.ingestion.ingest_multiclinsum
    python -m src.ingestion.ingest_multiclinsum --only-summaries
    python -m src.ingestion.ingest_multiclinsum --only-fulltexts
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.formatters import (
    format_multiclinsum_summary,
    format_multiclinsum_fulltext,
)
from src.ingestion.chunkers import chunk_document
from src.ingestion.store import FAISSStore
from src.settings import settings

# Tamaño de lote para add_documents
BATCH_SIZE = 256

# Guardar checkpoint en disco cada N documentos procesados
CHECKPOINT_EVERY = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_store() -> FAISSStore:
    """Carga el índice existente o crea uno vacío."""
    idx = settings.faiss_store_path / "index.faiss"
    if idx.exists():
        print("[Multiclinsum] Índice existente detectado — cargando para continuar.")
        return FAISSStore.load()
    print("[Multiclinsum] Creando índice nuevo.")
    return FAISSStore.create_empty()


def _already_ingested(phase: str) -> bool:
    """Comprueba si una fase ya fue completada en una ejecución anterior."""
    p = settings.faiss_store_path / "build_info.json"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        info = json.load(f)
    return info.get(f"multiclinsum_{phase}_done", False)


def _mark_done(store: FAISSStore, phase: str) -> None:
    """Marca una fase como completada en build_info.json y guarda el índice."""
    p = settings.faiss_store_path / "build_info.json"
    info = store.build_info() if p.exists() else {}
    info[f"multiclinsum_{phase}_done"] = True
    info["embedding_model"]    = settings.embedding_model
    info["faiss_index_type"]   = "IndexFlatIP"
    info["dimension"]          = store.index.d
    info["total_vectors"]      = store.total

    settings.faiss_store_path.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    store.save()
    print(f"[Multiclinsum] Fase '{phase}' completada y guardada. Total vectores: {store.total:,}")


def _checkpoint(store: FAISSStore, phase: str, processed: int) -> None:
    """Guarda checkpoint intermedio sin marcar la fase como completada."""
    p = settings.faiss_store_path / "build_info.json"
    info = store.build_info() if p.exists() else {}
    info[f"multiclinsum_{phase}_checkpoint"] = processed
    info["total_vectors"] = store.total

    settings.faiss_store_path.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    store.save()
    print(f"[Checkpoint] {phase}: {processed:,} docs procesados | Total índice: {store.total:,}")


# ---------------------------------------------------------------------------
# Fase 1: Summaries
# ---------------------------------------------------------------------------

def ingest_summaries(store: FAISSStore) -> None:
    if _already_ingested("summaries"):
        print("[Multiclinsum] Summaries ya ingestados — saltando.")
        return

    sm_dir = settings.dataset_multiclinsum_path / "summaries"
    files  = sorted(sm_dir.glob("*_sum.txt"))
    print(f"[Multiclinsum] Summaries encontrados: {len(files):,}")

    batch_docs = []
    processed  = 0

    for fpath in tqdm(files, desc="Summaries", unit="archivo"):
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            continue

        doc    = format_multiclinsum_summary(fpath.name, text)
        chunks = chunk_document(doc)
        batch_docs.extend(chunks)

        if len(batch_docs) >= BATCH_SIZE:
            store.add_documents(batch_docs, batch_size=BATCH_SIZE)
            processed += len(batch_docs)
            batch_docs = []

            if processed % CHECKPOINT_EVERY == 0:
                _checkpoint(store, "summaries", processed)

    if batch_docs:
        store.add_documents(batch_docs, batch_size=BATCH_SIZE)

    _mark_done(store, "summaries")


# ---------------------------------------------------------------------------
# Fase 2: Fulltexts
# ---------------------------------------------------------------------------

def ingest_fulltexts(store: FAISSStore) -> None:
    if _already_ingested("fulltexts"):
        print("[Multiclinsum] Fulltexts ya ingestados — saltando.")
        return

    ft_dir = settings.dataset_multiclinsum_path / "fulltext"
    files  = sorted(ft_dir.glob("*.txt"))
    print(f"[Multiclinsum] Fulltexts encontrados: {len(files):,}")

    batch_docs = []
    processed  = 0

    for fpath in tqdm(files, desc="Fulltexts", unit="archivo"):
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            continue

        doc    = format_multiclinsum_fulltext(fpath.name, text)
        chunks = chunk_document(doc)
        batch_docs.extend(chunks)

        if len(batch_docs) >= BATCH_SIZE:
            store.add_documents(batch_docs, batch_size=BATCH_SIZE)
            processed += len(batch_docs)
            batch_docs = []

            if processed % CHECKPOINT_EVERY == 0:
                _checkpoint(store, "fulltexts", processed)

    if batch_docs:
        store.add_documents(batch_docs, batch_size=BATCH_SIZE)

    _mark_done(store, "fulltexts")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(only_summaries: bool = False, only_fulltexts: bool = False) -> None:
    print("=" * 60)
    print("INGESTIÓN — Multiclinsum")
    print("=" * 60)

    store = _load_store()

    if not only_fulltexts:
        ingest_summaries(store)

    if not only_summaries:
        ingest_fulltexts(store)

    print()
    print(f"[Multiclinsum] COMPLETO. Total vectores en índice: {store.total:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestión de Multiclinsum")
    parser.add_argument("--only-summaries", action="store_true")
    parser.add_argument("--only-fulltexts", action="store_true")
    args = parser.parse_args()
    main(only_summaries=args.only_summaries, only_fulltexts=args.only_fulltexts)
