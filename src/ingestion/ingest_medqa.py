"""
ingest_medqa.py — Ingesta MedQA (US + Taiwan + Mainland) y textbooks EN.

Procesa:
  - questions/US/       train+dev+test.jsonl  → format_medqa_us
  - questions/Taiwan/   train+dev+test.jsonl  → format_medqa_taiwan
  - questions/Mainland/ train+dev+test.jsonl  → format_medqa_mainland
  - textbooks/en/       18 archivos .txt      → format_textbook + recursive split

Chunking:
  - Questions: passthrough (1 Q = 1 vector)
  - Textbooks: recursive_split 400 tokens / 80 overlap

Checkpoint cada 5,000 docs por fase.

Uso:
    python src/ingestion/ingest_medqa.py
    python src/ingestion/ingest_medqa.py --only-textbooks
    python src/ingestion/ingest_medqa.py --skip-textbooks
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.formatters import (
    format_medqa_us,
    format_medqa_taiwan,
    format_medqa_mainland,
    format_textbook,
)
from src.ingestion.chunkers import chunk_document
from src.ingestion.store import FAISSStore
from src.settings import settings

BATCH_SIZE       = 256
CHECKPOINT_EVERY = 5000

QUESTIONS_BASE = Path(settings.dataset_medqa_path) / "questions"
TEXTBOOKS_EN   = Path(settings.dataset_medqa_path) / "textbooks" / "en"

# Splits a procesar por variante (excluimos 4_options y metamap — son subsets)
VARIANTS = {
    "us": {
        "formatter": format_medqa_us,
        "files": [
            QUESTIONS_BASE / "US" / "train.jsonl",
            QUESTIONS_BASE / "US" / "dev.jsonl",
            QUESTIONS_BASE / "US" / "test.jsonl",
        ],
    },
    "taiwan": {
        "formatter": format_medqa_taiwan,
        "files": [
            QUESTIONS_BASE / "Taiwan" / "train.jsonl",
            QUESTIONS_BASE / "Taiwan" / "dev.jsonl",
            QUESTIONS_BASE / "Taiwan" / "test.jsonl",
        ],
    },
    "mainland": {
        "formatter": format_medqa_mainland,
        "files": [
            QUESTIONS_BASE / "Mainland" / "train.jsonl",
            QUESTIONS_BASE / "Mainland" / "dev.jsonl",
            QUESTIONS_BASE / "Mainland" / "test.jsonl",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_store() -> FAISSStore:
    idx = settings.faiss_store_path / "index.faiss"
    if idx.exists():
        print("[MedQA] Índice existente detectado — cargando para continuar.")
        return FAISSStore.load()
    print("[MedQA] Creando índice nuevo.")
    return FAISSStore.create_empty()


def _already_ingested(phase: str) -> bool:
    p = settings.faiss_store_path / "build_info.json"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        info = json.load(f)
    return info.get(f"medqa_{phase}_done", False)


def _mark_done(store: FAISSStore, phase: str) -> None:
    p = settings.faiss_store_path / "build_info.json"
    info = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            info = json.load(f)
    info[f"medqa_{phase}_done"] = True
    info["total_vectors"] = store.total
    settings.faiss_store_path.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    store.save()
    print(f"[MedQA] Fase '{phase}' completada. Total vectores: {store.total:,}")


def _checkpoint(store: FAISSStore, phase: str, processed: int) -> None:
    p = settings.faiss_store_path / "build_info.json"
    info = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            info = json.load(f)
    info[f"medqa_{phase}_checkpoint"] = processed
    info["total_vectors"] = store.total
    settings.faiss_store_path.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    store.save()
    print(f"[Checkpoint] medqa/{phase}: {processed:,} docs | Total índice: {store.total:,}")


# ---------------------------------------------------------------------------
# Ingestión de preguntas (jsonl)
# ---------------------------------------------------------------------------

def _ingest_variant(store: FAISSStore, variant: str) -> None:
    if _already_ingested(variant):
        print(f"[MedQA] {variant} ya ingestado — saltando.")
        return

    cfg       = VARIANTS[variant]
    formatter = cfg["formatter"]
    files     = [f for f in cfg["files"] if f.exists()]

    if not files:
        print(f"[MedQA] No se encontraron archivos para {variant} — saltando.")
        _mark_done(store, variant)
        return

    batch_docs = []
    processed  = 0
    skipped    = 0

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        with open(fpath, encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines,
                             desc=f"medqa/{variant}/{fpath.name}", unit="reg"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                doc = formatter(record)
                if not doc.text.strip():
                    skipped += 1
                    continue

                chunks = chunk_document(doc)
                batch_docs.extend(chunks)

                if len(batch_docs) >= BATCH_SIZE:
                    store.add_documents(batch_docs, batch_size=BATCH_SIZE)
                    processed += len(batch_docs)
                    batch_docs = []
                    if processed % CHECKPOINT_EVERY == 0:
                        _checkpoint(store, variant, processed)

    if batch_docs:
        store.add_documents(batch_docs, batch_size=BATCH_SIZE)
        processed += len(batch_docs)

    if skipped:
        print(f"[MedQA] {skipped} registros omitidos en {variant}.")

    _mark_done(store, variant)


# ---------------------------------------------------------------------------
# Ingestión de textbooks EN
# ---------------------------------------------------------------------------

def _ingest_textbooks(store: FAISSStore) -> None:
    if _already_ingested("textbooks_en"):
        print("[MedQA] Textbooks EN ya ingestados — saltando.")
        return

    files = sorted(TEXTBOOKS_EN.glob("*.txt"))
    if not files:
        print(f"[MedQA] No se encontraron textbooks en {TEXTBOOKS_EN}")
        _mark_done(store, "textbooks_en")
        return

    print(f"[MedQA] Textbooks EN encontrados: {len(files)}")

    batch_docs = []
    processed  = 0

    for fpath in tqdm(files, desc="textbooks/en", unit="libro"):
        try:
            with open(fpath, encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(fpath, encoding="latin-1") as f:
                text = f.read()

        if not text.strip():
            continue

        doc    = format_textbook(fpath.name, text)
        # chunkers.py usa recursive_split para source_dataset=="textbook"
        chunks = chunk_document(doc)

        batch_docs.extend(chunks)

        if len(batch_docs) >= BATCH_SIZE:
            store.add_documents(batch_docs, batch_size=BATCH_SIZE)
            processed += len(batch_docs)
            batch_docs = []
            if processed % CHECKPOINT_EVERY == 0:
                _checkpoint(store, "textbooks_en", processed)

    if batch_docs:
        store.add_documents(batch_docs, batch_size=BATCH_SIZE)
        processed += len(batch_docs)

    _mark_done(store, "textbooks_en")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(only_textbooks: bool = False, skip_textbooks: bool = False) -> None:
    print("=" * 60)
    print("INGESTIÓN — MedQA")
    print("=" * 60)

    store = _load_store()

    if not only_textbooks:
        for variant in ("us", "taiwan", "mainland"):
            _ingest_variant(store, variant)

    if not skip_textbooks:
        _ingest_textbooks(store)

    print()
    print(f"[MedQA] COMPLETO. Total vectores en índice: {store.total:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestión de MedQA")
    parser.add_argument("--only-textbooks",  action="store_true")
    parser.add_argument("--skip-textbooks",  action="store_true")
    args = parser.parse_args()
    main(only_textbooks=args.only_textbooks, skip_textbooks=args.skip_textbooks)
