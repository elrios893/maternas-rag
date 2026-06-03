"""
ingest_medmcqa.py — Ingesta el dataset MedMCQA completo.

Procesa:
  - train.json  (~193,155 registros)
  - dev.json    (~4,183 registros)
  - test.json   (~6,150 registros) — sin respuesta correcta, se omite
    (test.json no tiene campo 'cop', el formatter lo maneja devolviendo [ANSWER] vacío)

Formato: exp-first → [EXPLANATION] + [QUESTION] + [ANSWER] + [SUBJECT] + [TOPIC]
Chunking: passthrough (1 registro = 1 vector)
Checkpoint: cada 5,000 docs procesados.

Uso:
    python src/ingestion/ingest_medmcqa.py
    python src/ingestion/ingest_medmcqa.py --skip-dev
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.formatters import format_medmcqa
from src.ingestion.chunkers import chunk_document
from src.ingestion.store import FAISSStore
from src.settings import settings

BATCH_SIZE       = 256
CHECKPOINT_EVERY = 5000


# ---------------------------------------------------------------------------
# Helpers (mismo patrón que ingest_multiclinsum.py)
# ---------------------------------------------------------------------------

def _load_store() -> FAISSStore:
    idx = settings.faiss_store_path / "index.faiss"
    if idx.exists():
        print("[MedMCQA] Índice existente detectado — cargando para continuar.")
        return FAISSStore.load()
    print("[MedMCQA] Creando índice nuevo.")
    return FAISSStore.create_empty()


def _already_ingested(phase: str) -> bool:
    p = settings.faiss_store_path / "build_info.json"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        info = json.load(f)
    return info.get(f"medmcqa_{phase}_done", False)


def _mark_done(store: FAISSStore, phase: str) -> None:
    p = settings.faiss_store_path / "build_info.json"
    info = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            info = json.load(f)
    info[f"medmcqa_{phase}_done"] = True
    info["total_vectors"] = store.total
    settings.faiss_store_path.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    store.save()
    print(f"[MedMCQA] Fase '{phase}' completada. Total vectores: {store.total:,}")


def _checkpoint(store: FAISSStore, phase: str, processed: int) -> None:
    p = settings.faiss_store_path / "build_info.json"
    info = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            info = json.load(f)
    info[f"medmcqa_{phase}_checkpoint"] = processed
    info["total_vectors"] = store.total
    settings.faiss_store_path.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    store.save()
    print(f"[Checkpoint] medmcqa/{phase}: {processed:,} docs | Total índice: {store.total:,}")


# ---------------------------------------------------------------------------
# Ingestión de un archivo JSON (un registro por línea)
# ---------------------------------------------------------------------------

def _ingest_file(store: FAISSStore, fpath: Path, phase: str) -> None:
    if _already_ingested(phase):
        print(f"[MedMCQA] {phase} ya ingestado — saltando.")
        return

    print(f"[MedMCQA] Procesando {fpath.name} ...")

    # Contar líneas para tqdm
    with open(fpath, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"[MedMCQA] Registros en {fpath.name}: {total_lines:,}")

    batch_docs = []
    processed  = 0
    skipped    = 0

    with open(fpath, encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc=phase, unit="reg"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            doc = format_medmcqa(record)
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
                    _checkpoint(store, phase, processed)

    if batch_docs:
        store.add_documents(batch_docs, batch_size=BATCH_SIZE)
        processed += len(batch_docs)

    if skipped:
        print(f"[MedMCQA] {skipped} registros omitidos en {phase}.")

    _mark_done(store, phase)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(skip_dev: bool = False) -> None:
    print("=" * 60)
    print("INGESTIÓN — MedMCQA")
    print("=" * 60)

    data_dir = Path(settings.dataset_medmcqa_path)
    store    = _load_store()

    _ingest_file(store, data_dir / "train.json", "train")
    if not skip_dev:
        _ingest_file(store, data_dir / "dev.json",   "dev")

    print()
    print(f"[MedMCQA] COMPLETO. Total vectores en índice: {store.total:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestión de MedMCQA")
    parser.add_argument("--skip-dev", action="store_true",
                        help="Omitir dev.json (ya incluido en train generalmente)")
    args = parser.parse_args()
    main(skip_dev=args.skip_dev)
