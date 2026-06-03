"""
run_ingestion.py — Orquestador de ingestión completa.

Ejecuta en orden:
  1. Multiclinsum (summaries + fulltexts)
  2. MedMCQA     (train + dev)
  3. MedQA       (US + Taiwan + Mainland + textbooks EN)

Cada script es idempotente: si una fase ya fue completada la salta.
Si el proceso se interrumpe, al relanzar continúa desde el último checkpoint.

Uso:
    python src/ingestion/run_ingestion.py
    python src/ingestion/run_ingestion.py --only multiclinsum
    python src/ingestion/run_ingestion.py --only medmcqa
    python src/ingestion/run_ingestion.py --only medqa
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(only: str | None = None) -> None:
    run_all       = only is None
    run_multiclin = run_all or only == "multiclinsum"
    run_medmcqa   = run_all or only == "medmcqa"
    run_medqa     = run_all or only == "medqa"

    if run_multiclin:
        from src.ingestion.ingest_multiclinsum import main as run_multiclinsum
        run_multiclinsum()

    if run_medmcqa:
        from src.ingestion.ingest_medmcqa import main as run_mmc
        run_mmc()

    if run_medqa:
        from src.ingestion.ingest_medqa import main as run_mq
        run_mq()

    print()
    print("=" * 60)
    print("INGESTIÓN COMPLETA")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestador de ingestión")
    parser.add_argument(
        "--only",
        choices=["multiclinsum", "medmcqa", "medqa"],
        default=None,
        help="Ejecutar solo un dataset específico",
    )
    args = parser.parse_args()
    main(only=args.only)
