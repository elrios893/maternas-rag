"""
sampler.py — Descarga y muestrea estratificadamente el split test de MaternaQA-es.

Fuente: https://github.com/JhonHander/MaternaQA-es (raw GitHub)
Split usado: qa_flat_jsonl/test.jsonl  (328 pares, nunca usados en entrenamiento)

Estrategia de muestreo:
  - 15 factual  (basico + intermedio)
  - 10 definicion
  - 10 razonamiento
  - 10 aplicacion
  - 5  comparacion / hipotetico (lo que haya)
  Total ~50 pares, balanceados por tipo y dificultad.

Uso:
    from src.evaluation.sampler import load_sample
    sample = load_sample()   # list[dict]
"""

from __future__ import annotations

import json
import logging
import random
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://github.com/JhonHander/MaternaQA-es/raw/refs/heads/master/"
    "datasets/obstetrics/qa/publication/qa_flat_jsonl/test.jsonl"
)

CACHE_PATH = Path("evaluation_reports/maternaqa_test.jsonl")

# Cuántos pares queremos por tipo
QUOTA: dict[str, int] = {
    "factual":      15,
    "definicion":   10,
    "razonamiento": 10,
    "aplicacion":   10,
    "comparacion":   3,
    "hipotetico":    2,
}


def _download_test_split() -> list[dict[str, Any]]:
    """Descarga test.jsonl de GitHub y lo cachea localmente."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        logger.info(f"[Sampler] Usando cache: {CACHE_PATH}")
    else:
        logger.info(f"[Sampler] Descargando test split desde GitHub...")
        urllib.request.urlretrieve(DATASET_URL, CACHE_PATH)
        logger.info(f"[Sampler] Guardado en {CACHE_PATH}")

    records = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"[Sampler] {len(records)} pares cargados del split test")
    return records


def load_sample(seed: int = 42) -> list[dict[str, Any]]:
    """
    Devuelve una muestra estratificada del split test de MaternaQA-es.

    Args:
        seed: semilla para reproducibilidad.

    Returns:
        Lista de dicts con campos: qa_id, pregunta, respuesta,
        contexto_fuente, tipo, dificultad, topics, source_pdf.
    """
    random.seed(seed)
    all_records = _download_test_split()

    # Agrupar por tipo
    by_type: dict[str, list[dict]] = {}
    for rec in all_records:
        t = rec.get("tipo", "otro")
        by_type.setdefault(t, []).append(rec)

    sample: list[dict] = []
    for tipo, quota in QUOTA.items():
        pool = by_type.get(tipo, [])
        if not pool:
            logger.warning(f"[Sampler] No hay pares de tipo '{tipo}'")
            continue
        # Dentro de cada tipo, mezclar y tomar quota
        random.shuffle(pool)
        taken = pool[:quota]
        sample.extend(taken)
        logger.info(f"[Sampler] tipo={tipo}: {len(taken)}/{quota} pares seleccionados")

    random.shuffle(sample)
    logger.info(f"[Sampler] Muestra final: {len(sample)} pares")
    return sample
