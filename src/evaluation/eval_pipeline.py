"""
eval_pipeline.py — Pipeline de evaluacion RAG con MaternaQA-es y Ragas.

Flujo en DOS fases separadas para evitar conflictos CUDA/CPU:

  FASE 1 (--generate): genera respuestas con el chatbot (usa GPU para embeddings)
    → guarda evaluation_reports/eval_raw_<ts>.json

  FASE 2 (--evaluate): evalua con Ragas el raw JSON generado (usa CPU)
    → guarda evaluation_reports/eval_results_<ts>.json
    → guarda evaluation_reports/eval_report_<ts>.md

  FASE COMPLETA (default): ejecuta ambas fases en secuencia.

Uso:
    python src/evaluation/eval_pipeline.py                   # completo (~50 pares)
    python src/evaluation/eval_pipeline.py --sample 10      # muestra reducida
    python src/evaluation/eval_pipeline.py --generate-only  # solo fase 1
    python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_XXX.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("evaluation_reports")


# ---------------------------------------------------------------------------
# FASE 1: generar respuestas con el chatbot
# ---------------------------------------------------------------------------

def phase_generate(sample_size: int | None, seed: int) -> Path:
    """
    Pasa cada pregunta de la muestra por chat() y guarda las respuestas.
    Retorna la ruta del archivo raw JSON generado.
    """
    from src.rag.chain import chat
    from src.rag.retriever import retrieve
    from src.evaluation.sampler import load_sample

    print("=" * 60)
    print("FASE 1 — Generando respuestas del chatbot Maternas")
    print("=" * 60)

    sample = load_sample(seed=seed)
    if sample_size:
        sample = sample[:sample_size]
    print(f"\nMuestra: {len(sample)} pares\n")

    rows = []
    failed = 0

    for i, item in enumerate(sample, 1):
        pregunta = item["pregunta"]
        tipo = item.get("tipo", "?")
        dificultad = item.get("dificultad", "?")
        print(f"  [{i:02d}/{len(sample)}] tipo={tipo:12s} dif={dificultad:12s} | {pregunta[:55]}...")

        try:
            result = chat(pregunta)

            # Recuperar textos completos para context_recall
            docs_full = retrieve(pregunta, k=5)
            contexts_text = [d.get("text", "")[:600] for d in docs_full]

            rows.append({
                "qa_id":               item.get("qa_id", f"item_{i}"),
                "question":            pregunta,
                "answer":              result.answer,
                "contexts":            contexts_text,
                "ground_truth":        item["respuesta"],
                "reference":           item["respuesta"],
                "contexto_fuente":     item.get("contexto_fuente", ""),
                "tipo":                tipo,
                "dificultad":          dificultad,
                "source_pdf":          item.get("source_pdf", ""),
                "topics":              item.get("topics", []),
                "needs_clarification": result.needs_clarification,
                "intent":              result.intent,
                "risk_level":          result.risk_level,
            })

        except Exception as e:
            logger.error(f"Error en item {i} ({pregunta[:40]}): {e}")
            failed += 1

        time.sleep(1.5)  # respetar rate limit de Groq

    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = REPORTS_DIR / f"eval_raw_{ts}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "n_total": len(sample),
            "n_ok": len(rows),
            "n_failed": failed,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  OK: {len(rows)} | Fallidos: {failed}")
    print(f"  Guardado: {raw_path}\n")
    return raw_path


# ---------------------------------------------------------------------------
# FASE 2: evaluar con Ragas
# ---------------------------------------------------------------------------

def phase_evaluate(raw_path: Path) -> dict:
    """
    Lee el raw JSON de la fase 1 y evalua con Ragas.
    Retorna el dict de resultados completo.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from datasets import Dataset
    from src.settings import settings

    print("=" * 60)
    print("FASE 2 — Evaluando con Ragas")
    print("=" * 60)

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = raw["rows"]
    ts = raw["timestamp"]
    print(f"\n  Evaluando {len(rows)} pares con Ragas...\n")

    # LLM judge
    llm = LangchainLLMWrapper(ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    ))

    # Embeddings en CPU (Ragas no necesita GPU)
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
            "reference":    r["reference"],
        }
        for r in rows
    ])

    metrics_by_row: list[dict] = [{} for _ in rows]
    try:
        result_ragas = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=llm,
            embeddings=emb,
            raise_exceptions=False,
        )
        df = result_ragas.to_pandas()
        for i in range(len(rows)):
            if i < len(df):
                metrics_by_row[i] = {
                    "faithfulness":     float(df.iloc[i].get("faithfulness", -1)),
                    "answer_relevancy": float(df.iloc[i].get("answer_relevancy", -1)),
                    "context_recall":   float(df.iloc[i].get("context_recall", -1)),
                }
    except Exception as e:
        logger.error(f"Error Ragas: {e}")

    # Enriquecer rows con metricas
    for i, row in enumerate(rows):
        row.update(metrics_by_row[i])

    # Metricas agregadas
    def mean(vals):
        v = [x for x in vals if isinstance(x, float) and x >= 0]
        return round(sum(v) / len(v), 4) if v else -1

    metrics_global = {
        "faithfulness":     mean([r.get("faithfulness", -1) for r in rows]),
        "answer_relevancy": mean([r.get("answer_relevancy", -1) for r in rows]),
        "context_recall":   mean([r.get("context_recall", -1) for r in rows]),
        "n_evaluated":      len(rows),
        "n_failed":         raw["n_failed"],
    }

    # Por tipo
    by_tipo: dict[str, list] = {}
    for r in rows:
        by_tipo.setdefault(r["tipo"], []).append(r)

    metrics_by_tipo = {
        t: {
            "n": len(rs),
            "faithfulness":     mean([r.get("faithfulness", -1) for r in rs]),
            "answer_relevancy": mean([r.get("answer_relevancy", -1) for r in rs]),
            "context_recall":   mean([r.get("context_recall", -1) for r in rs]),
        }
        for t, rs in by_tipo.items()
    }

    # Por dificultad
    by_dif: dict[str, list] = {}
    for r in rows:
        by_dif.setdefault(r["dificultad"], []).append(r)

    metrics_by_dificultad = {
        d: {
            "n": len(rs),
            "faithfulness":     mean([r.get("faithfulness", -1) for r in rs]),
            "answer_relevancy": mean([r.get("answer_relevancy", -1) for r in rs]),
            "context_recall":   mean([r.get("context_recall", -1) for r in rs]),
        }
        for d, rs in by_dif.items()
    }

    results = {
        "timestamp":             ts,
        "dataset":               "JhonHander/MaternaQA-es (split: test)",
        "n_sample":              raw["n_total"],
        "n_evaluated":           len(rows),
        "n_failed":              raw["n_failed"],
        "metrics_global":        metrics_global,
        "metrics_by_tipo":       metrics_by_tipo,
        "metrics_by_dificultad": metrics_by_dificultad,
        "rows":                  rows,
    }

    # Guardar
    json_path = REPORTS_DIR / f"eval_results_{ts}.json"
    md_path   = REPORTS_DIR / f"eval_report_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    _write_markdown_report(results, md_path)

    # Imprimir resumen
    print("\nRESULTADOS GLOBALES:")
    for k, v in metrics_global.items():
        print(f"  {k:<22}: {v}")
    print("\nPOR TIPO:")
    for t, m in sorted(metrics_by_tipo.items(), key=lambda x: -x[1].get("faithfulness", 0)):
        print(f"  {t:<16} n={m['n']:2d}  "
              f"faith={m['faithfulness']:.3f}  "
              f"rel={m['answer_relevancy']:.3f}  "
              f"recall={m['context_recall']:.3f}")
    print("\nPOR DIFICULTAD:")
    for d, m in sorted(metrics_by_dificultad.items()):
        print(f"  {d:<12} n={m['n']:2d}  "
              f"faith={m['faithfulness']:.3f}  "
              f"rel={m['answer_relevancy']:.3f}  "
              f"recall={m['context_recall']:.3f}")
    print(f"\n  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Reporte Markdown
# ---------------------------------------------------------------------------

def _write_markdown_report(results: dict, path: Path) -> None:
    ts = results["timestamp"]
    mg = results.get("metrics_global", {})
    mt = results.get("metrics_by_tipo", {})
    md = results.get("metrics_by_dificultad", {})

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and v >= 0 else "N/A"

    lines = [
        "# Reporte de Evaluacion RAG — Maternas",
        "",
        f"**Fecha:** {ts[:4]}-{ts[4:6]}-{ts[6:8]}  ",
        f"**Dataset:** MaternaQA-es · split test (JhonHander/MaternaQA-es)  ",
        f"**Pares evaluados:** {results['n_evaluated']}  ",
        f"**Fallidos en generacion:** {results['n_failed']}  ",
        "",
        "## Metricas Globales",
        "",
        "| Metrica | Valor | Referencia MaternaQA-es |",
        "|---|---|---|",
        f"| faithfulness | {fmt(mg.get('faithfulness',-1))} | 0.7726 (train) |",
        f"| answer_relevancy | {fmt(mg.get('answer_relevancy',-1))} | 0.6466 (train) |",
        f"| context_recall | {fmt(mg.get('context_recall',-1))} | N/A (corpus distinto) |",
        "",
        "> **faithfulness**: la respuesta esta respaldada por los fragmentos recuperados",
        "> **answer_relevancy**: la respuesta es pertinente a la pregunta",
        "> **context_recall**: el retrieval capturo informacion relevante del ground truth",
        "",
        "## Metricas por Tipo de Pregunta",
        "",
        "| Tipo | N | Faithfulness | Answer Relevancy | Context Recall |",
        "|---|---|---|---|---|",
    ]
    for tipo, m in sorted(mt.items(), key=lambda x: -x[1].get("faithfulness", 0)):
        lines.append(
            f"| {tipo} | {m['n']} | {fmt(m['faithfulness'])} | "
            f"{fmt(m['answer_relevancy'])} | {fmt(m['context_recall'])} |"
        )

    lines += [
        "",
        "## Metricas por Dificultad",
        "",
        "| Dificultad | N | Faithfulness | Answer Relevancy | Context Recall |",
        "|---|---|---|---|---|",
    ]
    for dif, m in sorted(md.items()):
        lines.append(
            f"| {dif} | {m['n']} | {fmt(m['faithfulness'])} | "
            f"{fmt(m['answer_relevancy'])} | {fmt(m['context_recall'])} |"
        )

    lines += [
        "",
        "## Notas de Interpretacion",
        "",
        "- **context_recall bajo** es esperable: el corpus RAG actual (textbooks EN +",
        "  MedMCQA + Multiclinsum) no incluye los PDFs de MaternaQA-es (GPC Colombia,",
        "  revistas de obstetricia). Mejorara significativamente tras ingestar esos chunks.",
        "- **faithfulness alto** implica que el LLM no inventa datos cuando tiene contexto.",
        "- **answer_relevancy** evalua pertinencia independientemente de las fuentes.",
        "",
        "## Comparacion con Linea Base MaternaQA-es",
        "",
        "| Sistema | Faithfulness | Answer Relevancy |",
        "|---|---|---|",
        f"| Maternas RAG (este reporte) | {fmt(mg.get('faithfulness',-1))} | {fmt(mg.get('answer_relevancy',-1))} |",
        "| MaternaQA-es baseline (train) | 0.7726 | 0.6466 |",
        "| MaternaQA-es baseline (test) | 0.7132 | 0.5583 |",
        "",
        "---",
        "*Generado por src/evaluation/eval_pipeline.py*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluacion RAG Maternas vs MaternaQA-es")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true",
                        help="Solo fase 1: generar respuestas")
    parser.add_argument("--evaluate-only", type=str, default=None, metavar="RAW_JSON",
                        help="Solo fase 2: evaluar un raw JSON ya generado")
    args = parser.parse_args()

    if args.evaluate_only:
        phase_evaluate(Path(args.evaluate_only))
    elif args.generate_only:
        phase_generate(args.sample, args.seed)
    else:
        raw_path = phase_generate(args.sample, args.seed)
        phase_evaluate(raw_path)

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
from src.rag.chain import chat
from src.settings import settings
from src.evaluation.sampler import load_sample

# ---------------------------------------------------------------------------
# Imports de Ragas
# ---------------------------------------------------------------------------
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset


# ---------------------------------------------------------------------------
# Configurar LLM judge y embeddings para Ragas
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    """LLM judge: Groq llama-3.3-70b-versatile via LangChain wrapper."""
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Embeddings para answer_relevancy: mismo modelo del proyecto, en CPU para Ragas."""
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},   # Ragas corre en CPU, el proyecto usa CUDA
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_evaluation(
    sample_size: int | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """
    Ejecuta la evaluacion RAG sobre la muestra estratificada de MaternaQA-es.

    Args:
        sample_size: si se pasa, limita la muestra a N pares (util para pruebas rapidas).
        seed:        semilla para reproducibilidad del muestreo.
        dry_run:     si True, no llama al LLM judge de Ragas (solo genera las respuestas).

    Returns:
        dict con metricas agregadas y detalle por par.
    """
    print("=" * 60)
    print("EVALUACION RAG — Maternas vs MaternaQA-es")
    print("=" * 60)

    # 1. Cargar muestra
    sample = load_sample(seed=seed)
    if sample_size:
        sample = sample[:sample_size]
    print(f"\n[1/4] Muestra: {len(sample)} pares del split test de MaternaQA-es\n")

    # 2. Generar respuestas con el sistema RAG
    print("[2/4] Generando respuestas con el chatbot Maternas...")
    rows = []
    failed = 0

    for i, item in enumerate(sample, 1):
        pregunta = item["pregunta"]
        respuesta_ref = item["respuesta"]
        contexto_ref = item.get("contexto_fuente", "")
        tipo = item.get("tipo", "?")
        dificultad = item.get("dificultad", "?")

        print(f"  [{i:02d}/{len(sample)}] tipo={tipo} dif={dificultad} | {pregunta[:60]}...")

        try:
            result = chat(pregunta)

            # Si el sistema pidio clarificacion, usamos la pregunta tal cual
            # (es un comportamiento valido — evaluamos si la respuesta igualmente informa)
            answer = result.answer
            # Fragmentos recuperados como contextos para Ragas
            contexts = []
            for doc in result.sources:
                # Reconstruimos texto desde sources (sin el campo text, usamos metadata)
                src = doc.get("source_dataset", "")
                ctx = f"[{src}]"
                contexts.append(ctx)

            # Para context_recall necesitamos el texto real de los fragmentos
            # Lo recuperamos directamente del retriever
            from src.rag.retriever import retrieve
            docs_full = retrieve(pregunta, k=5)
            contexts_text = [d.get("text", "")[:500] for d in docs_full]

            rows.append({
                "qa_id":            item.get("qa_id", f"item_{i}"),
                "question":         pregunta,
                "answer":           answer,
                "contexts":         contexts_text,
                "ground_truth":     respuesta_ref,
                "reference":        respuesta_ref,
                "tipo":             tipo,
                "dificultad":       dificultad,
                "source_pdf":       item.get("source_pdf", ""),
                "needs_clarification": result.needs_clarification,
                "intent":           result.intent,
                "risk_level":       result.risk_level,
            })

        except Exception as e:
            logger.error(f"Error en item {i}: {e}")
            failed += 1
            # Pausa para no exceder rate limit
            time.sleep(2)
            continue

        # Pausa entre llamadas para respetar rate limit de Groq
        time.sleep(1.5)

    print(f"\n  Completados: {len(rows)} | Fallidos: {failed}")

    # Guardar respuestas intermedias (por si Ragas falla)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("evaluation_reports")
    reports_dir.mkdir(exist_ok=True)

    raw_path = reports_dir / f"eval_raw_{ts}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n  Respuestas intermedias guardadas: {raw_path}")

    if dry_run or not rows:
        print("\n[dry-run] Saltando evaluacion Ragas.")
        return {"rows": rows, "metrics": {}, "raw_path": str(raw_path)}

    # 3. Evaluar con Ragas
    print("\n[3/4] Evaluando con Ragas (faithfulness + answer_relevancy + context_recall)...")
    print("  Esto puede tardar varios minutos por las llamadas al LLM judge...\n")

    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
            "reference":    r["reference"],
        }
        for r in rows
    ])

    try:
        result_ragas = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False,
        )
        metrics_df = result_ragas.to_pandas()

        # Agregar metricas por fila de vuelta a rows
        for i, row in enumerate(rows):
            if i < len(metrics_df):
                row["faithfulness"]     = float(metrics_df.iloc[i].get("faithfulness", -1))
                row["answer_relevancy"] = float(metrics_df.iloc[i].get("answer_relevancy", -1))
                row["context_recall"]   = float(metrics_df.iloc[i].get("context_recall", -1))

        # Metricas agregadas globales
        def safe_mean(vals):
            v = [x for x in vals if x >= 0]
            return round(sum(v) / len(v), 4) if v else -1

        metrics_global = {
            "faithfulness":     safe_mean([r.get("faithfulness", -1) for r in rows]),
            "answer_relevancy": safe_mean([r.get("answer_relevancy", -1) for r in rows]),
            "context_recall":   safe_mean([r.get("context_recall", -1) for r in rows]),
            "n_evaluated":      len(rows),
            "n_failed":         failed,
        }

        # Metricas por tipo
        by_tipo: dict[str, list] = {}
        for row in rows:
            t = row["tipo"]
            by_tipo.setdefault(t, []).append(row)

        metrics_by_tipo = {}
        for t, t_rows in by_tipo.items():
            metrics_by_tipo[t] = {
                "n": len(t_rows),
                "faithfulness":     safe_mean([r.get("faithfulness", -1) for r in t_rows]),
                "answer_relevancy": safe_mean([r.get("answer_relevancy", -1) for r in t_rows]),
                "context_recall":   safe_mean([r.get("context_recall", -1) for r in t_rows]),
            }

    except Exception as e:
        logger.error(f"Error en evaluacion Ragas: {e}")
        metrics_global = {"error": str(e), "n_evaluated": len(rows)}
        metrics_by_tipo = {}

    # 4. Guardar resultados finales
    print("\n[4/4] Guardando reportes...")

    results = {
        "timestamp":        ts,
        "dataset":          "JhonHander/MaternaQA-es (split: test)",
        "n_sample":         len(sample),
        "n_evaluated":      len(rows),
        "n_failed":         failed,
        "metrics_global":   metrics_global,
        "metrics_by_tipo":  metrics_by_tipo,
        "rows":             rows,
    }

    json_path = reports_dir / f"eval_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_path = reports_dir / f"eval_report_{ts}.md"
    _write_markdown_report(results, md_path)

    print(f"\n  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    print("\n" + "=" * 60)
    print("RESULTADOS GLOBALES")
    print("=" * 60)
    for k, v in metrics_global.items():
        print(f"  {k:<22}: {v}")
    print("\nPOR TIPO DE PREGUNTA:")
    for tipo, m in metrics_by_tipo.items():
        print(f"  {tipo:<15} n={m['n']:2d}  "
              f"faith={m['faithfulness']:.3f}  "
              f"rel={m['answer_relevancy']:.3f}  "
              f"recall={m['context_recall']:.3f}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Reporte Markdown
# ---------------------------------------------------------------------------

def _write_markdown_report(results: dict, path: Path) -> None:
    ts = results["timestamp"]
    mg = results.get("metrics_global", {})
    mt = results.get("metrics_by_tipo", {})

    lines = [
        "# Reporte de Evaluacion RAG — Maternas",
        "",
        f"**Fecha:** {ts[:8][:4]}-{ts[4:6]}-{ts[6:8]}  ",
        f"**Dataset:** MaternaQA-es (split test)  ",
        f"**Pares evaluados:** {results['n_evaluated']}  ",
        f"**Fallidos:** {results['n_failed']}  ",
        "",
        "## Metricas Globales",
        "",
        "| Metrica | Valor |",
        "|---|---|",
    ]
    for k, v in mg.items():
        if k not in ("n_evaluated", "n_failed", "error"):
            lines.append(f"| {k} | {v:.4f} |")

    lines += [
        "",
        "> **faithfulness**: que tan respaldada esta la respuesta por los fragmentos recuperados (0-1)",
        "> **answer_relevancy**: que tan bien responde la pregunta formulada (0-1)",
        "> **context_recall**: que tanto del contexto de referencia fue recuperado (0-1)",
        "",
        "## Metricas por Tipo de Pregunta",
        "",
        "| Tipo | N | Faithfulness | Answer Relevancy | Context Recall |",
        "|---|---|---|---|---|",
    ]
    for tipo, m in sorted(mt.items(), key=lambda x: -x[1].get("faithfulness", 0)):
        lines.append(
            f"| {tipo} | {m['n']} | {m['faithfulness']:.3f} | "
            f"{m['answer_relevancy']:.3f} | {m['context_recall']:.3f} |"
        )

    lines += [
        "",
        "## Notas de Interpretacion",
        "",
        "- **context_recall bajo** es esperable: el sistema RAG actual no tiene indexados",
        "  los PDFs de MaternaQA-es (GPC Colombia, revistas de obstetricia). Los fragmentos",
        "  provienen de textbooks EN y MedMCQA, que cubren el dominio pero no los documentos exactos.",
        "- **faithfulness** mide si el LLM se inventa cosas o se cine a lo que recupero.",
        "  Un valor alto indica que el sistema es honesto con sus fuentes.",
        "- **answer_relevancy** mide si la respuesta es pertinente a la pregunta,",
        "  independientemente de si usa las fuentes correctas.",
        "",
        "## Interpretacion de Linea Base",
        "",
        "| Rango | Significado |",
        "|---|---|",
        "| > 0.80 | Bueno |",
        "| 0.60 – 0.80 | Aceptable, hay margen de mejora |",
        "| < 0.60 | Requiere mejoras significativas |",
        "",
        "---",
        "*Generado automaticamente por src/evaluation/eval_pipeline.py*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluacion RAG Maternas vs MaternaQA-es")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limitar muestra a N pares (default: todos los estratificados ~50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo genera respuestas, sin evaluar con Ragas")
    args = parser.parse_args()

    run_evaluation(
        sample_size=args.sample,
        seed=args.seed,
        dry_run=args.dry_run,
    )
