# Setup Crítico de Evaluación RAG — NO MODIFICAR

> **IMPORTANTE:** Este archivo documenta una configuración que costó múltiples sesiones
> de debugging para estabilizar. Antes de cambiar cualquier componente del pipeline de
> evaluación, leer este documento completo.

---

## Arquitectura de dos modelos — separación de roles

El pipeline de evaluación usa **dos LLMs completamente distintos** para roles distintos.
**No son intercambiables. No usar el mismo modelo para ambos roles.**

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 1 — Generación (el sistema RAG real)                      │
│                                                                 │
│  Pregunta → FAISS retrieval → llama-3.3-70b-versatile (Groq)   │
│           → Respuesta en español + contextos recuperados        │
│           → eval_raw_<config>_<ts>.json                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  FASE 2 — Evaluación (juez externo independiente)               │
│                                                                 │
│  eval_raw_*.json → gemma-4-31b (Cerebras) como Ragas judge     │
│                  → faithfulness, answer_correctness,            │
│                    answer_relevancy, context_recall,            │
│                    context_precision                            │
│                  → eval_report_<config>_<ts>.md                │
└─────────────────────────────────────────────────────────────────┘
```

| Rol | Modelo | Provider | Key en .env | Por qué |
|---|---|---|---|---|
| Chatbot (sistema evaluado) | `llama-3.3-70b-versatile` | Groq | `GROQ_API_KEY` | Es el LLM de producción del sistema RAG |
| Ragas judge (evaluador) | `gemma-4-31b` | Cerebras | `CEREBRAS_KEY` | Independiente, sin cuota diaria estricta, JSON válido |

La separación es obligatoria: si el mismo modelo evaluara sus propias respuestas, las métricas de faithfulness y answer_correctness estarían sesgadas hacia sus propios patrones de respuesta.

---

## Por qué Cerebras gemma-4-31b como juez (y no otros modelos)

Esta decisión fue el resultado de múltiples intentos fallidos. **No revertir sin leer esto.**

### Modelos descartados y razón

| Modelo | Problema |
|---|---|
| `llama-3.3-70b` (Groq, judge) | ~4.500 tokens/par en faithfulness → agota 100k/día en ~12 pares |
| `llama-3.1-8b-instant` (Groq) | Genera JSON válido directo, pero falla en el loop de reintentos de Ragas (`RagasOutputParserException`) |
| `gemma2-9b-it` (Groq) | Dado de baja por Groq (error 400) |
| `llama-3.2-3b-preview` (Groq) | Dado de baja por Groq (error 400) |
| `gpt-oss-120b` (Cerebras) | Devuelve `None` en `choices[0].message.content` |
| OpenRouter modelos `:free` | 404 o 429 constante — no disponibles de forma estable |
| `nvidia/nemotron-*` (OpenRouter) | Rate limit bajo — 429 frecuente |

### Por qué gemma-4-31b (Cerebras) funciona

1. **JSON válido y confiable** en el prompt interno de Ragas (verificado en prueba directa con `evaluate()`)
2. **~296 tokens/par** — 15x menos que llama-3.3-70b, 15 pares × 5 métricas ≈ 22k tokens total
3. **Sin límite diario de tokens** — límite es por minuto, no por día → permite completar evaluaciones largas
4. **Completó 15/15 pares** en evaluación real sin un solo error de rate limit ni parser

---

## Fix crítico: `is_finished_parser` permisivo

Ragas verifica el `finish_reason` de cada respuesta del LLM. Si el modelo termina con
`"length"` (respuesta truncada) en vez de `"stop"`, lanza `LLMDidNotFinishException`
y marca el par como fallido.

**Este fix es obligatorio** en `_make_llm()` dentro de `eval_pipeline.py`:

```python
from langchain_core.outputs import LLMResult

def _is_finished(response: LLMResult) -> bool:
    """
    Parser permisivo: acepta stop, length, MAX_TOKENS, end_turn.
    Sin esto, Ragas falla en pares con respuestas largas en español.
    """
    VALID = {"stop", "STOP", "length", "MAX_TOKENS", "end_turn", "eos"}
    for g in response.flatten():
        resp = g.generations[0][0]
        finish = None
        if resp.generation_info:
            finish = resp.generation_info.get("finish_reason") or resp.generation_info.get("stop_reason")
        if finish is None and hasattr(resp, "message"):
            meta = getattr(resp.message, "response_metadata", {})
            finish = meta.get("finish_reason") or meta.get("stop_reason")
        if finish is not None and finish not in VALID:
            return False
    return True

# Usar así:
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gemma-4-31b",
        api_key=settings.cerebras_key,
        base_url="https://api.cerebras.ai/v1",
        temperature=0,
        max_tokens=1024,
    ),
    is_finished_parser=_is_finished,
)
```

---

## Configuración de RunConfig — no usar defaults de Ragas

Los defaults de Ragas (`max_workers=16`, `max_retries=10`) lanzan demasiados jobs
concurrentes y agotan la cuota de cualquier API gratuita en minutos.

```python
from ragas.run_config import RunConfig

run_cfg = RunConfig(
    max_workers=1,    # un job a la vez — sin ráfagas
    max_retries=2,    # máximo 2 reintentos (no 10)
    max_wait=15,
    timeout=120,
)
```

Con `batch_size=1` en `evaluate()` se procesa un par a la vez, predeciblemente.

---

## Variables de entorno requeridas (.env)

```env
# Sistema RAG — chatbot de producción
GROQ_API_KEY=gsk_...           # llama-3.3-70b-versatile (fase 1)
GROQ_API_KEY_2=gsk_...         # fallback / segunda key Groq

# Evaluación Ragas — juez independiente
CEREBRAS_KEY=csk_...           # gemma-4-31b (fase 2)
OPENROUTER_KEY=sk-or-v1-...    # backup, actualmente inestable para Ragas
```

Todos registrados en `src/settings.py` como campos Pydantic.

---

## Comandos de ejecución

```bash
# FASE 1: generar respuestas (usa GROQ_API_KEY + FAISS)
python src/evaluation/eval_pipeline.py --config configB --sample 15 --generate-only
# Salida: evaluation_reports/eval_raw_configB_<ts>.json

# FASE 2: evaluar con Ragas (usa CEREBRAS_KEY)
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_configB_<ts>.json
# Salida: evaluation_reports/eval_report_configB_<ts>.md
#         evaluation_reports/eval_results_configB_<ts>.json

# Smoke test (3 pares para verificar que el pipeline funciona antes de correr todo)
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_configB_smoketest.json
```

**Tiempo estimado fase 2 con 15 pares y Cerebras:** ~35-40 minutos (secuencial, batch_size=1)

---

## Resultados finales — Comparativa A vs B vs C

Evaluaciones completadas: **19 de julio de 2026 — 15/15 pares sin errores en las 3 configs**
Judge: Cerebras `gemma-4-31b` | Dataset: MaternaQA-es split test (seed=42, mismos 15 pares)

| Metrica | Config A (FAISS puro) | Config B (FAISS+BM25) | Config C (+obstetrics ES) | Mejor |
|:--------|:--------------------:|:---------------------:|:------------------------:|:-----:|
| `faithfulness` | 0.1615 | **0.2278** | 0.1334 | **B** |
| `answer_correctness` | 0.3500 | 0.3381 | **0.3782** | **C** |
| `answer_relevancy` | 0.6345 | 0.6305 | **0.6909** | **C** |
| `context_recall` | 0.000 | 0.000 | **0.033** | **C** |
| `context_precision` | 0.000 | 0.000 | **0.143** | **C** |
| `latency_avg_s` | 11.35s | 10.36s | **10.26s** | **C** |

**Interpretacion clave:**

- **context_recall y context_precision**: Config C es la unica con valores reales > 0.
  Primer resultado significativo tras ingestar 2.112 chunks de obstetricia en espanol.

- **faithfulness C < B**: los chunks del corpus LM son muy densos (~879 tok) con texto
  de guias clinicas. El LLM responde mas generalmente sin anclarse en parrafos especificos
  → Ragas penaliza. Probable mejora con re-chunking a ~400 tok.

- **answer_relevancy C=0.691** > baseline test 0.5583 — mejor resultado de la serie.
  El contexto en espanol ayuda al LLM a responder mas pertinentemente.

- **answer_correctness C=0.378** > B y A — mayor coincidencia semantica con ground truth.

**Arquitectura de produccion: Config B** (faithfulness mas alto, corpus limpio).
**Config C** es el camino correcto pero requiere optimizacion del chunking del corpus ES.

Archivos Config C:
- Raw: `evaluation_reports/eval_raw_configC_20260719_215215.json`
- Resultados: `evaluation_reports/eval_results_configC_20260719_215215.json`
- Reporte MD: `evaluation_reports/eval_report_configC_20260719_215215.md`

| Archivo | Config | Estado |
|---|---|---|
| `src/rag/retriever.py` | **Config B (activa)** | Produccion — hibrido FAISS+BM25 |
| `src/rag/retriever_configA.py` | Config A | FAISS puro — para referencia/replicar |
| `src/rag/retriever_configB.py` | Config B (backup) | Copia de seguridad del actual |
| `src/rag/retriever_configC.py` | Config C | FAISS+BM25 + maternaqaes_lm (2112 chunks ES) |

---

## Archivos clave del pipeline

| Archivo | Rol | NO tocar sin leer este doc |
|---|---|---|
| `src/evaluation/eval_pipeline.py` | Pipeline completo fases 1 y 2 | `_make_llm()`, `_is_finished()`, `phase_evaluate()` |
| `src/evaluation/sampler.py` | Muestrea MaternaQA-es estratificadamente | — |
| `src/settings.py` | `cerebras_key`, `openrouter_key`, `groq_api_key_2` | — |
| `foragents/retrieval_arquitecturas_configs.md` | Config A vs Config B — código para replicar | — |
| `evaluation_reports/` | Reportes JSON y MD (gitignored) | No versionar — muy pesados |

---

*Creado: 18 de julio de 2026 | Actualizado: 19 de julio de 2026 — evaluacion 3 configs completada (A, B, C)*
*Basado en: Q23-Q27 de `foragents/qa_technical.md`*
