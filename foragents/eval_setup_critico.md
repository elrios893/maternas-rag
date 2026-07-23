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

## Archivos clave del pipeline

| Archivo | Rol | NO tocar sin leer este doc |
|---|---|---|
| `src/evaluation/eval_pipeline.py` | Pipeline completo fases 1 y 2 | `_make_llm()`, `_is_finished()`, `phase_evaluate()` |
| `src/evaluation/sampler.py` | Muestrea MaternaQA-es estratificadamente | — |
| `src/settings.py` | `cerebras_key`, `openrouter_key`, `groq_api_key_2` | — |
| `foragents/retrieval_arquitecturas_configs.md` | Config A vs Config B — código para replicar | — |
| `evaluation_reports/` | Reportes JSON y MD (gitignored) | No versionar — muy pesados |

---

*Creado: 18 de julio de 2026 | Actualizado: 21 de julio de 2026*
*Ver resultados completos en: `foragents/eval_runbook.md`*
