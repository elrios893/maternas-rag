# Guía de Evaluación RAG — Maternas

> **LEER ANTES DE TOCAR CUALQUIER ARCHIVO DEL PIPELINE DE EVALUACIÓN.**
> Ver también `foragents/eval_setup_critico.md` para decisiones arquitectónicas
> y resultados históricos.

---

## Arquitectura de dos modelos — NO confundir

```
FASE 1 (generación)        FASE 2 (evaluación)
llama-3.3-70b (Groq)  →   gemma-4-31b (Cerebras)
GROQ_API_KEY               CEREBRAS_KEY
El chatbot RAG real        El juez de Ragas
```

**Nunca usar el mismo modelo para ambas fases.** El juez debe ser independiente.

---

## Variables de entorno requeridas (.env)

```env
GROQ_API_KEY=gsk_...        # chatbot — fase 1
CEREBRAS_KEY=csk_...        # Ragas judge — fase 2
GROQ_API_KEY_2=gsk_...      # backup Groq (opcional)
OPENROUTER_KEY=sk-or-...    # backup OpenRouter (opcional, inestable)
```

---

## Configs de retrieval disponibles

| Archivo | Activar con | Descripción |
|---|---|---|
| `src/rag/retriever_configA.py` | `copy configA retriever.py` | FAISS puro, top-k global sin filtro |
| `src/rag/retriever_configB.py` | `copy configB retriever.py` | FAISS+BM25 híbrido (producción actual) |
| `src/rag/retriever_configC.py` | `copy configC retriever.py` | FAISS+BM25 + corpus obstetrics ES |
| `src/rag/retriever.py` | — | **Activo en producción — Config B** |

**Siempre restaurar Config B al terminar una evaluación:**
```bash
copy src\rag\retriever_configB.py src\rag\retriever.py
```

---

## Corpus en el índice FAISS

| Dataset | Chunks | Idioma | Fuente |
|---|---|---|---|
| textbook | ~135k | EN | 18 libros médicos |
| medmcqa | ~187k | EN | Exámenes médicos India |
| medqa_* | ~53k | EN/ZH | USMLE + Taiwan + Mainland |
| multiclinsum | ~51.8k | ES | Casos clínicos reales |
| **maternaqaes_lm** | **5.063** | **ES** | **Corpus obstetrico colombiano (train+val, rechunked 336tok)** |
| **Total** | **~380.455** | — | — |

**IMPORTANTE sobre maternaqaes_lm:**
- Solo contiene splits `train` y `validation` del corpus LM
- El split `test` fue excluido deliberadamente para evitar data leakage
- Los 3 PDFs del split test son: `GPC-Atencion-Prenatal-de-Bajo-Riesgo-2023.pdf`, `vol831-1.pdf`, `4142_stamped.pdf`
- Re-chunkeado a ~336 tok con `clinical_score >= 15` (sin intro/biblio)
- Script de ingestión: `src/ingestion/ingest_maternaqaes_lm.py`

---

## Comandos para correr una evaluación completa

### Paso 0 — Verificar tokens disponibles

```bash
python C:\Users\Usuario\AppData\Local\Temp\opencode\check_quota.py
# Debe mostrar: KEY_1: OK | KEY_2: OK
# Si muestra LIMIT, esperar renovación a las 00:00 UTC
```

### Paso 1 — Activar la config a evaluar

```bash
# Config B (producción actual)
copy src\rag\retriever_configB.py src\rag\retriever.py

# Config C (+ corpus obstetrics ES)
copy src\rag\retriever_configC.py src\rag\retriever.py
```

### Paso 2 — Generar respuestas (Fase 1)

```bash
python src/evaluation/eval_pipeline.py --config configB --sample 15 --generate-only
# Salida: evaluation_reports/eval_raw_configB_<ts>.json
```

**Verificar SIEMPRE que no hay fallbacks antes de fase 2:**
```bash
# Abrir el JSON y confirmar que ninguna respuesta contiene "Lo siento, tuve un problema"
# Si hay fallbacks, la KEY_1 estaba agotada — regenerar con tokens frescos
```

### Paso 3 — Evaluar con Ragas (Fase 2)

```bash
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports\eval_raw_configB_<ts>.json
# Salida: evaluation_reports/eval_report_configB_<ts>.md
#         evaluation_reports/eval_results_configB_<ts>.json
```

**Tiempo estimado fase 2:** ~35-45 minutos para 15 pares con Cerebras.

### Paso 4 — Restaurar Config B

```bash
copy src\rag\retriever_configB.py src\rag\retriever.py
```

---

## Flujo completo en un solo bloque (copiar y pegar)

```bash
# 1. Activar config
copy src\rag\retriever_configC.py src\rag\retriever.py

# 2. Generar (usa GROQ_API_KEY)
python src/evaluation/eval_pipeline.py --config configC --sample 15 --generate-only

# 3. Verificar fallbacks en el JSON generado
# 4. Evaluar (usa CEREBRAS_KEY)
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports\eval_raw_configC_<ts>.json

# 5. Restaurar producción
copy src\rag\retriever_configB.py src\rag\retriever.py
```

---

## Métricas que calcula Ragas y su interpretación

| Métrica | Qué mide | Referencia baseline test |
|---|---|---|
| `faithfulness` | ¿La respuesta está respaldada por los fragmentos recuperados? | 0.7132 |
| `answer_correctness` | ¿Qué tan correcta es la respuesta vs el ground truth? | N/A |
| `answer_relevancy` | ¿La respuesta responde bien la pregunta? | 0.5583 |
| `context_recall` | ¿El retrieval capturó contexto del ground truth? | N/A |
| `context_precision` | ¿Los fragmentos recuperados son precisos y útiles? | N/A |
| `latency_s` | Tiempo end-to-end por par (medido en fase 1) | — |

**Por qué context_recall y context_precision son bajos:**
El corpus RAG no contiene los PDFs exactos del benchmark MaternaQA-es (split test excluido por leakage). Los valores subirán al ingestar esos documentos.

---

## Resultados históricos (seed=42, 15 pares, Cerebras judge)

| Config | Faithfulness | Ans. Correct. | Ans. Relev. | Ctx. Recall | Ctx. Prec. |
|---|---|---|---|---|---|
| A — FAISS puro | 0.162 | 0.350 | 0.634 | 0.000 | 0.000 |
| B — FAISS+BM25 | 0.228 | 0.338 | 0.631 | 0.000 | 0.000 |
| C v1 — +obstetrics 879tok | 0.133 | 0.378 | 0.691 | 0.033 | 0.143 |
| **C v2 — +obstetrics 336tok** | **0.358** | **0.337** | **0.631** | **0.067** | **0.083** |
| Baseline MaternaQA-es (test) | 0.713 | — | 0.558 | — | — |

**Config activa en producción: B**
**Mejor faithfulness hasta ahora: C v2 (0.358)**

---

## Por qué faithfulness no llega al baseline (0.71)

1. **Causa principal**: los 3 PDFs del split test (`GPC-Atencion-Prenatal-2023`, `vol831-1`, `4142_stamped`) no están en el índice — son exactamente los documentos que generaron las 328 preguntas del benchmark. El LLM responde desde conocimiento general, no desde fuentes indexadas.

2. **Segunda causa**: cuando el sistema lanza una clarification question, faithfulness=0 automáticamente porque una pregunta no tiene statements verificables. Excluir `needs_clarification=True` de la muestra mejoraría el promedio.

3. **Tercera causa**: varianza alta con 15 pares (std~0.25). Con 30 pares la estimación sería más estable.

### Cómo mejorar faithfulness sin ingestar el split test

- **System prompt más estricto**: instruir al LLM a decir explícitamente "no tengo información" en vez de responder con conocimiento general → faithfulness de los "no sé" sube (como se vio en Config B pares con contexto irrelevante)
- **Excluir clarification questions** de la muestra de evaluación: filtrar `needs_clarification=True`
- **Más pares (30)**: reduce varianza estadística aunque no sube el techo estructural
- **Ingestar el split test** (con flag `--include-test`): upper bound real, invalida comparación justa con benchmark pero útil para medir el techo del sistema

---

## Problema conocido: `TimeoutError` en Cerebras

Algunos batches del grupo 2 (answer_relevancy/context_recall/context_precision) tienen
`TimeoutError` ocasional cuando Cerebras tarda más de 120s. Es benigno — Ragas lo
marca como fallo del par pero el resto continúa. Los pares fallidos quedan como -1
y se excluyen del promedio.

Si hay muchos TimeoutError aumentar el timeout en `_run_ragas_group()`:
```python
run_cfg = RunConfig(max_workers=1, max_retries=2, max_wait=15, timeout=180)  # 180 en vez de 120
```

---

*Actualizado: 20 de julio de 2026*
*Ver también: `foragents/eval_setup_critico.md`, `foragents/qa_technical.md` Q23-Q27*
