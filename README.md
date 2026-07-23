# Maternas — Chatbot RAG de Salud Materna

Chatbot conversacional basado en arquitectura RAG orientado a madres gestantes. Clasifica la intención del usuario, evalúa el riesgo clínico y genera respuestas fundamentadas en literatura médica.

> Proyecto de investigación — Convocatoria 890 Minciencias · Institución Universitaria de Envigado

---

## Stack

| Capa | Tecnología |
|---|---|
| Embedding | `intfloat/multilingual-e5-base` (768 dims, ES/EN/ZH) en CUDA |
| Vector store | FAISS `IndexFlatIP` — 375,392 vectores |
| LLM | `llama-3.3-70b-versatile` vía Groq API |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| Bot | Telegram (`python-telegram-bot`) |

## Datasets indexados

- **Multiclinsum** — 25,902 casos clínicos en español
- **MedMCQA** — 187,005 preguntas médicas (EN)
- **MedQA** — US / Taiwan / Mainland + 18 textbooks médicos (EN)

## Estructura

```
src/
├── ingestion/      # formatters, chunkers, embedder, FAISS store, scripts de ingestión
├── classifiers/    # intent_classifier.py, risk_detector.py
├── rag/            # retriever.py, chain.py
├── api/            # main.py (FastAPI), schemas.py
├── ui/             # app.py (Streamlit)
└── settings.py
foragents/          # plan técnico y Q&A del proyecto
```

## Inicio rápido

```bash
# 1. Entorno
python -m venv venv
.\venv\Scripts\activate       # Windows
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers==2.7.0
pip install -r requirements.txt

# 2. Configuración
cp .env.example .env          # completar GROQ_API_KEY y rutas de datasets

# 3. Ingestión (una sola vez, ~5h en GPU)
python src/ingestion/run_ingestion.py

# 4. Arrancar
python -m uvicorn src.api.main:app --port 8080   # Terminal 1
streamlit run src/ui/app.py                       # Terminal 2
python src/bot/maternas_bot.py                    # Terminal 3 (opcional, Telegram bot)
```

UI disponible en `http://localhost:8501` · API docs en `http://localhost:8080/docs` · Bot Telegram: `python src/bot/maternas_bot.py`

## Bot Telegram

El bot permite chatear con Maternas directamente desde Telegram usando polling.

```bash
python src/bot/maternas_bot.py   # Terminal 3 (requiere API ya corriendo)
```

Comandos: `/start` — bienvenida · `/help` — instrucciones · `/reset` — reinicia historial · `/stats` — estadísticas del bot.

Historial conversacional en RAM por usuario. Mensajes separados: header informativo (HTML) + respuesta de Maternas (texto plano) para evitar errores de parseo.

El token se configura en `.env` como `TELEGRAM_BOT_TOKEN`.

## Flujo por turno

```
query → classify_intent() → detect_risk()
              │
              ├─ ¿query vaga? → pregunta de clarificación al usuario
              │
              └─ Búsqueda híbrida:
                   FAISS densa (textbook + medmcqa + medqa)
                 + BM25 léxico  (multiclinsum, solo si hay match exacto)
                       │
                       └─ Groq LLM → respuesta con citas [n]
```

- **Riesgo HIGH** → alerta inmediata + notificación email + respuesta de urgencia
- **Riesgo MEDIUM** → respuesta con recomendación médica (LLM decide si notificar)
- **Riesgo LOW** → respuesta educativa con citas a la fuente

## Retrieval híbrido

El índice tiene 375,392 vectores de tres fuentes con naturaleza distinta. Para no contaminar el contexto con casos clínicos irrelevantes, se usa una estrategia de búsqueda por tipo de fuente:

| Fuente | Estrategia | Motivo |
|---|---|---|
| `textbook`, `medmcqa`, `medqa_*` | FAISS densa (semántica) | Conocimiento estructurado, responde preguntas generales |
| `multiclinsum_*` | BM25 léxico (exacto) | Solo aparece si hay coincidencia real de términos clínicos |

Si no hay coincidencia léxica en Multiclinsum, esos fragmentos no se incluyen — evitando que casos raros de pacientes contaminen la respuesta.

## Preguntas de clarificación

Cuando la query es corta y le falta contexto clínico (semana de gestación, síntoma específico, si está en lactancia), el sistema pide esa información antes de recuperar fragmentos:

- `"me duele la cabeza"` → *"¿Cuántas semanas de embarazo estás actualmente?"*
- `"puedo tomar algo"` → *"¿Para qué síntoma y en qué semana de gestación estás?"*
- `"me siento triste"` → *"¿Cuánto tiempo llevas así y estás embarazada o en postparto?"*

**Nunca** se pide clarificación para `signos_de_alarma` ni cuando el riesgo es medium/high — en esos casos siempre se responde de inmediato.

## Skill System

Arquitectura extensible de herramientas (`tools`) agrupadas en habilidades (`skills`). Cada skill vive en `src/skills/<nombre>/` y expone tools registrables con nombre, descripción, esquema de parámetros y función asociada.

### Notifier (notificaciones por email)

Envío de alertas SMTP cuando se detecta riesgo clínico alto:
- **Risk HIGH** → notificación automática siempre
- **Risk MEDIUM** → una llamada adicional al LLM decide si amerita notificar
- **Risk LOW** → no notifica

Configuración en `.env` con prefijo `NOTIFIER_*`. Por defecto remitente y destinatario son la misma cuenta Google (`maternasrag@gmail.com`).

### Crear una skill nueva

```
src/skills/mi_skill/
├── __init__.py
├── skill.py      # Skill(name, desc, tools=[ToolSpec(...)])
└── tool.py       # Implementación de la función
```

Registrar en `ToolRegistry` y ejecutar desde `chain.py` vía `ToolRegistry.execute("tool_name", ...)`.

## Estructura

```
src/
├── ingestion/      # formatters, chunkers, embedder, FAISS store, scripts de ingestión
├── classifiers/    # intent_classifier.py, risk_detector.py
├── rag/            # retriever.py, bm25_index.py, chain.py
├── api/            # main.py (FastAPI), schemas.py
├── ui/             # app.py (Streamlit)
├── bot/            # maternas_bot.py (Telegram)
├── skills/         # notifier/ (email SMTP), base ToolRegistry
└── settings.py
foragents/          # plan técnico y Q&A del proyecto (Q1–Q21)
```

## Evaluación automática

El sistema se evalúa con el framework **Ragas** sobre el benchmark **MaternaQA-es** (split test, 328 pares QA de obstetricia en español). La evaluación opera en dos fases con modelos independientes:

| Fase | Modelo | Rol |
|---|---|---|
| 1 — Generación | `llama-3.3-70b-versatile` (Groq) | El chatbot RAG genera respuestas reales |
| 2 — Evaluación | `gemma-4-31b` (Cerebras) | Juez externo independiente vía Ragas |

```bash
# Fase 1: generar respuestas
python src/evaluation/eval_pipeline.py --config configC --sample 15 --generate-only

# Fase 2: evaluar con Ragas
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_configC_<TIMESTAMP>.json
```

Ver guía completa en `foragents/eval_runbook.md`.

### Mejores resultados obtenidos — Config C v3

Configuración: FAISS + BM25 híbrido + corpus obstétrico MaternaQA-es LM completo (train+val+test, 5 353 chunks a ~336 tok), evaluado sobre 14 pares sin preguntas de clarificación.

| Métrica | Config C v3 | Baseline MaternaQA-es |
|---|:---:|:---:|
| `faithfulness` | **0.456** | 0.713 |
| `answer_relevancy` | **0.816** | 0.558 |
| `answer_correctness` | **0.532** | — |
| `context_recall` | **0.452** | — |
| `context_precision` | **0.388** | — |
| `latency_avg_s` | ~10.2 s | — |

`answer_relevancy` supera el baseline publicado. La brecha en `faithfulness` se debe principalmente a que cuatro de los cinco datasets indexados son de medicina general (no obstétrica) — incluidos como requisito del proyecto — lo que introduce ruido en el retrieval.

## Siguientes mejoras

- **Reranker cross-encoder local** (`BAAI/bge-reranker-v2-m3`) — recuperar k=20 candidatos y reranquear a top-5 antes del LLM; mejora `context_precision` sin costo de API ni latencia significativa
- **System prompt restrictivo** — instruir al LLM a responder solo con información de los fragmentos recuperados y declarar explícitamente cuando no tiene suficiente contexto; sube `faithfulness` en pares donde el retrieval ya es correcto
- **HyDE (Hypothetical Document Embeddings)** — generar una respuesta hipotética antes de la búsqueda y usarla como query de embedding; mejora `context_recall` en consultas cortas alineando el vocabulario de búsqueda con el del corpus clínico
