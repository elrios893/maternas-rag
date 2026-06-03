# Plan Técnico Completo — Chatbot RAG Maternas

> Generado el 31 de mayo de 2026  
> Estado: Aprobado — listo para implementación  
> Última modificación: estrategia de chunking y embedding actualizada con datos reales verificados (conteos, distribuciones de longitud, estructura real de archivos)

---

## 1. Resumen Ejecutivo

El proyecto **Maternas** consiste en desarrollar un chatbot conversacional basado en arquitectura RAG (Retrieval-Augmented Generation) orientado a madres gestantes. El sistema debe responder preguntas clínicas y educativas en español, clasificar la intención del usuario, evaluar el nivel de riesgo clínico, y fundamentar todas las respuestas en los datasets médicos disponibles, sin fine-tuning del modelo base.

El entregable es un MVP funcional ejecutable en hardware local (RTX 2050 / 16 GB RAM), con exposición vía interfaz Streamlit y API FastAPI, y con costo operativo cercano a cero usando la API gratuita de Groq.

---

## 2. Requisitos Funcionales

| ID | Requisito |
|----|-----------|
| RF-01 | Recibir preguntas en lenguaje natural en español sobre salud materna |
| RF-02 | Clasificar la intención de cada mensaje en una de 12 categorías definidas |
| RF-03 | Evaluar el nivel de riesgo clínico: `low`, `medium`, `high` |
| RF-04 | Recuperar fragmentos relevantes de los datasets mediante RAG |
| RF-05 | Generar respuestas fundamentadas con citas al fragmento fuente |
| RF-06 | Detectar signos de alarma y escalar con acción recomendada |
| RF-07 | Mantener contexto conversacional (historial de turns) |
| RF-08 | Limitar el conocimiento del LLM exclusivamente a los datasets |
| RF-09 | Recomendar siempre consulta médica cuando el riesgo lo amerite |
| RF-10 | Exponer el chatbot como interfaz gráfica interactiva |
| RF-11 | Exponer los clasificadores como endpoints de API REST |
| RF-12 | Evaluar métricas RAG (faithfulness, relevance, answer correctness) |

**Categorías de intención (skill `intent_classification`):**

`control_prenatal` · `signos_de_alarma` · `sintomas_embarazo` · `postparto` · `lactancia` · `salud_mental_perinatal` · `medicamentos` · `nutricion` · `actividad_fisica` · `planificacion_familiar` · `consulta_administrativa` · `pregunta_fuera_de_alcance`

**Niveles de riesgo:**
- **LOW**: pregunta informativa/educativa → `educational_answer`
- **MEDIUM**: requiere recomendación de consulta → `medical_consultation`
- **HIGH**: requiere atención urgente → `urgent_care`

---

## 3. Requisitos No Funcionales

| ID | Requisito | Criterio de Aceptación |
|----|-----------|------------------------|
| RNF-01 | Latencia de respuesta | < 8 segundos por turno en hardware local |
| RNF-02 | Costo operativo | $0 o mínimo (Groq free tier) |
| RNF-03 | Sin fine-tuning / QLoRA | Solo RAG + prompting |
| RNF-04 | Ejecución local | Compatible con RTX 2050 + 16 GB RAM |
| RNF-05 | Idioma principal | Español (con base de conocimiento en inglés) |
| RNF-06 | Escalabilidad mínima | Soportar 1 usuario simultáneo en MVP |
| RNF-07 | Reproducibilidad | Pipeline de ingestión idempotente |
| RNF-08 | Seguridad básica | No exponer datos brutos del paciente |
| RNF-09 | Trazabilidad | Toda respuesta incluye fuente del fragmento recuperado |

---

## 4. Arquitectura General del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                   USUARIO (Streamlit UI)                 │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────┐
│                    FastAPI Backend                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  /chat      │  │ /classify    │  │  /evaluate     │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬───────┘  │
└─────────┼────────────────┼───────────────────┼──────────┘
          │                │                   │
┌─────────▼────────────────▼───────────────────▼──────────┐
│                    Orchestration Layer (LangChain)        │
│                                                          │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ Intent Classifier │    │   Risk Classifier         │   │
│  │ (LLM prompt)     │    │   (LLM prompt + rules)    │   │
│  └─────────┬────────┘    └──────────┬───────────────┘   │
│            │                        │                    │
│  ┌─────────▼────────────────────────▼───────────────┐   │
│  │             RAG Chain (LangChain)                  │   │
│  │  Query → Embed → Retrieve → Rerank → Generate     │   │
│  └─────────────────────┬─────────────────────────────┘   │
└────────────────────────┼────────────────────────────────┘
                         │
         ┌───────────────┴──────────────┐
         │                              │
┌────────▼──────────┐        ┌──────────▼──────────┐
│  FAISS Vector DB  │        │   Groq API (LLM)    │
│  (local disk)     │        │   (llama3/mixtral)  │
└───────────────────┘        └─────────────────────┘
```

---

## 5. Arquitectura RAG

### Flujo de una consulta

```
Input texto (usuario)
        │
        ▼
[1] Preprocesamiento → limpiar, detectar idioma
        │
        ▼
[2] Intent Classifier → categoría + confianza + razón
        │
        ▼
[3] Risk Classifier → nivel riesgo + signos alarma + acción
        │
        ▼
[4] Query Embedding (sentence-transformers local)
        │
        ▼
[5] FAISS Retrieval → top-k fragmentos (k=5)
        │
        ▼
[6] Reranking opcional → score de relevancia
        │
        ▼
[7] Context Assembly → fragmentos + historial conversacional
        │
        ▼
[8] LLM Generation (Groq API) con system_prompt estricto
        │
        ▼
[9] Output estructurado → respuesta + intent + risk + citas
        │
        ▼
[10] Registro para evaluación (RAGAS)
```

### System Prompt base

El system prompt instruirá al LLM a:
- Responder **únicamente** con la información recuperada de los fragmentos del contexto
- Citar la fuente del fragmento (dataset + ID/archivo)
- No inventar información clínica fuera del contexto
- Incluir siempre el disclaimer de consulta médica cuando `risk_level != low`
- Responder siempre en español, aunque la fuente sea en inglés

---

## 6. Pipeline de Ingestión de Datasets

> **Decisión de diseño:** Se utilizan TODOS los registros de todos los datasets sin ningún tipo de filtrado por asignatura, idioma o relevancia. Esta es una restricción del proyecto.

### Conteos reales verificados

| Dataset | Split | Registros reales |
|---------|-------|-----------------|
| MedMCQA | train | 182,822 |
| MedMCQA | dev | 4,183 |
| MedMCQA | test | 6,150 |
| MedQA US | train/dev/test | 12,723 |
| MedQA Taiwan | train/dev/test | 14,123 |
| MedQA Mainland | train/dev/test | 34,251 |
| Multiclinsum fulltext | — | 25,902 |
| Multiclinsum summaries | — | 25,902 |
| Textbooks EN | — | 18 archivos |

### Estrategia por dataset

| Dataset | Volumen | Acción de ingestión |
|---------|---------|---------------------|
| `data/` (MedMCQA) | 193,155 registros JSONL | Formatear como documento priorizando el campo `exp`. Ver sección 8 para detalle del formato. |
| `data_clean/` (MedQA US + Taiwan + Mainland) | 61,097 preguntas + 18 libros TXT | Preguntas de las 3 regiones completas sin chunking. Textbooks: recursive character split. |
| `multiclinsum_large-scale_train_es/` | 25,902 pares TXT | Summaries sin chunking (documentos primarios). Fulltexts con paragraph grouping chunking. |

### Pasos del pipeline

```
1. LOAD       → leer archivos por dataset
2. FORMAT     → construir texto plano unificado por documento
3. CHUNK      → dividir en fragmentos (ver sección 8)
4. METADATA   → asignar metadatos: source_dataset, subject, language, doc_id
5. EMBED      → generar vectores con sentence-transformer
6. INDEX      → agregar a FAISS con doc_id → texto mapping en disco
7. PERSIST    → guardar índice FAISS + metadata store
```

> El pipeline debe ejecutarse por bloques (un dataset a la vez) con checkpoints intermedios para poder retomar en caso de fallo, dado el volumen total (~365k vectores).

---

## 7. Estrategia de Embeddings

### Modelo seleccionado

**`intfloat/multilingual-e5-base`**

### Justificación basada en los datos reales

La exploración de los datasets confirmó tres idiomas activos en el corpus:

| Idioma | Fuente | Evidencia |
|--------|--------|-----------|
| Inglés | MedMCQA, MedQA US, 18 textbooks | Verificado — texto limpio y fluido |
| Español | Multiclinsum (25,902 casos) | Verificado — narrativa clínica nativa |
| Chino tradicional | MedQA Taiwan (14,123 preguntas) | Verificado con Python UTF-8 |
| Chino simplificado | MedQA Mainland (34,251 preguntas) | Verificado con Python UTF-8 |

Las queries del usuario llegan en **español**, pero el 73% del corpus está en **inglés**. El retrieval cross-lingüe ES → EN es el caso más frecuente y crítico. `multilingual-e5-base` proyecta todos los idiomas en el mismo espacio vectorial de 768 dimensiones — es la única clase de modelo que resuelve esto sin traducciones intermedias.

Los registros en chino tendrán naturalmente baja similitud coseno con queries en español sobre salud materna, lo cual es correcto y deseable: el retrieval los dejará atrás y priorizará fragmentos relevantes en inglés y español.

### Parámetros de embedding

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Modelo | `intfloat/multilingual-e5-base` | Mejor balance calidad/velocidad para ES+EN+ZH |
| Dimensión | 768 | Estándar del modelo |
| Batch size ingestión | 64 | Seguro para 4 GB VRAM de RTX 2050 |
| Dispositivo ingestión | CUDA (RTX 2050) | Ingestión offline — aprovechar GPU |
| Dispositivo query-time | CPU | El LLM vía Groq no compite por VRAM local |
| Normalización | L2 obligatoria | Habilita similitud coseno via inner product en FAISS |
| Prefix query | `"query: "` | Protocolo obligatorio del modelo E5 |
| Prefix documento | `"passage: "` | Protocolo obligatorio del modelo E5 |

> **Nota:** El modelo E5 requiere explícitamente los prefijos `"query: "` y `"passage: "`. Omitirlos degrada la calidad del retrieval de forma significativa.

### Alternativa de respaldo

`paraphrase-multilingual-MiniLM-L12-v2` (384 dims) — usar solo si la memoria en ingestión resulta insuficiente. Menor calidad en retrieval cross-lingüe.

---

## 8. Estrategia de Chunking

> Estrategia definida a partir de la inspección real de los datos. Se midieron distribuciones de longitud en todos los datasets antes de tomar las decisiones.

### Hallazgos de la exploración

| Dataset | Métrica | Valor |
|---------|---------|-------|
| MedMCQA — tokens por registro completo | mediana / p95 | 114 / 432 |
| MedMCQA — campo `exp` solo | mediana / p95 | 77 / 382 |
| MedMCQA — registros con `exp` nulo | porcentaje | **11.8%** |
| Multiclinsum summaries | mediana / p95 | 177 / 343 |
| Multiclinsum fulltexts | mediana / p95 | 940 / 1,850 |
| Williams Obstetrics — párrafos naturales (`\n\n`) | total | 20,038 |
| Williams Obstetrics — tokens por párrafo | mediana / p95 | 49 / 223 |

### Estrategia por tipo de fuente

#### 1. MedMCQA y MedQA — todas las regiones y splits

**Sin chunking. 1 registro = 1 documento.**

Todos los registros caben dentro de los 512 tokens del modelo (p95 = 432). La unidad semántica ya está completa.

El insight clave es que el campo `exp` contiene la explicación médica con referencias bibliográficas — es el gold nugget del retrieval. El documento se formatea **priorizando la explicación**, no la pregunta de examen:

```
[EXPLANATION] {exp}
[QUESTION] {question}
[ANSWER] {texto_opcion_correcta}
[SUBJECT] {subject_name}
```

Cuando `exp` es null (11.8% de MedMCQA), se formatea solo con `[QUESTION]` + `[ANSWER]`. El registro no se descarta.

#### 2. Multiclinsum — Summaries

**Sin chunking. 1 summary = 1 documento.**

Mediana de 177 tokens, todos los registros bajo 512 tokens (p95 = 343). Son la destilación semántica del caso clínico — el embedding captura el caso completo en un solo vector de alta densidad.

#### 3. Multiclinsum — Fulltexts

**Paragraph grouping chunking.**

Los fulltexts no pueden indexarse sin chunking (mediana 940 tokens, pico 7,442). Estrategia:

- Dividir por `\n\n` para obtener párrafos naturales
- **Agrupar párrafos consecutivos** hasta alcanzar 350-400 tokens por chunk
- Overlap de **1 párrafo completo** entre chunks contiguos — preserva el hilo diagnóstico que frecuentemente se extiende entre párrafos
- Agregar `doc_id` en metadata para poder recuperar el summary asociado al mismo caso

Se usa 350-400 tokens (no 512) porque los párrafos clínicos en español son semánticamente densos. Un chunk de 512 tokens abarcaría secciones muy distintas del caso (motivo + diagnóstico + tratamiento + evolución), diluyendo el vector.

#### 4. Textbooks EN (18 archivos)

**Recursive Character Splitter con ventana deslizante.**

Los párrafos naturales tienen mediana de solo **49 tokens** — demasiado cortos para un embedding significativo. No se puede usar paragraph splitter simple.

- Chunk size: **400 tokens**
- Overlap: **80 tokens** (20%)
- Separadores en orden de prioridad: `["\n\n", "\n", ". ", " "]`

El overlap es del 20% (mayor al estándar del 12%) porque con párrafos tan cortos, sin overlap suficiente se pierde el contexto del argumento médico que se construye a través de varios párrafos.

### Tabla resumen

| Fuente | Estrategia | Chunk size | Overlap | Docs estimados |
|--------|-----------|-----------|---------|----------------|
| MedMCQA (todos) | Sin chunking — formato exp-first | ~114 tokens mediana | — | 193,155 |
| MedQA US/TW/ML (todos) | Sin chunking | ~80-200 tokens | — | 61,097 |
| Multiclinsum summaries | Sin chunking | ~177 tokens mediana | — | 25,902 |
| Multiclinsum fulltexts | Paragraph grouping | 350-400 tokens | 1 párrafo | ~90,000 est. |
| Textbooks EN (18) | Recursive char split | 400 tokens | 80 tokens | ~32,000 est. |
| **TOTAL** | | | | **~402,000 vectores** |

### Metadatos de cada chunk

```python
{
  "chunk_id": "uuid4",
  "source_dataset": "medmcqa | medqa_us | medqa_tw | medqa_ml | multiclinsum_summary | multiclinsum_fulltext | textbook",
  "doc_id": "id_original_o_nombre_archivo",
  "subject": "string — asignatura o nombre del textbook",
  "language": "en | es | zh-hant | zh-hans",
  "chunk_index": 0,
  "text": "texto del chunk con prefijo 'passage: ' al momento de embeddear"
}
```

---

## 9. Diseño de la Vector Store FAISS

### Configuración

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Tipo de índice | `IndexFlatIP` (inner product) | Exacto, adecuado para hasta ~500k vectores |
| Dimensión | 768 | Dimensión del embedding |
| Métrica | Similitud coseno (vectores L2-normalizados) | Estándar para búsqueda semántica |
| Top-k retrieval | 5 | Balance contexto/tokens enviados al LLM |
| Persistencia | `faiss.write_index()` a disco | Evitar re-indexar en cada arranque |

### Estimación de volumen (conteos reales verificados)

| Dataset | Registros reales | Vectores estimados | Estrategia |
|---------|-----------------|-------------------|------------|
| MedMCQA completo | 193,155 | 193,155 | Sin chunking |
| MedQA US | 12,723 | 12,723 | Sin chunking |
| MedQA Taiwan | 14,123 | 14,123 | Sin chunking |
| MedQA Mainland | 34,251 | 34,251 | Sin chunking |
| Multiclinsum summaries | 25,902 | 25,902 | Sin chunking |
| Multiclinsum fulltexts | 25,902 | ~90,000 est. | Paragraph grouping |
| Textbooks EN (18 libros) | 18 archivos | ~32,000 est. | Recursive char split |
| **TOTAL** | | **~402,000 aprox.** | |

~402k vectores × 768 dims × 4 bytes ≈ **~1.2 GB en RAM** — manejable con 16 GB, monitorear uso.

### Estructura en disco

```
faiss_store/
├── index.faiss          ← índice binario FAISS
├── metadata.pkl         ← dict {chunk_id → metadata + texto}
└── build_info.json      ← versión, fecha, parámetros de construcción
```

---

## 10. Diseño del Clasificador de Intención

### Enfoque: LLM-based zero-shot con structured output

No se entrena un clasificador clásico. Se usa el LLM de Groq con un prompt estructurado que fuerza la salida JSON.

### Input / Output

```python
# Input
user_query: str
conversation_history: list[dict]  # últimos N turnos

# Output
{
  "intent": "signos_de_alarma",
  "confidence": 0.92,
  "reason": "La pregunta menciona cefalea intensa durante el embarazo."
}
```

### Prompt del clasificador de intención

```
Eres un clasificador de intención para un chatbot de salud materna.
Analiza el mensaje del usuario y devuelve ÚNICAMENTE un JSON con:
- "intent": una de las 12 categorías listadas abajo
- "confidence": float entre 0.0 y 1.0
- "reason": explicación breve en español de por qué se asignó esa categoría

Categorías válidas:
control_prenatal, signos_de_alarma, sintomas_embarazo, postparto,
lactancia, salud_mental_perinatal, medicamentos, nutricion,
actividad_fisica, planificacion_familiar, consulta_administrativa,
pregunta_fuera_de_alcance

Mensaje del usuario: {query}
```

### Fallback

Si `confidence < 0.6` o el LLM falla el parse JSON → `intent = "pregunta_fuera_de_alcance"`.

---

## 11. Diseño del Clasificador de Riesgo

### Enfoque: Híbrido (reglas deterministicas + LLM)

**Capa 1 — Reglas deterministicas** (rápido, sin API, red de seguridad crítica):

Detectar keywords de alarma alta en el mensaje del usuario:
- `sangrado`, `convulsión`, `pérdida de conciencia`, `no se mueve el bebé`
- `dolor de cabeza fuerte`, `visión borrosa`, `dificultad para respirar`
- `fiebre y flujo con mal olor`, `hacerme daño`, `desmayé`
- `bebé no se mueve`, `movimientos fetales`, `presión muy alta`

Si hay match → `risk_level = high`, `requires_escalation = true`, `recommended_action = urgent_care` de forma inmediata, sin invocar LLM.

**Capa 2 — LLM assessment** (para casos no cubiertos por Capa 1):

Prompt estructurado que devuelve JSON con evaluación de riesgo.

### Output

```python
{
  "risk_level": "low | medium | high",
  "requires_escalation": bool,
  "detected_alert_signs": ["sangrado vaginal", "cefalea intensa"],
  "recommended_action": "educational_answer | medical_consultation | urgent_care"
}
```

### Matriz de decisión intent × riesgo mínimo

| Intent | Riesgo mínimo | Lógica adicional |
|--------|--------------|-----------------|
| `signos_de_alarma` | `medium` | Keywords Capa 1 → `high` |
| `salud_mental_perinatal` + ideación | `high` | Siempre `urgent_care` |
| `medicamentos` | `medium` | Siempre `medical_consultation` |
| `control_prenatal` sin controles previos | `medium` | — |
| `consulta_administrativa` | `low` | — |
| `nutricion` / `lactancia` informativa | `low` | Pérdida de peso severa → `medium` |
| `postparto` con fiebre/flujo | `high` | Capa 1 detecta keywords |

---

## 12. Diseño del Chatbot Conversacional

### Gestión de contexto

- Historial de conversación: últimos **6 turnos** (3 user + 3 assistant)
- Session ID: UUID por sesión de Streamlit
- Memoria: `LangChain ConversationBufferWindowMemory` (k=6)

### Flujo por turno

```
1.  Recibir user_message
2.  Ejecutar Capa 1 Risk (reglas) → si HIGH inmediato, ir a paso 5
3.  Ejecutar Intent Classifier (LLM) → intent_result
4.  Ejecutar Capa 2 Risk (LLM) → risk_result
5.  Si risk_level == "high":
    → Emitir respuesta de emergencia (template fijo) de forma prioritaria
6.  Construir query aumentada = user_message + intent + contexto previo
7.  Retrieve k=5 fragmentos desde FAISS
8.  Construir prompt final = system_prompt + fragments + historial + query
9.  Llamar Groq API → respuesta en español
10. Post-procesar: agregar citas, disclaimer médico si aplica
11. Devolver: respuesta + intent_result + risk_result + sources
12. Loguear turno completo para evaluación
```

### Template de respuesta de emergencia (HIGH risk)

```
⚠️ ATENCIÓN: Tu consulta indica una posible situación de urgencia médica.
Signos detectados: {detected_alert_signs}

Por favor, ACUDE INMEDIATAMENTE a urgencias o llama a servicios de emergencia.
No esperes más información de este chatbot en este momento.

──────────────────────────────────────────
Información de apoyo basada en los datasets:
{rag_context_si_aplica}
```

---

## 13. Diseño de la API (FastAPI)

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/chat` | Turno conversacional principal |
| `POST` | `/classify/intent` | Solo clasificador de intención |
| `POST` | `/classify/risk` | Solo clasificador de riesgo |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Métricas agregadas de la sesión |

### Schema `/chat` — Request

```json
{
  "session_id": "uuid",
  "message": "string",
  "history": [
    { "role": "user", "content": "string" },
    { "role": "assistant", "content": "string" }
  ]
}
```

### Schema `/chat` — Response

```json
{
  "answer": "string",
  "intent": {
    "intent": "control_prenatal",
    "confidence": 0.92,
    "reason": "string"
  },
  "risk": {
    "risk_level": "low | medium | high",
    "requires_escalation": false,
    "detected_alert_signs": [],
    "recommended_action": "educational_answer | medical_consultation | urgent_care"
  },
  "sources": [
    {
      "chunk_id": "string",
      "source_dataset": "medmcqa | medqa | multiclinsum",
      "text_preview": "string"
    }
  ],
  "latency_ms": 0
}
```

---

## 14. Diseño de la Interfaz (Streamlit)

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  Chatbot Maternas                         [Nueva sesión]  │
├───────────────────────────┬──────────────────────────────┤
│                           │  Panel de análisis            │
│   Ventana de chat         │  ┌──────────────────────────┐ │
│                           │  │ Intención detectada       │ │
│  [Mensajes usuario]       │  │ control_prenatal          │ │
│  [Respuestas bot]         │  │ Confianza: 92%            │ │
│  [Badges de riesgo]       │  └──────────────────────────┘ │
│                           │  ┌──────────────────────────┐ │
│                           │  │ Nivel de riesgo           │ │
│                           │  │ MEDIO                     │ │
│                           │  │ Acción: consulta médica   │ │
│   [Input de texto]        │  └──────────────────────────┘ │
│   [Botón Enviar]          │  ┌──────────────────────────┐ │
│                           │  │ Fuentes recuperadas       │ │
│                           │  │ 1. multiclinsum_ls_es_... │ │
│                           │  │ 2. MedMCQA id: ...        │ │
│                           │  └──────────────────────────┘ │
└───────────────────────────┴──────────────────────────────┘
```

### Características UI

- Badge de color por nivel de riesgo: 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH
- Secciones expandibles con fragmentos fuente recuperados
- Disclaimer médico automático en respuestas MEDIUM y HIGH
- Banner de emergencia persistente y destacado para HIGH risk
- Opción de exportar historial de sesión completo en JSON

---

## 15. Métricas de Evaluación

### Métricas RAG (framework RAGAS)

| Métrica | Descripción | Target MVP |
|---------|-------------|------------|
| **Faithfulness** | Respuesta fundamentada en el contexto recuperado | ≥ 0.75 |
| **Answer Relevancy** | Respuesta relevante a la pregunta del usuario | ≥ 0.70 |
| **Context Recall** | Fragmentos recuperados cubren la respuesta esperada | ≥ 0.65 |
| **Context Precision** | Fragmentos relevantes sobre total recuperado | ≥ 0.65 |

### Métricas de clasificación (sobre los 24 test cases)

| Métrica | Descripción | Target MVP |
|---------|-------------|------------|
| **Intent Accuracy** | Intención correcta / total | ≥ 0.85 |
| **Risk Level Accuracy** | Nivel de riesgo correcto / total | ≥ 0.90 |
| **Escalation Recall** | HIGH/MEDIUM correctamente escalados | **1.0** (crítico, no negociable) |
| **False Alarm Rate** | LOW clasificados incorrectamente como HIGH | ≤ 0.05 |

### Evaluación clínica manual — Informe de Triage

Se generará un informe ejecutando los 24 test cases definidos en `test_cases.md` más prompts adicionales cubriendo los 10 temas requeridos:

| Tema | Prompts mínimos |
|------|----------------|
| Control prenatal | 2 (TC-001, TC-002) |
| Síntomas durante el embarazo | 3 (TC-003, TC-004, TC-005) |
| Signos de alarma | 3+ (TC-006, TC-007, TC-008) |
| Postparto | 2 (TC-009, TC-010) |
| Lactancia | 2 (TC-011, TC-012) |
| Medicamentos | 2 (TC-013, TC-014) |
| Salud mental | 2 (TC-015, TC-016) |
| Nutrición | 2 (TC-017, TC-018) |
| Urgencia / emergencia | 2 (TC-019, TC-020) |
| Pregunta administrativa / educativa | 4 (TC-021 a TC-024) |

Cada respuesta evaluada por: correctitud clínica · presencia de disclaimer · cita de fuente · manejo correcto de urgencia.

---

## 16. Riesgos Técnicos

| ID | Riesgo | Probabilidad | Impacto | Mitigación |
|----|--------|-------------|---------|------------|
| R-01 | VRAM insuficiente para embedding + inferencia simultáneos | Alta | Alto | Embedding en CPU o batches offline; LLM vía API no consume VRAM local |
| R-02 | Tiempo de ingestión de ~365k documentos muy alto | Alta | Medio | Pipeline por bloques con checkpoints; ingestión es one-time |
| R-03 | Groq API rate limits en free tier (~30 RPM) | Alta | Medio | Cachear respuestas frecuentes; manejar `429` con retry + backoff |
| R-04 | Mismatch semántico idioma query (ES) vs corpus (EN) | Alta | Alto | Multilingual-E5 resuelve este problema nativamente |
| R-05 | LLM genera información médica fuera del contexto recuperado | Media | Crítico | System prompt estricto + faithfulness ≥ 0.75 como umbral |
| R-06 | Clasificador de riesgo falla en detectar HIGH (falso negativo) | Baja | Crítico | Capa 1 de reglas deterministicas como red de seguridad inamovible |
| R-07 | RAM insuficiente para índice FAISS completo en memoria | Media | Alto | ~1.1 GB estimados, dentro de 16 GB disponibles; monitorear |
| R-08 | Latencia total > 10 segundos por turno | Media | Medio | Groq es muy rápido; reducir k, truncar historial, cachear embeddings |
| R-09 | Multilingüe chino (Taiwan/Mainland) reduce precisión en queries ES | Media | Bajo | E5 multilingüe maneja esto; el retrieval priorizará texto relevante |

---

## 17. MVP Mínimo Funcional

El MVP debe demostrar el flujo completo end-to-end con los 24 test cases aprobados.

### Componentes mínimos del MVP

**M1 — Pipeline de ingestión** *(offline, one-time)*
- Ingestión completa de Multiclinsum summaries + fulltexts (español)
- Ingestión completa de MedMCQA (~192k registros)
- Ingestión completa de MedQA (US + Taiwan + Mainland + 18 textbooks)
- FAISS index persistido en disco con metadata store

**M2 — Backend FastAPI**
- Endpoint `/chat` funcional end-to-end
- Clasificador de intención (LLM prompt + JSON parse)
- Clasificador de riesgo (reglas Capa 1 + LLM Capa 2)
- RAG chain con LangChain + FAISS + Groq

**M3 — Interfaz Streamlit**
- Chat conversacional con historial
- Panel lateral: intención + riesgo + fuentes recuperadas
- Banner de emergencia para HIGH risk

**M4 — Evaluación básica**
- Script que ejecuta los 24 test cases automáticamente
- Reporte de accuracy de intent + risk vs. valores esperados
- Informe de triage: 10 temas × N prompts con evaluación manual

---

## 18. Backlog Priorizado

| # | Tarea | Prioridad | Depende de |
|---|-------|-----------|------------|
| 1 | Configurar entorno Python (venv, requirements.txt) | P0 | — |
| 2 | Explorar y verificar estructura local de los 3 datasets | P0 | — |
| 3 | Script de ingestión: Multiclinsum (summaries + fulltexts) → FAISS | P0 | 1, 2 |
| 4 | Script de ingestión: MedMCQA completo → FAISS | P0 | 1, 2 |
| 5 | Script de ingestión: MedQA completo (3 regiones + 18 textbooks) → FAISS | P0 | 1, 2 |
| 6 | Configurar Groq API key + test básico de llamada | P0 | 1 |
| 7 | Implementar clasificador de intención (prompt + JSON parse + fallback) | P1 | 6 |
| 8 | Implementar clasificador de riesgo Capa 1 (reglas) | P1 | — |
| 9 | Implementar clasificador de riesgo Capa 2 (LLM) | P1 | 6 |
| 10 | Implementar RAG chain básica (embed query → retrieve → generate) | P1 | 3, 4, 5, 6 |
| 11 | Integrar intent + risk + RAG en flujo único de un turno | P1 | 7, 8, 9, 10 |
| 12 | Implementar gestión de historial conversacional (k=6 turns) | P1 | 11 |
| 13 | Implementar sistema de respuesta de emergencia para HIGH risk | P1 | 8, 9 |
| 14 | Exponer flujo completo como API FastAPI `/chat` | P2 | 11, 12, 13 |
| 15 | Construir interfaz Streamlit con panel de análisis | P2 | 14 |
| 16 | Script de evaluación automática: 24 test cases | P2 | 14 |
| 17 | Generación de informe de triage (10 temas × N prompts) | P2 | 15, 16 |
| 18 | Integrar métricas RAGAS (faithfulness, relevancy, recall, precision) | P3 | 16 |
| 19 | Explorar opciones de despliegue (HF Spaces / ngrok / Railway) | P3 | 15 |
| 20 | Documentar decisiones técnicas y guía de ejecución | P3 | 17 |

---

## Notas de Implementación Críticas

### Modelo LLM (Groq)
Opciones recomendadas en Groq free tier:
- `llama-3.1-70b-versatile` — mejor calidad, mayor latencia
- `mixtral-8x7b-32768` — contexto largo (32k), balance calidad/velocidad
- `llama3-8b-8192` — más rápido, menor calidad, útil para clasificadores

### Modelo de Embedding
`intfloat/multilingual-e5-base` — corre en CPU sin problema para ingestión offline. Para queries en tiempo real, también en CPU dado que el LLM vía Groq no compite por VRAM local.

### Restricción de seguridad inamovible
La **Capa 1 del clasificador de riesgo** (reglas deterministicas con keywords) **NUNCA debe omitirse ni ser bypasseada**. Es la red de seguridad para casos de alto riesgo que el LLM podría clasificar incorrectamente o donde la latencia de API podría costar tiempo crítico.

### Gestión de costos Groq
- Clasificador de intención: ~1 llamada por turno
- Clasificador de riesgo Capa 2: ~1 llamada por turno (solo si Capa 1 no determina HIGH)
- Generación RAG: ~1 llamada por turno
- Total: máximo 3 llamadas por turno de usuario
- Implementar caché LRU para queries idénticas o muy similares

---

*Fin del plan técnico — Chatbot RAG Maternas*
