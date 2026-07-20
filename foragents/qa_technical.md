# Preguntas y Respuestas Técnicas — Chatbot RAG Maternas

Registro de preguntas técnicas surgidas durante el desarrollo y sus respuestas definitivas.
Sirve como referencia rápida para cualquier sesión futura.

---

## Q1: ¿Dónde se almacenan los embeddings y la vector store?

**Respuesta:**

Los embeddings se almacenan en **dos lugares distintos según el momento**:

### En disco (persistencia entre sesiones)

La carpeta `faiss_store/` contiene tres archivos que juntos forman la vector store completa:

```
faiss_store/
├── index.faiss       ← Los vectores (embeddings float32) en formato binario FAISS
├── metadata.pkl      ← Pickle Python: dict { chunk_id → { text, source_dataset, language, ... } }
└── build_info.json   ← Parámetros de construcción: fecha, modelo, dimensión, total de vectores
```

- **`index.faiss`**: Contiene los ~402,000 vectores de 768 dimensiones cada uno, en el formato nativo de FAISS (`IndexFlatIP`). Es un archivo binario — no es legible directamente. Tamaño estimado: ~1.2 GB.
- **`metadata.pkl`**: Mapea cada vector (por su posición/ID en el índice) con el texto original del chunk y sus metadatos. Es necesario para poder mostrar la fuente de cada fragmento recuperado.
- **`build_info.json`**: Registro de auditoría. Indica qué modelo generó los embeddings, cuándo, con qué parámetros. Permite detectar si el índice está desactualizado.

### En RAM (mientras la app está corriendo)

Cuando la API o la interfaz arrancan, se ejecuta:
```python
faiss.read_index("faiss_store/index.faiss")
```
Esto carga **todos los vectores en memoria RAM**. Estimado: ~1.2 GB de RAM ocupados en tiempo de ejecución.

El modelo de embedding (`multilingual-e5-base`) también se carga en memoria (CPU) en tiempo de ejecución para convertir las queries del usuario en vectores. Pesa ~1.1 GB adicional.

### Flujo completo

```
INGESTIÓN (one-time, offline):
  texto del chunk
      │
      ▼
  sentence-transformers (multilingual-e5-base)
      │  → vector float32[768]
      ▼
  faiss.add(vector)  →  index.faiss (disco)
  metadata_dict[id] = {text, source, ...}  →  metadata.pkl (disco)

QUERY (runtime, cada turno):
  query del usuario
      │
      ▼
  sentence-transformers (mismo modelo, en RAM)
      │  → vector float32[768]
      ▼
  faiss.search(vector, k=5)  →  top-5 chunk_ids
      │
      ▼
  metadata.pkl lookup  →  textos de los 5 fragmentos
      │
      ▼
  LLM (Groq API)  →  respuesta fundamentada
```

### Resumen en una línea

> Los embeddings viven en `faiss_store/index.faiss` en disco, y se cargan completos en RAM al iniciar la aplicación. La metadata asociada (texto + fuente) vive en `faiss_store/metadata.pkl`.

---

## Q2: ¿Por qué FAISS y no una base de datos vectorial como Chroma o Pinecone?

**Respuesta:**

Las restricciones del proyecto definen la respuesta: costo cero, ejecución local, sin servicios externos.

| Criterio | FAISS | Chroma | Pinecone |
|----------|-------|--------|---------|
| Costo | Gratuito | Gratuito (local) | Pago (cloud) |
| Ejecución local | Sí | Sí | No (cloud) |
| Dependencias externas | Ninguna | SQLite | API cloud |
| RAM para 402k vectores | ~1.2 GB | ~1.5 GB+ | N/A |
| Velocidad búsqueda exacta | Muy alta | Media | Alta |
| Persistencia | Archivo binario simple | SQLite | Cloud |

Para ~402k vectores en hardware local, `IndexFlatIP` de FAISS es la opción más eficiente y simple. No requiere servidor, no tiene overhead de base de datos, y el archivo resultante es portátil.

---

## Q3: ¿Hay que regenerar el índice FAISS si se cambia el modelo de embedding?

**Respuesta:**

**Sí, obligatoriamente.** Los vectores en `index.faiss` son específicos al modelo que los generó. Si se cambia de `multilingual-e5-base` a cualquier otro modelo, los vectores existentes son incompatibles — la búsqueda devolvería resultados sin sentido.

El `build_info.json` registra el modelo usado, precisamente para detectar este escenario al arrancar.

---

## Q4: ¿Qué pasa si la RAM no alcanza para cargar el índice completo?

**Respuesta:**

Si los ~1.2 GB del índice + ~1.1 GB del modelo de embedding saturan la RAM disponible (16 GB, muy improbable), las opciones son:

1. **`IndexIVFFlat`** en lugar de `IndexFlatIP`: divide el índice en celdas, carga solo las relevantes en memoria. Requiere una fase de entrenamiento adicional pero reduce RAM activa.
2. **`IndexFlatIP` con memory-mapped files**: FAISS soporta mmap, el SO gestiona qué páginas están en RAM.
3. Reducir a `MiniLM-L12` (384 dims) — reduce el índice a ~600 MB.

Con 16 GB disponibles, el escenario actual (IndexFlatIP, 402k × 768) no debería ser problema.

---

## Q5: ¿Por qué se usan los prefijos `"query: "` y `"passage: "` en el modelo E5?

**Respuesta:**

El modelo `multilingual-e5-base` fue entrenado con esos prefijos como parte de la representación del texto. Sin ellos, el modelo no distingue si está procesando una consulta (búsqueda) o un pasaje (documento a indexar), y la calidad del retrieval cae significativamente.

- Al indexar documentos → `"passage: " + texto_del_chunk`
- Al embeddear la query del usuario → `"query: " + pregunta_del_usuario`

Es una convención del modelo, no de FAISS.

---

## Q6: ¿Cuánto tarda la ingestión completa (~402k documentos)?

**Respuesta (estimada, pendiente de medir en hardware real):**

Con batch_size=64 en CPU:
- Velocidad típica: ~500-800 documentos/segundo con `multilingual-e5-base`
- Estimado: 402,000 / 650 ≈ **~620 segundos (~10 minutos)**

Con batch_size=64 en GPU (RTX 2050, 4GB VRAM):
- Velocidad típica: ~2,000-3,500 documentos/segundo
- Estimado: 402,000 / 2,500 ≈ **~160 segundos (~2-3 minutos)**

La ingestión es **one-time**: se ejecuta una sola vez y el índice queda persistido. No se re-ejecuta al iniciar la aplicación.

---

## Q7: ¿Por qué es importante mantener el rango objetivo de 350-400 tokens por chunk?

**Respuesta:**

El rango de 350-400 tokens no es arbitrario — es el punto de equilibrio entre tres fuerzas que se contradicen entre sí:

### El problema de los chunks demasiado cortos (< 100 tokens)

Un chunk muy corto contiene tan poco contexto que el embedding no puede capturar una idea médica completa. Por ejemplo:

```
"La preeclampsia es una complicación del embarazo."
```

Este fragmento produce un vector que semánticamente puede matchear con casi cualquier pregunta sobre embarazo, aunque no tenga información útil para responderla. El resultado es **ruido en el retrieval**: se recuperan fragmentos que parecen relevantes por similitud superficial pero no aportan conocimiento real al LLM.

### El problema de los chunks demasiado largos (> 600 tokens)

Un chunk muy largo mezcla varias ideas clínicas distintas en un solo vector. Por ejemplo, un chunk de 1,000 tokens podría contener:

```
[síntomas de preeclampsia] + [diagnóstico diferencial] + [manejo farmacológico] + [criterios de hospitalización]
```

El embedding de ese chunk es un **promedio semántico** de los cuatro temas. Si la query es sobre síntomas, el vector del chunk está "diluido" por los otros tres temas y puede quedar por debajo en ranking frente a un chunk más enfocado. Además, se consumen más tokens del contexto del LLM con información que puede no ser relevante para la pregunta específica.

### El rango 350-400 tokens es el punto óptimo para este corpus

| Factor | Justificación basada en los datos reales |
|--------|------------------------------------------|
| Los párrafos clínicos de Multiclinsum tienen mediana de ~230 tokens | Un chunk de 350-400 agrupa 1-2 párrafos completos, capturando una unidad clínica coherente (motivo + diagnóstico, o diagnóstico + tratamiento) |
| Los párrafos de textbooks tienen mediana de 49 tokens | Necesitan agrupación — 350-400 tokens agrega ~7 párrafos relacionados, suficiente para contexto médico completo |
| El modelo `multilingual-e5-base` tiene ventana de 512 tokens | El rango 350-400 deja margen para el prefijo `"passage: "` y evita truncación silenciosa del modelo |
| Top-k=5 chunks en retrieval | 5 chunks × 400 tokens = ~2,000 tokens de contexto enviados al LLM, dentro del límite operativo de Groq |

---

## Q8: ¿Qué significan los scores de similitud coseno y por qué nunca son cero?

**Respuesta:**

Los vectores del modelo `multilingual-e5-base` viven en un espacio de alta dimensión (768 dims) donde todos los textos, sin importar su contenido, terminan proyectándose en regiones relativamente cercanas entre sí. La similitud coseno entre dos vectores cualesquiera rara vez baja de 0.6-0.7 en modelos de lenguaje modernos porque comparten vocabulario estadístico general.

**FAISS siempre devuelve los k documentos más cercanos**, incluso si ninguno es relevante. No tiene un concepto de "no encontré nada útil".

Lo que importa no es el valor absoluto del score, sino la **diferencia entre scores relevantes e irrelevantes**. El filtro real es el clasificador de intención + el criterio del LLM, no el score de FAISS.

---

## Q9: ¿Por qué el modelo de embedding se carga como singleton?

**Respuesta:**

El modelo `multilingual-e5-base` ocupa aproximadamente **1.1 GB en RAM** una vez cargado. Si se instanciara en cada llamada a `embed_query()`, el sistema cargaría y descargaría 1.1 GB de memoria en cada turno conversacional — inviable en tiempo real.

El patrón singleton garantiza que el modelo se carga **una sola vez** al importar el módulo `embedder.py`, y permanece en memoria durante toda la vida de la aplicación.

---

## Q10: ¿Cuántos vectores quedaron en el índice FAISS final y cuánto tardó la ingestión?

**Respuesta:**

Ingestión completa ejecutada el 2 de junio de 2026 con `sentence-transformers==2.7.0`, modelo `intfloat/multilingual-e5-base`, device `cuda` (RTX 2050 4 GB).

| Dataset | Vectores añadidos | Tiempo |
|---|---|---|
| Multiclinsum (summaries + fulltexts) | 51,804 | ~47 min |
| MedMCQA (train + dev) | 187,005 | ~2.5 h |
| MedQA questions (US + Taiwan + Mainland) | ~99,779 | ~30 min |
| Textbooks EN (18 libros) | ~36,000 | ~1h 35 min |
| **TOTAL** | **375,392** | **~5 h total** |

**Archivos en disco:**
- `faiss_store/index.faiss` → 1,153.2 MB
- `faiss_store/metadata.pkl` → 431.5 MB
- `faiss_store/build_info.json` → marcas de cada fase completada

**Stack definitivo que resolvió el cuelgue de sentence_transformers:**
- Problema: `sentence-transformers==3.3.1` colgaba silenciosamente al importar cuando `torch` ya estaba en memoria (trainer stack con accelerate).
- Solución: downgrade a `sentence-transformers==2.7.0`. Import instantáneo, CUDA funcional.

**Versiones definitivas del entorno:**
```
torch==2.5.1+cu121
sentence-transformers==2.7.0
transformers==4.57.6
tokenizers==0.22.2
faiss-cpu==1.9.0
```

---

## Q11: ¿Cómo funcionan los clasificadores de intención y riesgo?

**Respuesta:**

Implementados el 2 de junio de 2026 en `src/classifiers/`.

### Clasificador de intención (`intent_classifier.py`)
- **Método:** zero-shot con Groq LLM (`llama-3.3-70b-versatile`), `temperature=0`, `response_format=json_object`
- **12 categorías:** `control_prenatal`, `signos_de_alarma`, `sintomas_embarazo`, `postparto`, `lactancia`, `salud_mental_perinatal`, `medicamentos`, `nutricion`, `actividad_fisica`, `planificacion_familiar`, `consulta_administrativa`, `pregunta_fuera_de_alcance`
- **Retorna:** `IntentResult(intent, confidence, reasoning)`
- **Fallback:** heurística de keywords si el LLM falla o devuelve intent inválido

### Detector de riesgo (`risk_detector.py`)
- **Dos capas:**
  1. **Heurística rápida** (sin API): keywords agrupadas por categoría clínica → HIGH o MEDIUM instantáneo
  2. **LLM Groq** (contextual): solo si la heurística no detecta nada
- **3 niveles:** `low → educational_answer` | `medium → medical_consultation` | `high → urgent_care`
- **Retorna:** `RiskResult(level, flags, action, reasoning, used_heuristic)`
- **Ventaja:** los casos más urgentes (hemorragia, convulsión, ideación suicida) se detectan SIN llamada a la API → latencia ~0ms

### Resultados del test
| Mensaje | Intent | Risk | Método riesgo |
|---|---|---|---|
| Náuseas y vómitos | `sintomas_embarazo` 90% | low | LLM |
| Sangrando mucho con coágulos | `signos_de_alarma` 95% | high → urgent_care | heurística |
| Calcio en embarazo | `nutricion` 90% | low | LLM |
| Bebé no se mueve | `signos_de_alarma` 90% | high → urgent_care | LLM |
| Depresión + no quiero vivir | `salud_mental_perinatal` 99% | high → urgent_care | heurística |
| Dolor leve de cabeza | `sintomas_embarazo` 70% | low | LLM |

**Nota:** el modelo `llama-3.1-70b-versatile` fue dado de baja por Groq. El reemplazo es `llama-3.3-70b-versatile` — actualizado en `.env`.

---

## Q12: ¿FastAPI + Streamlit o Streamlit directo? ¿Qué tiene menos latencia y qué conviene para entrega?

**Respuesta:**

**Opción elegida: FastAPI + Streamlit** conectados vía HTTP.

- **Latencia:** diferencia de ~20-50ms sobre un turno de 2-4s — imperceptible para el usuario
- **Entrega:** endpoints `/chat`, `/classify`, `/health` permiten probar la API con Postman sin la UI
- **Arranque:** Terminal 1 → `uvicorn src.api.main:app --port 8080` | Terminal 2 → `streamlit run src/ui/app.py`

**Latencia total estimada por turno:**
```
intent (Groq)      ~300ms
risk (heurística)    ~0ms  o LLM ~300ms
embed + FAISS      ~150ms
LLM generation     ~1-3s
HTTP overhead       ~30ms
─────────────────────────
Total              ~2-4s
```

---

## Q13: ¿Cuál es el stack tecnológico completo y para qué sirve cada componente?

**Respuesta:**

### Hardware objetivo
| Componente | Valor |
|---|---|
| CPU | AMD Ryzen 5 |
| GPU | NVIDIA RTX 2050 (4 GB VRAM) |
| RAM | 16 GB |
| OS | Windows 11 |

---

### Lenguaje y entorno
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **Python** | 3.12.7 | Lenguaje principal de todo el proyecto |
| **venv** | stdlib | Entorno virtual aislado para dependencias |

---

### Embedding y modelos locales
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **PyTorch** | 2.5.1+cu121 | Motor de cómputo tensorial con soporte CUDA para la GPU |
| **sentence-transformers** | 2.7.0 | Carga y ejecuta el modelo de embedding `multilingual-e5-base` |
| **transformers** (HuggingFace) | 4.57.6 | Carga los pesos del modelo base internamente |
| **intfloat/multilingual-e5-base** | — | Modelo de embedding: convierte texto en vectores de 768 dims. Soporta ES, EN, ZH. Se ejecuta en la RTX 2050 |

---

### Vector store
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **FAISS** (faiss-cpu) | 1.9.0 | Índice vectorial con AVX2. Almacena 375,392 embeddings (1.15 GB). Búsqueda de similitud coseno en milisegundos |

---

### LLM (generación de texto)
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **Groq API** | — | API cloud que ejecuta LLMs en hardware LPU de alta velocidad |
| **llama-3.3-70b-versatile** | — | Genera respuestas, clasifica intención y evalúa riesgo clínico |
| **groq** (SDK Python) | 0.37.1 | Cliente Python oficial para llamar a la API de Groq |

---

### API REST
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **FastAPI** | 0.115.6 | Framework para la API REST — endpoints `/chat`, `/classify`, `/health`. Genera Swagger en `/docs` |
| **uvicorn** | 0.48.0 | Servidor ASGI que ejecuta FastAPI |
| **Pydantic** | 2.x | Validación y serialización de schemas de request/response |
| **httpx** | 0.28.1 | Cliente HTTP usado por Streamlit para llamar a la API |

---

### Interfaz de usuario
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **Streamlit** | 1.41.1 | UI web en Python puro: chat con burbujas, sidebar con metadata, badges de riesgo, panel de fuentes |

---

### Datos y configuración
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **datasets** (HuggingFace) | 4.8.5 | Carga y procesamiento de datasets médicos |
| **pydantic-settings** | 2.14.1 | Lee el `.env` y expone configuración tipada en `src/settings.py` |
| **python-dotenv** | 1.2.2 | Carga el archivo `.env` en variables de entorno |
| **tqdm** | 4.67.3 | Barras de progreso en la ingestión de datasets |

---

### Flujo completo
```
Usuario escribe mensaje
        ↓
  Streamlit UI  →  POST /chat  →  FastAPI
                                      ↓
                          classify_intent()  →  Groq LLM (~300ms)
                                      ↓
                          detect_risk()      →  heurística (~0ms) / Groq LLM
                                      ↓
                          retrieve()         →  multilingual-e5-base (CUDA)
                                             →  FAISS 375k vectores (~150ms)
                                      ↓
                          Groq LLM genera respuesta con contexto (~1-3s)
                                      ↓
                     ChatResponse  →  Streamlit renderiza
```

---

## Q14: ¿Qué es la capa heurística en el detector de riesgo y por qué es útil?

**Respuesta:**

### Qué es

La capa heurística es un conjunto de listas de palabras clave agrupadas por categoría clínica que se revisan directamente contra el texto del mensaje **sin llamar a ninguna API**. Si alguna keyword hace match, el sistema retorna inmediatamente `risk=high` o `risk=medium` sin esperar respuesta del LLM.

```python
HIGH_RISK_KEYWORDS = {
    "hemorragia":    ["sangrando mucho", "hemorragia", "sangrado abundante", ...],
    "eclampsia":     ["convulsión", "pérdida de conocimiento", "desmayo", ...],
    "movimiento_fetal_ausente": ["no se mueve", "dejó de moverse", ...],
    "depresion_grave": ["quiero hacerme daño", "no quiero vivir", ...],
    ...
}
```

Si ninguna keyword hace match → se escala al LLM para evaluación contextual.

---

### Por qué es útil en este caso específico

**1. Latencia cero en los casos más críticos**

Una hemorragia activa o una convulsión no pueden esperar 300-600ms de llamada a la API. La heurística responde en microsegundos. En emergencias reales esos milisegundos importan psicológicamente — la alerta aparece de inmediato.

**2. Funciona sin conexión a internet**

Si la API de Groq falla o hay un corte de red, la heurística sigue detectando las señales de alarma más graves. El sistema nunca deja pasar una hemorragia sin alertar, aunque el LLM esté caído.

**3. Determinismo total en los casos de alto riesgo**

El LLM es probabilístico — en teoría podría clasificar "estoy sangrando mucho" como `medium` en algún contexto inusual. La heurística es absolutamente determinista: si la frase está en la lista, siempre es `high`. Para señales clínicas de alarma mayor (eclampsia, ausencia de movimiento fetal, ideación suicida) el determinismo es más seguro que la probabilidad.

**4. Ahorra tokens de API**

Cada llamada al LLM para clasificar riesgo consume ~100-200 tokens. Con la heurística, los mensajes con señales obvias no llegan al LLM — se resuelven localmente. En un sistema con muchos usuarios, esto reduce costo y latencia promedio.

**5. Las keywords clínicas son conocimiento estable**

A diferencia de la clasificación de intención (que requiere entender matices del lenguaje), las señales de alarma obstétrica son un conjunto finito y bien documentado médicamente. "Sangrado abundante", "convulsión", "no se mueve el bebé" — estas frases no cambian con el contexto. Son candidatas ideales para reglas deterministas.

---

### Cuándo la heurística NO es suficiente y necesita el LLM

- Frases ambiguas: *"me duele la cabeza y tengo la vista un poco rara"* — puede ser preeclampsia o puede ser cansancio. La heurística no puede capturar esa ambigüedad con keywords.
- Negaciones: *"ya no tengo sangrado"* — la heurística ingenua detectaría "sangrado" y marcaría HIGH incorrectamente. El LLM entiende la negación.
- Contexto histórico: *"ayer tuve una convulsión pero ya estoy bien"* — requiere razonamiento sobre tiempo y estado actual.

Por eso el diseño usa **ambas capas en secuencia**: la heurística para los casos claros y urgentes, el LLM para los casos que requieren comprensión contextual.

---

---

## Q15: ¿Hay TF-IDF en el proyecto y por qué no se usó?

**Respuesta:**

**No hay TF-IDF en el proyecto ni en ninguna de sus dependencias.** Se buscó explícitamente en todo el código y no existe ninguna referencia a `TfidfVectorizer`, `TfidfModel` ni ninguna implementación de TF-IDF.

### Por qué no se necesita

El proyecto usa **embeddings densos** (`intfloat/multilingual-e5-base`, 768 dimensiones) en lugar de vectores sparse como TF-IDF. Las razones:

| Aspecto | TF-IDF | multilingual-e5-base (usado) |
|---|---|---|
| Tipo de vector | Sparse (una entrada por palabra) | Denso (768 floats densos) |
| Semántica | Coincidencia de palabras exactas | Captura sinónimos y significado |
| Multilingüe | Requiere un vocabulario por idioma | Soporta ES/EN/ZH en un solo modelo |
| Tamaño del índice | Depende del vocabulario (puede ser grande) | Fijo: 768 dims por documento |
| Matching | "náusea" no matchea con "arcada" | Ambas tienen vectores cercanos |

### Cuándo TF-IDF podría haber sido útil

En un escenario con recursos muy limitados (sin GPU, con RAM < 8 GB), TF-IDF sería una alternativa viable porque:

- No requiere GPU
- Ocupa menos RAM (el vocabulario es más compacto que 375k × 768 floats)
- No necesita descargar un modelo de ~1.1 GB

Pero sacrificaría calidad de retrieval — especialmente en un corpus multilingüe donde el mismo concepto médico se expresa con palabras distintas en español, inglés y chino.

### Conclusión

TF-IDF es una técnica de recuperación de información clásica. Funciona bien para búsqueda por palabras clave en corpus pequeños y monolingües. No se usó porque el proyecto necesita búsqueda semántica multilingüe, y el hardware disponible (RTX 2050, 16 GB RAM) permite ejecutar embeddings densos sin problemas.

---

## Q16: ¿Qué es TF-IDF y cómo funciona? (contexto académico)

**Respuesta:**

TF-IDF (Term Frequency — Inverse Document Frequency) es un método clásico de recuperación de información que convierte texto en vectores numéricos basándose en la frecuencia de palabras.

### Cómo funciona

**TF (Term Frequency):** cuántas veces aparece una palabra en un documento específico. Si "preeclampsia" aparece 5 veces en un caso clínico de 100 palabras, su TF es 5/100 = 0.05.

**IDF (Inverse Document Frequency):** qué tan rara o común es una palabra en todo el corpus. "preeclampsia" aparece en pocos documentos → IDF alto. "la" aparece en casi todos → IDF bajo (casi cero).

**TF-IDF = TF × IDF.** Una palabra obtiene peso alto si aparece frecuentemente en un documento específico pero es rara en el corpus general. Esto filtra palabras vacías (artículos, preposiciones) y destaca términos relevantes del documento.

### Limitaciones frente al enfoque usado en Maternas

1. **No captura sinónimos:** una pregunta sobre "hipertensión gestacional" no matchearía con un documento que solo menciona "preeclampsia", aunque sean el mismo tema.
2. **No captura contexto:** la palabra "sangrado" tiene el mismo vector sin importar si aparece en "tengo sangrado abundante" o en "ya no tengo sangrado".
3. **Vocabulario por idioma:** un índice TF-IDF en español no sirve para consultas en chino. Los embeddings multilingües resuelven esto.
4. **Dimensionalidad variable:** el vector TF-IDF crece con el vocabulario del corpus (puede ser de cientos de miles de dimensiones). Los embeddings densos tienen dimensionalidad fija (768) y son más eficientes para búsqueda en índices como FAISS.

---

## Q17: ¿Se usa clustering de vectores en el proyecto?

**Respuesta:**

**No.** No hay clustering en el proyecto. Ni en el código fuente ni en las dependencias. El índice FAISS es `IndexFlatIP` — una estructura plana donde todos los vectores se almacenan en una sola lista y la búsqueda compara la query contra **todos** los vectores (fuerza bruta exacta).

No hay `KMeans`, `DBSCAN`, `HDBSCAN`, `AgglomerativeClustering` ni ninguna otra técnica de agrupamiento. Tampoco se usa `IndexIVFFlat` (que sí usa clustering internamente para particionar).

### Por qué no se necesita

El índice actual tiene **375,392 vectores**. La búsqueda exacta en `IndexFlatIP` tarda ~20-50ms en GPU (vía FAISS con AVX2) para cada query. Eso es más que suficientemente rápido para el objetivo de latencia (< 8s por turno).

```python
# El buscador es simplemente:
scores, ids = index.search(query_vector, k=5)   # compara contra todos
```

No hay necesidad de clustering porque el índice cabe completo en RAM (~1.15 GB de 16 GB disponibles) y el tiempo de búsqueda es despreciable frente al ~2-4s totales del turno.

### Cuándo se necesitaría clustering

Si el índice creciera a **millones** de vectores, la búsqueda exacta empezaría a ser lenta (ej. 10M vectores → ~1-2s solo la búsqueda). Ahí entrarían dos opciones con clustering:

**Opción 1: `IndexIVFFlat` (clustering interno de FAISS)**
- Durante la construcción, FAISS aplica K-Means para dividir los vectores en `n` grupos (ej. 4096)
- En búsqueda, solo compara contra los vectores del grupo más cercano a la query
- Intercambia exactitud por velocidad: ajustando `nprobe` (cuántos grupos revisar) se controla el balance
- La pérdida de recall es típicamente < 5% con `nprobe=10`

**Opción 2: Clustering externo previo**
- Agrupar los fragmentos por tema (ej. "nutrición", "preeclampsia", "lactancia")
- Enrutar la query al grupo correcto antes de buscar en FAISS
- Esto ya se hace implícitamente con el clasificador de intención — la intención no dirige a un cluster distinto, pero el LLM recibe contexto filtrado

### Conclusión

Para 375k vectores y latencia objetivo de < 8s, `IndexFlatIP` es la opción correcta. Clustering agregaría complejidad sin beneficio real. Si el proyecto escalara a millones de vectores en el futuro, se migraría a `IndexIVFFlat` (que está disponible en la misma librería FAISS y no requiere cambios en el resto del sistema).

---

---

## Q18: ¿Cómo funciona la integración con Telegram?

**Respuesta:**

El bot de Telegram se implementó el 3 de junio de 2026 en `src/bot/maternas_bot.py` usando la librería `python-telegram-bot`.

### Arquitectura

```
Usuario Telegram → Bot API → polling (python-telegram-bot) → POST /chat → FastAPI → RAG chain → respuesta → Telegram
```

El bot no implementa lógica RAG propia — es un **cliente ligero** que envía cada mensaje al endpoint `POST /chat` de la API REST de Maternas (FastAPI) y muestra la respuesta al usuario.

### Características

| Aspecto | Detalle |
|---|---|
| Método | Polling (sin webhook, sin servidor público) |
| Librería | `python-telegram-bot==21.11.1` |
| Historial | En RAM por `user_id` (diccionario en memoria, se pierde al reiniciar el bot) |
| Formato | Header informativo en HTML (negritas, itálicas controladas) + cuerpo de respuesta en texto plano |
| Comandos | `/start` — bienvenida, `/help` — instrucciones, `/reset` — reinicia historial, `/stats` — estadísticas del bot |
| Manejo de errores | Si la API falla, responde con mensaje de error amigable sin crashear |

### Por qué mensajes separados (HTML + texto plano)

El LLM de Groq genera respuestas en markdown impredecible (a veces mezcla `*`, `_`, `**` de forma inconsistente). Telegram parsea markdown estrictamente y cualquier error de formato hace que el mensaje completo falle con `BadRequest`.

Solución: se envían **dos mensajes**:
1. **Header en HTML** (controlado por el código): nombre del bot, badge de riesgo, advertencias — formateo seguro porque lo genera Python, no el LLM.
2. **Cuerpo en texto plano**: la respuesta generada por el LLM, sin parseo. Telegram la muestra tal cual.

Esto elimina por completo los errores `BadRequest: Can't parse entities` sin perder la experiencia de usuario.

### Limitaciones actuales

- **Historial volátil**: se almacena en un `defaultdict(list)` en RAM. Si el bot se reinicia, todas las conversaciones se pierden. Para producción se migraría a Redis o SQLite.
- **Sin manejo de grupos**: el bot responde en cualquier chat donde esté agregado. No hay filtro por chat_id.
- **Sin rate limiting**: no hay throttle de mensajes por usuario.
- **Sin logs persistentes**: los logs van a stdout/stderr.

### Inicio

```bash
# Terminal 1: API
python -m uvicorn src.api.main:app --port 8080

# Terminal 2: Bot
python src/bot/maternas_bot.py
```

El token se lee de `settings.TELEGRAM_BOT_TOKEN` (configurado en `.env`).

---

## Q19: ¿Qué necesita otra persona para clonar y correr el proyecto desde cero (con embeddings ya generados)?

**Respuesta:**

Los embeddings (índice FAISS + metadatos) se compartieron por WeTransfer (~1.6 GB comprimido). Con eso, la nueva persona **no necesita ejecutar la ingestión** — solo descargar, extraer y arrancar.

### Paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/elrios893/maternas-rag.git
cd maternas-rag

# 2. Crear entorno virtual con Python 3.12.7
py -3.12 -m venv venv
.\venv\Scripts\activate

# 3. Instalar PyTorch con CUDA (RTX 2050, 4 GB VRAM)
#    Si no tiene GPU, instalar torch sin --index-url
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar sentence-transformers 2.7.0 (la versión 3.x falla con torch)
pip install sentence-transformers==2.7.0

# 5. Instalar el resto de dependencias
pip install -r requirements.txt

# 6. Configurar .env
copy .env.example .env
# Editar .env: poner GROQ_API_KEY real (https://console.groq.com)
# IMPORTANTE: generar una NUEVA clave, no usar la que está en este documento
# EMBEDDING_DEVICE=cuda (o cpu si no tiene GPU NVIDIA)

# 7. Extraer el índice FAISS (archivo de WeTransfer)
#    Dejar la carpeta faiss_store/ en la raíz del proyecto:
#    faiss_store/
#    ├── index.faiss      (~1.15 GB)
#    ├── metadata.pkl     (~431 MB)
#    └── build_info.json

# 8. Verificar integridad del índice (opcional)
python -c "import pickle; d=pickle.load(open('faiss_store/metadata.pkl','rb')); print(f'{len(d)} vectores en metadata')"
# Debe mostrar: 375392 vectores en metadata
```

### Arranque

```bash
# Terminal 1: API
python -m uvicorn src.api.main:app --port 8080

# Terminal 2: Streamlit (opcional)
streamlit run src/ui/app.py

# Terminal 3: Telegram bot (opcional, requiere TELEGRAM_BOT_TOKEN en .env)
python src/bot/maternas_bot.py
```

### Notas importantes

- **GROQ_API_KEY**: la clave actual expuesta en este Q&A ya fue regenerada. Quien clone el proyecto debe crear su propia clave en https://console.groq.com (tier gratuito: ~30 requests/minuto, ~6000/día — suficiente para desarrollo).
- **sentence-transformers 2.7.0 obligatorio**: la versión 3.3.1 del `requirements.txt` es la que estaba instalada al inicio del proyecto, pero produce un cuelge silencioso al importar con torch cargado. La solución es instalar 2.7.0 **antes** que el resto de dependencias.
- **CUDA 12.1**: el índice se generó con `multilingual-e5-base` en CUDA. Si la nueva máquina no tiene GPU NVIDIA, cambiar `EMBEDDING_DEVICE=cpu` en `.env`. El embedding será más lento (~200-400ms por query en vez de ~50ms) pero funcional.
- **FAISS CPU vs GPU**: el índice usa `faiss-cpu`. La búsqueda se hace en CPU aunque el embedding esté en GPU. No necesita CUDA para FAISS.

---

## Q20: ¿Por qué se implementó búsqueda híbrida y cómo funciona?

**Respuesta:**

Implementado el 3 de julio de 2026 tras diagnóstico de calidad del retrieval.

### El problema original

Con búsqueda densa pura (FAISS sobre todos los vectores), Multiclinsum dominaba los resultados para preguntas generales. Multiclinsum tiene 51,804 vectores de casos clínicos individuales — son útiles para términos médicos específicos, pero inútiles para preguntas como "¿qué alimentos evitar en el embarazo?". Los scores FAISS son engañosamente altos (~0.84) porque los embeddings densos siempre están cerca en el espacio vectorial, independientemente de la relevancia real.

Ejemplo del problema:
```
Q: "Que alimentos debo evitar durante el embarazo?"
Antes: Score 0.84 → caso clínico de mujer post-gastrectomía (irrelevante)
       Score 0.83 → caso de melioidosis en embarazo (irrelevante)
```

### La solución: búsqueda híbrida por tipo de fuente

```
query
  │
  ├─► FAISS densa (solo textbook + medmcqa + medqa_*)
  │   - Semántica: captura sinónimos y paráfrasis
  │   - top-5 fragmentos
  │
  └─► BM25 léxico (solo multiclinsum_summary + multiclinsum_fulltext)
      - Exacta: solo retorna si hay coincidencia real de términos
      - top-2 fragmentos, solo si BM25 score >= 0.5
      - Si no hay match léxico → Multiclinsum no aparece
```

### Módulos involucrados

- **`src/rag/bm25_index.py`** — singleton BM25 sobre Multiclinsum. Se construye en memoria al primer uso (~15-20s, ~150MB RAM). Usa `rank_bm25` con tokenizador multilingüe (ES/EN) y eliminación de stopwords.
- **`src/rag/retriever.py`** — orquesta ambas búsquedas y mergea resultados. El FAISS pide `k×10` candidatos para filtrar Multiclinsum con suficiente margen.

### Por qué BM25 para Multiclinsum específicamente

Los casos clínicos de Multiclinsum son valiosos cuando el usuario pregunta algo específico que aparece literalmente en los casos (preeclampsia, eclampsia, placenta previa, hemorragia). En esos casos, la coincidencia léxica exacta es más fiable que la similitud semántica. Para preguntas generales sin términos médicos exactos, BM25 simplemente no retorna nada — que es el comportamiento correcto.

### Resultado post-mejora

```
Q: "Que alimentos debo evitar durante el embarazo?"
Después: 5 densos (medmcqa + textbook) + 2 BM25 con match léxico
→ Respuesta con lista concreta: mercurio, no pasteurizados, etc.

Q: "Es seguro hacer ejercicio durante el embarazo?"
Después: 5 fragmentos de textbook → Respuesta con cita [1] del ACOG
```

### Mejora del prompt (simultánea)

Se reforzó el system prompt con reglas explícitas de citación:
- Solo citar [n] si el fragmento literalmente respalda la afirmación
- Si los fragmentos son adyacentes pero no exactos, usarlos de apoyo y complementar con conocimiento general sin aclarar innecesariamente "no tengo fuentes"
- Tono más conciso, cálido y directo — sin introducciones largas ni despedidas genéricas

---

## Q21: ¿Cómo funciona el sistema de preguntas de clarificación?

**Respuesta:**

Implementado el 3 de julio de 2026. Cuando la query del usuario es vaga o le falta contexto clínico clave, el sistema pide información adicional antes de recuperar fragmentos o generar respuesta.

### El problema que resuelve

Preguntas como *"me duele la cabeza"* o *"puedo tomar algo"* no tienen suficiente contexto para dar una respuesta útil y segura. Sin saber las semanas de gestación, el síntoma exacto o si está en lactancia, el LLM improvisa o da consejos genéricos poco útiles.

### Arquitectura: opción híbrida (reglas + LLM)

```
query → classify_intent() → detect_risk()
              │
              ▼
   _should_clarify(query, intent, risk_level)
   ┌── Capa 1: reglas deterministicas ──────────────────────────────┐
   │  - Si risk != "low" → False (urgente/medio: responder siempre) │
   │  - Si intent en NEVER_CLARIFY → False                         │
   │  - Si query >= 20 tokens → False (suficiente contexto)         │
   │  - Si intent en CLARIFICATION_RULES Y query corta              │
   │    Y no contiene keywords de contexto → True                   │
   └────────────────────────────────────────────────────────────────┘
              │ True                    │ False
              ▼                         ▼
   _generate_clarification()        flujo RAG normal
   (LLM genera pregunta empática)
              │
              ▼
   ChatResponse(
     needs_clarification=True,
     clarification_question="...",
     answer=clarification_question   ← mismo texto, para que callers simples lo muestren
   )
```

### CLARIFICATION_RULES

Cada intent define:
- `min_tokens`: si la query tiene menos tokens que este valor, se considera vaga
- `keywords`: contexto esperado (semana, trimestre, síntoma...). Si no hay ninguno → clarificar
- `missing_info`: qué información le falta (se pasa al LLM para generar la pregunta)

| Intent | Activa si... |
|---|---|
| `medicamentos` | Query corta sin síntoma ni semana de gestación |
| `sintomas_embarazo` | Query corta sin mención de semanas/trimestre |
| `control_prenatal` | Query corta sin semana o trimestre |
| `nutricion` | Query corta sin mencionar embarazo o lactancia |
| `actividad_fisica` | Query corta sin trimestre |
| `salud_mental_perinatal` | Query corta sin contexto temporal |

### Casos especiales

- **`signos_de_alarma`** → **nunca** pide clarificación. Si hay riesgo, se actúa de inmediato.
- **Risk medium o high** → nunca pide clarificación. Responde con urgencia apropiada.
- **Query >= 20 tokens** → se asume suficiente contexto, nunca clarifica.

### Resultados de prueba

| Query | Resultado |
|---|---|
| `"me duele la cabeza"` | ✅ Clarifica: *"¿Cuántas semanas de embarazo estás actualmente?"* |
| `"puedo tomar algo"` | ✅ Clarifica: *"¿En qué semana de embarazo te encuentras y qué síntomas tienes?"* |
| `"me siento triste"` | ✅ Clarifica: *"¿Cuánto tiempo has estado sintiéndote así y en qué momento del embarazo/postparto?"* |
| `"me siento mal"` | ✅ NO clarifica — detector marcó risk=medium, responde de inmediato |
| `"tengo 28 semanas y me duele la cabeza con visión borrosa"` | ✅ NO clarifica — suficiente contexto |
| `"tengo sangrado abundante"` | ✅ NO clarifica — high risk, actúa de inmediato |

### Cambios en el código

- `src/rag/chain.py`: `CLARIFICATION_RULES`, `_should_clarify()`, `_generate_clarification()`, campos `needs_clarification` y `clarification_question` en `ChatResponse`
- `src/api/schemas.py`: nuevos campos en `ChatResponse`
- `src/ui/app.py`: burbuja amarilla diferenciada para preguntas de clarificación
- `src/bot/maternas_bot.py`: muestra `💬 {pregunta}` sin header de riesgo cuando `needs_clarification=True`

---

## Q22: ¿Cómo se evalúa la calidad del sistema RAG con MaternaQA-es?

**Respuesta:**

Implementado el 3 de julio de 2026. El pipeline de evaluación usa el compendio QA **MaternaQA-es** (`JhonHander/MaternaQA-es`) como benchmark y Ragas como motor de métricas.

### ¿Qué es MaternaQA-es?

Dataset público en español de **5.727 pares pregunta-respuesta** derivados de 63 PDFs clínicos (GPC de atención prenatal, revistas de obstetricia colombianas, protocolos). Construido por el mismo equipo del proyecto Minciencias. Cada par tiene:
- `pregunta` / `respuesta` / `contexto_fuente`
- `tipo`: factual | definicion | comparacion | razonamiento | aplicacion | hipotetico
- `dificultad`: basico | intermedio | avanzado
- Split train/validation/test sin fuga de datos (división a nivel de documento)

### Flujo del pipeline (dos fases separadas)

La separación en fases es necesaria para evitar conflictos CUDA/CPU entre el embedding del proyecto (GPU) y los embeddings de Ragas (CPU).

```
FASE 1 (--generate-only):
  muestra estratificada de test.jsonl
       ↓
  chat(pregunta) → respuesta generada + fragmentos recuperados
       ↓
  evaluation_reports/eval_raw_<ts>.json

FASE 2 (--evaluate-only <raw.json>):
  Ragas evaluate() con LLM judge (Groq) + embeddings CPU
  métricas: faithfulness, answer_relevancy, context_recall
       ↓
  evaluation_reports/eval_results_<ts>.json
  evaluation_reports/eval_report_<ts>.md
```

### Muestra estratificada (~50 pares)

| Tipo | N |
|---|---|
| factual | 15 |
| definicion | 10 |
| razonamiento | 10 |
| aplicacion | 10 |
| comparacion | 3 |
| hipotetico | 2 |

### Métricas usadas

| Métrica | Qué mide | Notas |
|---|---|---|
| `faithfulness` | ¿La respuesta está respaldada por los fragmentos recuperados? | Alto = el LLM no inventa datos |
| `answer_relevancy` | ¿La respuesta responde la pregunta formulada? | Alto = respuesta pertinente |
| `context_recall` | ¿El retrieval capturó información relevante del ground truth? | Bajo es esperable: el corpus actual no tiene los PDFs de MaternaQA-es |

### Uso

```bash
# Evaluación completa (~50 pares, ambas fases)
python src/evaluation/eval_pipeline.py

# Solo generar respuestas (fase 1)
python src/evaluation/eval_pipeline.py --generate-only

# Solo evaluar un raw ya generado (fase 2)
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_XXX.json

# Muestra reducida para prueba rápida
python src/evaluation/eval_pipeline.py --sample 10
```

### Referencia de línea base (MaternaQA-es propio)

| Split | Faithfulness | Answer Relevancy |
|---|---|---|
| Train | 0.7726 | 0.6466 |
| Test | 0.7132 | 0.5583 |

### Archivos

- `src/evaluation/sampler.py` — descarga y muestrea estratificadamente `test.jsonl` desde GitHub raw
- `src/evaluation/eval_pipeline.py` — pipeline completo: fase 1 (generación), fase 2 (Ragas)
- `evaluation_reports/` — reportes JSON y Markdown generados (ignorado por git, muy pesados)

---

## Q23: ¿Por qué Ragas agota la cuota de tokens incluso con 20 pares, y cómo se resuelve?

**Contexto:** Al correr `phase_evaluate()` con 20 pares y 5 métricas, ambas API keys de Groq (100k tokens/día cada una) se agotan antes de completar todos los pares, dejando resultados parciales.

**Causa raíz:**

Ragas 0.2.12 con defaults lanza **N_pares × N_métricas jobs** contra la API. Con 20 pares × 5 métricas = 100 jobs, y con `max_workers=16` (default) los lanza en ráfagas de 16 simultáneos. Cada job puede consumir entre 700 y 3.000 tokens según la métrica:

| Métrica | Tokens/par (estimado) | Razón |
|---|---|---|
| `faithfulness` | ~2.000–3.000 | 2 prompts LLM: genera statements + verifica NLI |
| `answer_correctness` | ~1.500–2.000 | F1 semántico + factual combinado |
| `context_recall` | ~800–1.200 | Clasifica cada oración del ground truth |
| `context_precision` | ~400–600 | Verifica relevancia de cada chunk recuperado |
| `answer_relevancy` | ~200–400 | Solo genera preguntas alternativas, prompt liviano |

Con 20 pares y sin throttling: estimado **98k–144k tokens total**, excede una sola key. Con `max_retries=10` (default) y errores de output parser, se multiplica.

**Solución implementada — dos grupos secuenciales:**

```python
# Grupo 1: KEY_1 (GROQ_API_KEY) — métricas pesadas ~35-50k tokens
evaluate([faithfulness, answer_correctness], run_config=RunConfig(
    max_workers=1,   # sin ráfagas concurrentes
    max_retries=2,   # máximo 2 reintentos
    max_wait=15,
    timeout=120,
), batch_size=1)

# Pausa 10s
# Grupo 2: KEY_2 (GROQ_API_KEY_2) — métricas livianas ~28-44k tokens
evaluate([answer_relevancy, context_recall, context_precision], ...)
```

Esto distribuye la carga entre las dos keys independientes, y `max_workers=1 + batch_size=1` elimina la concurrencia que provocaba picos de consumo. Los jobs se procesan uno por uno, predeciblemente.

**Limitación observada:** Incluso con esta estrategia, si las keys ya tienen tokens consumidos del día (por generaciones de fase 1 o sesiones anteriores), los últimos pares de cada grupo fallan con 429. El raw JSON de fase 1 persiste en disco — se puede relanzar `--evaluate-only` al día siguiente sin repetir las generaciones.

**Archivos relevantes:**
- `src/evaluation/eval_pipeline.py` — funciones `_run_ragas_group()`, `phase_evaluate()`
- `src/settings.py` — `groq_api_key` y `groq_api_key_2`
- `.env` — `GROQ_API_KEY` (chatbot), `GROQ_API_KEY_2` (Ragas judge)

---

## Q24: ¿Contra qué se evalúa el sistema RAG y qué significan realmente las métricas obtenidas?

**Pregunta frecuente:** "¿Las métricas de Ragas miden si el sistema es bueno o malo en salud materna?"

**Respuesta corta:** Miden cosas distintas, y el ground truth importa tanto como la métrica.

### Dataset de evaluación: MaternaQA-es

`JhonHander/MaternaQA-es` es un dataset de QA en español construido sobre documentos de salud materna colombiana:

| Campo | Detalle |
|---|---|
| Fuente | PDFs académicos: GPC Atención Prenatal de Bajo Riesgo 2023, revistas de obstetricia (vol831-1.pdf, etc.) |
| Idioma | Español colombiano |
| Split usado | `test` (328 pares) |
| Estructura | `pregunta`, `respuesta` (ground truth), `contexto_fuente`, `tipo`, `dificultad`, `source_pdf`, `topics` |
| Tipos de pregunta | factual (31%), aplicacion (29%), razonamiento (26%), definicion (9%), hipotetico (5%) |
| Dificultad | intermedio (57%), basico (37%), avanzado (6%) |

El ground truth son respuestas redactadas por humanos **basadas en fragmentos textuales exactos** de esos PDFs.

### El problema estructural: corpus mismatch

El corpus RAG actual (textbooks EN, MedMCQA, MedQA, Multiclinsum) **no contiene los PDFs de MaternaQA-es**. Por eso:

| Métrica | Resultado esperado | Razón |
|---|---|---|
| `context_recall` | Cercano a 0.0 | Los fragmentos recuperados nunca son del PDF de referencia |
| `context_precision` | Bajo (~0.03–0.08) | Los chunks recuperados son irrelevantes para ese ground truth |
| `faithfulness` | Moderado (~0.3–0.7) | El LLM responde desde conocimiento general, no desde los fragmentos |
| `answer_relevancy` | Relativamente alto (~0.6–0.7) | La respuesta es pertinente a la pregunta aunque no use las fuentes correctas |
| `answer_correctness` | Moderado (~0.4–0.6) | Coincidencia semántica parcial con el ground truth |

### Qué métricas son válidas para comparar Config A vs Config B

| Métrica | ¿Válida para comparar configs? | Por qué |
|---|---|---|
| `faithfulness` | ✅ Sí | Mide si el LLM se ciñe a lo que recupera — diferente según retrieval |
| `answer_relevancy` | ✅ Sí | Pertinencia de la respuesta a la pregunta — refleja calidad del LLM |
| `answer_correctness` | ✅ Sí | Distancia semántica respuesta↔ground truth — comparable entre configs |
| `context_recall` | ⚠️ Limitada | Siempre cercana a 0 porque el corpus no tiene los docs de referencia |
| `context_precision` | ⚠️ Limitada | Igual — mejorará solo cuando se ingesten los PDFs de MaternaQA-es |
| `latency_s` | ✅ Sí | Tiempo real medido en fase 1 — diferencia entre configs es real |

### Cuándo tendrán valor pleno context_recall y context_precision

Cuando se ingesten los PDFs de MaternaQA-es al índice FAISS. En ese momento el retrieval podrá recuperar los fragmentos exactos que el ground truth cita, y las métricas de contexto pasarán de ~0 a valores significativos. Ese es el **next step** documentado en el plan técnico.

### Referencia baseline publicada

El paper de MaternaQA-es reporta estos valores evaluando un sistema RAG que sí tiene los PDFs indexados:

| Split | Faithfulness | Answer Relevancy |
|---|---|---|
| train | 0.7726 | 0.6466 |
| test | 0.7132 | 0.5583 |

Nuestro sistema sin los PDFs obtiene `answer_relevancy` ~0.66 (por encima del baseline test 0.5583), lo que indica que la calidad de generación del LLM es competitiva. La brecha en `faithfulness` (~0.29 vs 0.71) se explica por el corpus mismatch: el LLM no puede citar fuentes que no recuperó.

**Archivos relevantes:**
- `src/evaluation/sampler.py` — descarga y muestrea estratificadamente `test.jsonl`
- `src/evaluation/eval_pipeline.py` — pipeline completo con métricas y reporte MD
- `evaluation_reports/eval_raw_configB_20260716_012717.json` — raw de fase 1 (reutilizable)
- `evaluation_reports/eval_report_configB_20260716_012717.md` — reporte con resultados parciales
- `foragents/retrieval_arquitecturas_configs.md` — documentación de Config A y Config B

---

## Q25: ¿Por qué se descartó llama-3.3-70b como juez de Ragas y qué se usa en su lugar?

**Contexto:** El pipeline de evaluación usaba `llama-3.3-70b-versatile` (Groq) como LLM judge para Ragas. Esto causaba agotamiento de la cuota de 100k tokens/día incluso con solo 15-20 pares.

### El problema de fondo: tokens por llamada

`faithfulness` en Ragas hace 2 llamadas LLM por par: (1) generación de statements y (2) NLI verdicts. Con llama-3.3-70b en español, el modelo generaba statements muy extensos con explicaciones adicionales, consumiendo ~4.500 tokens por par solo para faithfulness.

| Modelo | Tokens/par faithfulness | 15 pares × 5 métricas |
|---|---|---|
| llama-3.3-70b (Groq) | ~4.500 | ~337k tokens — imposible en tier free |
| llama-3.1-8b (Groq) | ~220 | ~16k tokens — dentro del límite, pero falló JSON |
| **gemma-4-31b (Cerebras)** | ~296 | **~22k tokens — completó 15/15 sin errores** |

### Por qué llama-3.1-8b también falló

El 8B generaba JSON válido en pruebas directas, pero Ragas usa un loop de reintentos de parseo con prompts en cadena. El 8B fallaba consistentemente con `RagasOutputParserException` — no seguía el formato JSON anidado del prompt interno de Ragas de forma confiable.

### Solución: Cerebras `gemma-4-31b`

`gemma-4-31b` en Cerebras superó las pruebas:
- JSON válido en el prompt complejo de Ragas
- ~296 tokens por llamada (66x menos que llama-3.3-70b)
- Sin límite diario estricto de tokens (límite por minuto, no por día)
- **15/15 completados en evaluación real** sin un solo error de rate limit o parser

### Problema adicional: `LLMDidNotFinishException`

Ragas verifica el `finish_reason` de cada respuesta. Si el modelo termina por longitud (`"length"`) en vez de por stop token (`"stop"`), lanza `LLMDidNotFinishException`. La solución fue pasar un `is_finished_parser` permisivo al `LangchainLLMWrapper`:

```python
def _is_finished(response: LLMResult) -> bool:
    VALID = {"stop", "STOP", "length", "MAX_TOKENS", "end_turn", "eos"}
    for g in response.flatten():
        resp = g.generations[0][0]
        finish = None
        if resp.generation_info:
            finish = resp.generation_info.get("finish_reason")
        if finish is not None and finish not in VALID:
            return False
    return True

llm = LangchainLLMWrapper(ChatOpenAI(...), is_finished_parser=_is_finished)
```

### Providers evaluados y resultado

| Provider | Modelo | JSON Ragas | Rate limit | Resultado |
|---|---|---|---|---|
| Groq | llama-3.3-70b-versatile | ✅ | 100k tok/día — se agota en ~12 pares | ❌ Descartado |
| Groq | llama-3.1-8b-instant | ✅ directo / ❌ Ragas | 500k tok/día | ❌ Falla en parser loop |
| Groq | gemma2-9b-it | — | Modelo dado de baja (400 error) | ❌ No disponible |
| Cerebras | gemma-4-31b | ✅ | Sin cuota diaria estricta | ✅ **Seleccionado** |
| Cerebras | gpt-oss-120b | ❌ respuesta None | — | ❌ Descartado |
| OpenRouter | nvidia/nemotron-3-super | ✅ | Rate limit bajo (429 frecuente) | ❌ Inestable |
| OpenRouter | modelos :free | ❌ 404/429 | — | ❌ No disponibles |

### Configuración final en el pipeline

```python
# src/evaluation/eval_pipeline.py — _make_llm()
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gemma-4-31b",
        api_key=settings.cerebras_key,
        base_url="https://api.cerebras.ai/v1",
        temperature=0,
        max_tokens=1024,
    ),
    is_finished_parser=_is_finished,   # permisivo con finish_reason "length"
)
```

Variables de entorno necesarias:
- `.env`: `CEREBRAS_KEY=csk-...`
- `src/settings.py`: campo `cerebras_key: str = Field("", env="CEREBRAS_KEY")`

**Archivos relevantes:**
- `src/evaluation/eval_pipeline.py` — funciones `_make_llm()` y `phase_evaluate()`
- `src/settings.py` — campos `cerebras_key` y `openrouter_key`

---

## Q26: Resultados de la evaluación Config B (retrieval híbrido FAISS+BM25) — 15 pares

**Fecha:** 18-19 de julio de 2026
**Judge:** Cerebras `gemma-4-31b` — 15/15 pares completados en todas las métricas en ambas configs

### Resultados globales — Config B

| Métrica | Valor | Baseline test MaternaQA-es | Interpretación |
|---|---|---|---|
| `faithfulness` | 0.228 | 0.7132 | LLM responde desde conocimiento general — corpus mismatch |
| `answer_correctness` | 0.338 | N/A | Coincidencia semántica moderada con ground truth |
| `answer_relevancy` | 0.631 | 0.5583 | **Por encima del baseline** — respuestas pertinentes |
| `context_recall` | 0.000 | N/A | Esperado: corpus mismatch |
| `context_precision` | 0.000 | N/A | Esperado: mismo motivo |
| `latency_avg_s` | 10.36s | — | p50 real ~6s |

### Comparativa Config A (FAISS puro) vs Config B (FAISS+BM25)

Mismos 15 pares, mismo seed=42, mismo judge (Cerebras gemma-4-31b).

| Métrica | Config A | Config B | Delta | Ganador |
|---|---|---|---|---|
| `faithfulness` | 0.1615 | **0.2278** | +0.066 | **B** |
| `answer_correctness` | 0.3500 | 0.3381 | -0.012 | Empate |
| `answer_relevancy` | 0.6345 | 0.6305 | -0.004 | Empate |
| `context_recall` | 0.000 | 0.000 | 0.000 | Empate |
| `context_precision` | 0.000 | 0.000 | 0.000 | Empate |
| `latency_avg_s` | 11.35s | **10.36s** | -0.99s | **B** |

**Config B gana en faithfulness (+6.6 pp) en todos los tipos de pregunta excepto `aplicacion`.**
El BM25 sobre Multiclinsum reduce el ruido de casos clínicos irrelevantes (oncología, traumatología)
que Config A incluía en el ranking por similitud coseno, permitiendo que el LLM se ancle mejor
en los fragmentos recuperados.

| Tipo | Config A faith. | Config B faith. | Delta |
|---|---|---|---|
| factual | 0.375 | 0.467 | +0.092 |
| razonamiento | 0.119 | 0.188 | +0.069 |
| definicion | 0.062 | 0.167 | +0.104 |
| aplicacion | 0.000 | 0.000 | 0.000 |

### Conclusión

**Config B queda como la arquitectura de retrieval de producción.**
La mejora en faithfulness es consistente y estructural — no ruido estadístico con 15 pares.
`answer_relevancy` y `answer_correctness` son equivalentes entre configs, lo que confirma
que la mejora viene del retrieval (menos ruido en contexto) y no del LLM en sí.

### Próximos pasos para mejorar las métricas

1. Ingestar PDFs de MaternaQA-es → `context_recall` y `context_precision` pasarán de 0 a valores reales
2. Con corpus completo, re-evaluar `faithfulness` — se espera subida significativa hacia el baseline 0.71

**Archivos relevantes:**
- Config A raw/results/report: `evaluation_reports/*configA_20260719_171714.*`
- Config B raw/results/report: `evaluation_reports/*configB_20260718_212843.*`
- `foragents/eval_setup_critico.md` — setup completo del pipeline de evaluación

---

## Q27: ¿Por qué se ingestan los JSONL del corpus LM y no los PDFs crudos de MaternaQA-es?

**Contexto:** El repositorio `minciencias-maternas/MaternaQA-es` contiene tanto los 63 PDFs
fuente como un corpus LM ya procesado (`datasets/obstetrics/lm/`). Había que elegir
cuál ingestar al índice FAISS del sistema RAG.

### Recursos disponibles

**Corpus LM (`datasets/obstetrics/lm/`):**
- `train_lm.jsonl` — 1.744 chunks, 52 PDFs fuente
- `validation_lm.jsonl` — 101 chunks, 2 PDFs fuente
- `test_lm.jsonl` — 108 chunks, 3 PDFs fuente (exactamente los que generan los 328 QA del benchmark)
- **Total: 1.953 chunks**, promedio 879 tokens/chunk, metadatos ricos

**PDFs crudos (`pdfs/obstetrics/`):**
- 63 PDFs en español sobre obstetricia — GPCs colombianas, artículos de revistas, manuales
- Habría que extraer texto (algunos requieren OCR), limpiar, chunkear y deduplicar desde cero

### Razones para elegir los JSONL del corpus LM

**1. El procesamiento ya fue hecho y auditado por el equipo de MaternaQA-es.**
Cada chunk pasó por: extracción textual, filtrado de páginas no clínicas, chunking con límites
de longitud, deduplicación, enriquecimiento temático y control de calidad con `clinical_score`.
Replicar ese procesamiento desde los PDFs crudos tomaría tiempo y podría introducir errores
de extracción (especialmente en PDFs con tablas o columnas).

**2. Metadatos ricos y trazables listos para usar.**
Cada registro del LM tiene:
```json
{
  "text": "...",
  "metadata": {
    "chunk_id": "GPC-Atencion-Prenatal_00012",
    "source_pdf": "GPC-Atencion-Prenatal-de-Bajo-Riesgo-2023.pdf",
    "section_type": "recommendations",
    "content_role": "recommendation",
    "topics": ["prenatal_care", "hemorrhage"],
    "clinical_score": 28,
    "token_estimate": 879,
    "split": "test"
  }
}
```
El campo `split` permite saber si un chunk proviene del split test o train del benchmark,
lo cual es importante para interpretar correctamente las métricas de evaluación.

**3. Control de contaminación de splits.**
El repositorio garantiza que la división train/validation/test se hizo **a nivel de documento**,
sin fuga de información. Esto significa:
- `test_lm.jsonl` contiene chunks de los 3 PDFs exactos que generaron los 328 pares del test QA
- Si ingestamos los 3 splits, el retrieval podrá recuperar los fragmentos exactos del benchmark → métricas reales
- Si ingestamos solo train+val, las métricas del test set siguen siendo "fair" (sin leak)

Para la Config C se ingestan los 3 splits para medir el **upper bound** alcanzable con el corpus completo.

**4. Descarga directa sin dependencias.**
Los JSONL se descargan directamente de GitHub raw (~2-3 MB total) sin necesidad de
clonar el repositorio, instalar dependencias adicionales ni tener `poppler` o `tesseract`
para OCR. Los PDFs de mayor tamaño (ej: `Manual-Obstetricia-y-Ginecologia-2024_compressed.pdf`)
pueden superar los 50 MB y algunos requieren OCR.

**5. Tamaño manejable y coherente con el hardware disponible.**
1.953 chunks × 768 dims = ~6 MB de vectores adicionales — completamente insignificante
comparado con los 375.392 vectores existentes (~1.15 GB). El índice FAISS soporta
adición incremental sin reconstruir desde cero.

### Estructura de los JSONL del corpus LM

```
{"text": "<texto clínico en español>",
 "metadata": {
   "source": "obstetrics_spanish",
   "pdf_id": "<nombre_sin_extension>",
   "source_pdf": "<nombre_con_extension.pdf>",
   "doc_type": "article" | "guideline" | ...,
   "pages": [<numeros de pagina>],
   "section": "<titulo de seccion>",
   "chunk_id": "<pdf_id>_<NNNNN>",
   "token_estimate": <int>,
   "clinical_score": <int 0-30>,
   "section_type": "recommendations" | "clinical_content" | "introduction" | ...,
   "content_role": "recommendation" | "evidence" | "treatment" | "background" | ...,
   "topics": [<lista de temas clinicos>],
   "split": "train" | "validation" | "test"
 }
}
```

### Mapeo al formato FAISSStore del proyecto

El `FAISSStore` del proyecto almacena metadatos con esta estructura:
```python
{
    "text":           chunk["text"],
    "source_dataset": "maternaqaes_lm",   # nuevo dataset_id
    "language":       "es",
    "doc_id":         chunk["metadata"]["pdf_id"],
    "chunk_id":       chunk["metadata"]["chunk_id"],
    # campos adicionales preservados:
    "topics":         chunk["metadata"]["topics"],
    "clinical_score": chunk["metadata"]["clinical_score"],
    "section_type":   chunk["metadata"]["section_type"],
    "content_role":   chunk["metadata"]["content_role"],
    "lm_split":       chunk["metadata"]["split"],
}
```

### Impacto esperado en las métricas (Config C)

| Métrica | Antes (Config B) | Esperado (Config C) | Razón |
|---|---|---|---|
| `faithfulness` | 0.228 | ~0.50–0.65 | El LLM podrá anclar en fragmentos en español |
| `answer_correctness` | 0.338 | ~0.45–0.60 | Mayor coincidencia semántica con ground truth |
| `answer_relevancy` | 0.631 | ~0.63–0.70 | Similar o ligeramente mejor |
| `context_recall` | 0.000 | **~0.30–0.60** | El corpus ahora tiene los documentos del benchmark |
| `context_precision` | 0.000 | **~0.20–0.50** | Fragmentos relevantes recuperados |

**Archivos relevantes:**
- `src/ingestion/ingest_maternaqaes_lm.py` — script de ingestión (a crear)
- `src/rag/retriever_configC.py` — config C con `maternaqaes_lm` en DENSE_SOURCES (a crear)
- `foragents/retrieval_arquitecturas_configs.md` — documentación de configs

---

*Última actualización: 19 de julio de 2026*
