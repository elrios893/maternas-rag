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

*Última actualización: 3 de julio de 2026*
