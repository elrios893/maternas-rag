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
| Bot | Telegram (`python-telegram-bot`) con JobQueue integrada |
| Scheduler | Integrado dentro del bot (APScheduler vía PTB) — **1 solo proceso** |

## Datasets indexados

- **Multiclinsum** — 25,902 casos clínicos en español
- **MedMCQA** — 187,005 preguntas médicas (EN)
- **MedQA** — US / Taiwan / Mainland + 18 textbooks médicos (EN)

---

## Estructura del proyecto

```
src/
├── ingestion/          # formatters, chunkers, embedder, FAISS store, scripts de ingestión
├── classifiers/
│   ├── intent_classifier.py   # clasificación de intención (12 categorías vía Groq)
│   └── risk_detector.py       # detector de riesgo clínico (heurística + LLM)
├── rag/
│   ├── retriever.py           # búsqueda en FAISS (IndexFlatIP)
│   └── chain.py               # cadena RAG completa (orquestación del flujo)
├── api/
│   ├── main.py                # FastAPI (endpoints /chat, /health, /classify)
│   └── schemas.py             # modelos Pydantic para request/response
├── bot/
│   ├── maternas_bot.py        # [ÚNICO PROCESO] Bot Telegram + Status Check Scheduler
│   ├── active_users.py        # Registro de usuarios activos y puntajes de riesgo
│   └── status_scheduler.py    # [DEPRECADO] — ahora integrado en maternas_bot.py
├── ui/
│   └── app.py                 # Streamlit
└── settings.py                # Configuración central (pydantic-settings)

active_users.json               # Persistencia en disco: puntajes de riesgo por usuario
foragents/                      # plan técnico, casos de prueba y Q&A del proyecto
```

---

## Inicio rápido

```bash
# 1. Entorno
python -m venv venv
.\venv\Scripts\activate       # Windows
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers==2.7.0
pip install -r requirements.txt

# 2. Configuración
cp .env.example .env          # completar GROQ_API_KEY y TELEGRAM_BOT_TOKEN

# 3. Ingestión (una sola vez, ~5h en GPU)
python src/ingestion/run_ingestion.py

# 4. Arrancar (solo 2 procesos necesarios)
python -m uvicorn src.api.main:app --port 8080   # Terminal 1: API
python src/bot/maternas_bot.py                    # Terminal 2: Bot + Scheduler
```

UI disponible en `http://localhost:8501` · API docs en `http://localhost:8080/docs`

> **IMPORTANTE:** Ya **no** se ejecuta `python src/bot/status_scheduler.py`. El scheduler está integrado dentro de `maternas_bot.py`. Un solo proceso hace todo.

---

## Bot Telegram

### Comandos

| Comando | Descripción |
|---|---|
| `/start` | Mensaje de bienvenida |
| `/help` | Instrucciones de uso |
| `/reset` | Reinicia la conversación (borra historial en RAM) |
| `/stats` | Info del sistema (vectores indexados, estado FAISS) |

### Historial conversacional

- Almacenado **en RAM**: `histories: dict[int, list[dict]]` en `maternas_bot.py:55`
- Se pierde al reiniciar el bot — suficiente para MVP, evita complejidad de persistencia.
- Cada turno se envía a la API (`POST /chat`) incluyendo el historial acumulado.

### Formato de mensajes

Los mensajes se envían en **dos partes** para evitar errores de parseo del LLM:

1. **Header informativo** (HTML) — nivel de riesgo, banderas clínicas, acción recomendada
2. **Respuesta de Maternas** (texto plano, sin parse_mode) — el contenido generado por el LLM

---

## Flujo completo por turno

```
Usuario escribe en Telegram
  │
  ▼
maternas_bot.py :: handle_message()
  │
  ├─ call_chat(text, user_id)  →  httpx POST a http://localhost:8080/chat
  │
  ▼
api/main.py :: POST /chat
  │
  ├─ chain.chat(query, history)
  │    ├── 1. classify_intent()       → 12 categorías vía Groq
  │    ├── 2. detect_risk()           → heurística (rápida) + LLG (contextual)
  │    ├── 3. retrieve(query)         → FAISS: top-k fragmentos
  │    ├── 4. Groq LLM (con contexto) → respuesta generada
  │    └── 5. ChatResponse            → answer, intent, risk_level, sources
  │
  ▼
maternas_bot.py :: handle_message() (continuación)
  │
  ├─ _api_healthy = True              ← la API respondió OK
  ├─ register_active_user(chat_id, risk_level, risk_flags)
  │    └── active_users.py::register()
  │         └── active_users.json     ← se guarda en disco
  │
  ├─ Enviar header (HTML) con nivel de riesgo
  └─ Enviar respuesta (texto plano)
```

---

## Sistema de Puntajes de Riesgo

### ¿Por qué existe?

Cada interacción del usuario deja una "huella de riesgo". El sistema acumula esos puntajes para:
1. **Ajustar la frecuencia** de los mensajes de seguimiento (status checks)
2. **Dar prioridad** a usuarios con mayor riesgo clínico
3. **Permitir decaimiento natural** si el usuario está estable por un tiempo

### ¿Dónde se almacenan?

**`active_users.json`** — archivo JSON en la raíz del proyecto. Cada usuario tiene esta estructura:

```json
{
  "123456789": {
    "risk_points": 16,
    "latest_risk_level": "high",
    "latest_risk_flags": ["hemorragia", "dolor_intenso"],
    "last_activity": "2026-07-02T06:29:14+00:00",
    "last_check_sent": "2026-07-02T06:40:52+00:00"
  }
}
```

| Campo | Tipo | Descripción | Quién lo escribe |
|---|---|---|---|
| `risk_points` | int (0–50) | Puntaje acumulado de riesgo | `register()` |
| `latest_risk_level` | `"low"` / `"medium"` / `"high"` | Último nivel detectado | `register()` |
| `latest_risk_flags` | `list[str]` | Banderas clínicas del último mensaje | `register()` |
| `last_activity` | ISO 8601 | Timestamp del último mensaje del usuario | `register()` |
| `last_check_sent` | ISO 8601 | Timestamp del último status check enviado | `update_check_sent()` |

### ¿Cómo se calculan los puntos?

Archivo: `src/bot/active_users.py`

```python
_RISK_POINTS = {"low": 0, "medium": 3, "high": 10}
_RISK_DECAY_PER_HOUR = 1
_RISK_MAX_POINTS = 50
```

Cada vez que el usuario envía un mensaje, `register()` hace lo siguiente:

```
1. Aplicar decaimiento:
     risk_points = max(0, risk_points - horas_inactivo)
     (se descuenta 1 punto por cada hora desde last_activity)

2. Sumar puntos del nivel actual:
     risk_points = min(50, risk_points + _RISK_POINTS[risk_level])

3. Actualizar metadatos:
     last_activity = ahora
     latest_risk_level = nivel detectado
     latest_risk_flags = banderas del mensaje

4. Persistir en disco:
     active_users.json ← thread-safe con Lock
```

### Decaimiento automático — ejemplo práctico

| Escenario | risk_points resultante |
|---|---|
| Usuario dice algo LOW (0 pts) después de 5h inactivo con 20 pts | `max(0, 20 - 5) + 0 = 15` |
| Usuario dice algo HIGH (10 pts) después de 2h inactivo con 5 pts | `max(0, 5 - 2) + 10 = 13` |
| Usuario dice algo MEDIUM (3 pts) después de 24h inactivo con 50 pts | `max(0, 50 - 24) + 3 = 29` |
| Usuario nuevo dice algo LOW (0 pts) | 0 |
| Usuario nuevo dice algo HIGH (10 pts) | 10 |

### Thread-safety

El archivo `active_users.json` es accedido desde dos lugares concurrentemente:

| Desde | Cuándo | Operación |
|---|---|---|
| `maternas_bot.py::handle_message()` | Cada mensaje del usuario | `register()` → escribe |
| `maternas_bot.py::_sync_user_jobs()` | Cada 15 segundos | `get_all()` → lee |
| `maternas_bot.py::_send_status_check()` | Cada N segundos por usuario | `update_check_sent()` → escribe |

Para evitar condiciones de carrera, todas las operaciones de lectura/escritura usan `threading.Lock` (`active_users.py:31`).

### ¿Quién alimenta el sistema de puntajes?

1. **`risk_detector.py`** — evalúa el mensaje y produce un nivel (low/medium/high) + flags
2. **`chain.py`** — orquesta: llama a `classify_intent()` → `detect_risk()` → RAG → LLM
3. **`api/main.py`** — expone el endpoint `/chat` que devuelve `risk_level` y `risk_flags`
4. **`maternas_bot.py:195`** — recibe la respuesta de la API y llama a `register_active_user()`
5. **`active_users.py`** — persiste en `active_users.json`

---

## Detección de Riesgo Clínico

`src/classifiers/risk_detector.py`

Sistema de **dos capas** que prioriza velocidad en emergencias.

### Capa 1: Heurística (instantánea — sin llamadas a Groq)

Busca coincidencia de keywords en el mensaje del usuario. Si encuentra alguna, retorna inmediatamente sin消耗 recursos del LLM.

**HIGH si match en:**
| Categoría | Ejemplos de detección |
|---|---|
| `hemorragia` | "sangrando mucho", "coágulos", "hemorragia" |
| `preeclampsia` | "visión borrosa", "hinchazón súbita de cara" |
| `eclampsia_convulsion` | "convulsión", "pérdida de conocimiento" |
| `trabajo_parto_prematuro` | "contracciones antes de las 37" |
| `ruptura_membranas` | "rompí fuente", "líquido amniótico" |
| `movimiento_fetal_ausente` | "no se mueve", "no siento movimientos" |
| `dolor_intenso` | "dolor insoportable", "dolor abdominal agudo" |
| `signos_sepsis` | "fiebre de 40", "escalofríos con fiebre" |
| `depresion_grave` | "quiero hacerme daño", "ideas de suicidio" |

**MEDIUM si match en:**
| Categoría | Ejemplos |
|---|---|
| `sangrado_leve` | "manchado", "spotting" |
| `presion_alta_leve` | "presión alta", "hipertensión" |
| `edema_moderado` | "pies muy hinchados" |
| `dolor_moderado` | "dolor de cabeza que no pasa" |
| `fiebre_moderada` | "fiebre", "temperatura alta" |
| `reduccion_movimiento` | "se mueve menos" |
| `sintomas_infeccion` | "ardor al orinar", "flujo con mal olor" |

### Capa 2: LLM (Groq)

Solo se ejecuta si la heurística no encontró nada. Evalúa contextualmente el riesgo con un prompt estructurado. Usa `temperature=0.0` para consistencia y `response_format: json_object` para obtener JSON válido.

### Si el LLM falla

Se retorna `level="medium"` por precaución (nunca asumir LOW si no estamos seguros).

### Salida

```python
@dataclass
class RiskResult:
    level: str            # "low" | "medium" | "high"
    flags: list[str]      # ["hemorragia", "dolor_intenso"]
    action: str           # "educational_answer" | "medical_consultation" | "urgent_care"
    reasoning: str        # explicación en lenguaje natural
    used_heuristic: bool  # True si la detección fue instantánea
    raw: Optional[str]    # respuesta cruda del LLM (para debug)
```

---

## Status Check Scheduler

### ¿Qué es?

Es un sistema que envía mensajes periódicos de "check de estado" a los usuarios activos del bot Telegram, preguntando cómo se sienten. La frecuencia del mensaje depende del nivel de riesgo acumulado del usuario.

### Arquitectura (unificada)

Antes requería **2 procesos separados** (`maternas_bot.py` + `status_scheduler.py`). Ahora está **integrado dentro de `maternas_bot.py`** usando la `JobQueue` nativa de python-telegram-bot.

| Aspecto | Antes | Ahora |
|---|---|---|
| Procesos | 2 (bot + scheduler) | 1 (solo bot) |
| Cliente Telegram | Bot usaba PTB, scheduler usaba `requests` directo | Ambos usan `context.bot.send_message()` |
| Conexiones HTTP | 2 conexiones separadas a Telegram API | 1 sola conexión reutilizada |
| Sincronización | Vía archivo JSON compartido | Vía archivo JSON + variables en memoria |
| Congruencia | Podía enviar checks aunque el bot estuviera caído | Solo envía checks si la API responde |

### ¿Cómo usa el timer los puntajes de riesgo?

Cada usuario tiene un **job independiente** con su propio intervalo. El intervalo se determina así:

```python
def _check_interval(risk_level: str) -> float:
    return {
        "low":    60,   # 0–2 puntos: cada 1 minuto (en desarrollo)
        "medium": 45,   # 3–9 puntos: cada 45 segundos
        "high":   30,   # 10–50 puntos: cada 30 segundos
    }[risk_level]
```

| Nivel | risk_points | Intervalo (desarrollo) | Producción sugerida |
|---|---|---|---|
| LOW | 0–2 | 60 segundos | 3600s (1 hora) |
| MEDIUM | 3–9 | 45 segundos | 1800s (30 min) |
| HIGH | 10–50 | 30 segundos | 300s (5 min) |

**¿Por qué este mapeo y no el anterior?** El scheduler anterior usaba una fórmula matemática: `max(MIN, BASE / (1 + risk_points / 10))`. Esto hacía difícil predecir el intervalo exacto. El nuevo mapeo es directo por nivel de riesgo, más predecible y fácil de configurar.

### Componentes del scheduler

#### 1. Sync Job (`_sync_user_jobs`)
- Se ejecuta cada **15 segundos** (configurado en `main()`, línea 344)
- Lee `active_users.json` completo
- Compara los usuarios actuales contra los jobs activos en memoria
- **Crea** jobs para usuarios nuevos
- **Re-programa** jobs cuando el nivel de riesgo cambió
- **Elimina** jobs de usuarios que ya no existen en el JSON

#### 2. Per-User Jobs (`_send_status_check`)
- Un job recurrente independiente por cada usuario
- Envía `STATUS_CHECK_MESSAGE` al `chat_id` del usuario
- **Verifica `_api_healthy` antes de enviar** — si la API no responde, omite el envío
- Actualiza `last_check_sent` en `active_users.json`

### Congruencia: el respeto por la salud del bot

Variable global `_api_healthy` declarada en `maternas_bot.py:64`:

```python
_api_healthy: bool = False   # arranca en False
```

Se actualiza en cada `handle_message()`:

```python
# Si la API respondió OK
result = await call_chat(text, user_id)
if result is None:
    _api_healthy = False    # API caída
    return
_api_healthy = True         # API funcionando
```

El callback `_send_status_check` verifica el flag:

```python
async def _send_status_check(context):
    if not _api_healthy:
        return  # no envía nada, silenciosamente
    # ... enviar mensaje ...
```

**Escenarios:**

| Situación | _api_healthy | ¿Se envían checks? |
|---|---|---|
| Bot arranca, nadie ha escrito aún | `False` | No |
| Usuario escribe y API responde 200 | `True` | Sí (a partir del próximo ciclo) |
| API se cae durante operación | `False` | No (checks omitidos) |
| API vuelve, usuario escribe OK | `True` | Sí (se reanudan) |
| Bot corriendo pero nadie escribe desde hace horas | `True` (último valor conocido) | Sí (con el último riesgo conocido) |

### Flujo completo del scheduler

```
INICIO: main()
  │
  ├─ app.job_queue.run_repeating(_sync_user_jobs, interval=15s)
  │
  ▼
CADA 15 SEGUNDOS: _sync_user_jobs()
  │
  ├─ 1. Leer active_users.json → get_all()
  ├─ 2. Calcular: active = set(users.keys())
  │              existing = set(_user_status_jobs.keys())
  │
  ├─ 3. Eliminar jobs de usuarios que ya no existen
  │     for chat_id in existing - active:
  │         job.schedule_removal()
  │
  └─ 4. Para cada usuario en active_users.json:
        │
        ├─ risk_level = user["latest_risk_level"]
        ├─ interval = _check_interval(risk_level)
        │
        ├─ ¿Usuario NUEVO?
        │   └─ job = run_repeating(_send_status_check, interval, chat_id)
        │       _user_status_jobs[chat_id] = job
        │
        └─ ¿Usuario EXISTENTE pero riesgo cambió?
            └─ job.schedule_removal()
               job = run_repeating(_send_status_check, new_interval, chat_id)
               _user_status_jobs[chat_id] = job


CADA N SEGUNDOS (por usuario): _send_status_check()
  │
  ├─ ¿_api_healthy == False?
  │   └─ return  (no envía nada)
  │
  ├─ context.bot.send_message(chat_id, STATUS_CHECK_MESSAGE)
  └─ update_check_sent(chat_id) → escribe en active_users.json
```

### Diagrama de estados del scheduler

```
                         ┌─────────────────┐
                         │  Bot inicia     │
                         │  _api_healthy=F │
                         └────────┬────────┘
                                  │
                                  ▼
              ┌──────────────────────────┐
              │  Sync job arranca (15s)  │
              │  → Lee active_users.json │
              │  → No hay usuarios aún   │
              │  → No crea jobs          │
              └──────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Usuario escribe → handle_message()     │
    │  → call_chat() responde 200             │
    │  → _api_healthy = True                  │
    │  → register() escribe en JSON           │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │  Sync job corre a los ≤15s             │
    │  → Detecta usuario NUEVO en JSON       │
    │  → risk_level = "high" → interval = 30s│
    │  → Crea job para ese chat_id           │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │  Cada 30s: _send_status_check()         │
    │  → _api_healthy=True → envía mensaje    │
    │  → update_check_sent() → escribe en JSON│
    └─────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│ Usuario responde │    │ API se cae          │
│ → HIGH suma pts │    │ → _api_healthy=False │
│ → sync detecta  │    │ → checks omitidos    │
│ → re-programa   │    │ → hasta que API      │
│   con +frecuencia│   │   responda de nuevo   │
└─────────────────┘    └──────────────────────┘
```

---

## Recursos Utilizados

| Recurso | Archivo | Uso | Impacto |
|---|---|---|---|
| `active_users.json` | Raíz del proyecto | Persistencia de puntajes y metadatos | I/O en cada mensaje y cada status check |
| `histories` (RAM) | `maternas_bot.py:55` | `dict[int, list[dict]]` — historial conversacional | Se pierde al reiniciar, crece con usuarios activos |
| `threading.Lock` | `active_users.py:31` | Exclusión mutua para lectura/escritura del JSON | Baja contención, no es cuello de botella |
| JobQueue + thread pool | PTB (APScheduler interno) | Ejecución de jobs periódicos (1 sync + N por usuario) | ~1 hilo por job activo |
| Conexión Telegram | PTB (long polling) | Recepción de mensajes + envío de respuestas y checks | 1 sola conexión HTTPS persistente |
| `_user_status_jobs` (RAM) | `maternas_bot.py:243` | `dict[str, Job]` — referencia a jobs activos | Crece con usuarios activos |

### Consumo estimado

Para **N usuarios activos** simultáneamente:
- Jobs en scheduler: `N + 1` (1 sync + N per-user)
- Archivos abiertos: 1 (JSON, se abre/cierra en cada operación)
- RAM de historiales: aprox. 1–5 KB por usuario (depende del largo de la conversación)
- RAM de scheduler: aprox. 200 bytes por usuario (referencia al Job + string de risk_level)

---

## Posibles Optimizaciones

### 1. Cache en memoria para active_users.json

**Problema:** Cada `register()` y `update_check_sent()` abre, lee, escribe y cierra el archivo JSON. Con muchos usuarios y mensajes frecuentes, esto genera I/O repetitivo.

**Solución:** Mantener un dict en memoria y persistir al disco cada `N` segundos o en lote.

```python
# Idea general
_cache: dict[str, dict] = {}
_dirty: bool = False

def register(...):
    _cache[key] = data
    _dirty = True

def _flush():
    if _dirty:
        _save(_cache)
        _dirty = False

# Un job aparte cada 30s ejecuta _flush()
```

**Beneficio:** reduce I/O de disco de O(mensajes) a O(1 cada 30s). **Riesgo:** perder hasta 30s de datos si el bot se cae.

### 2. Persistencia del historial conversacional

**Problema:** `histories` vive en RAM. Si el bot se reinicia, todos los usuarios pierden el contexto de la conversación.

**Solución:** Usar SQLite (sin servidor, embedido) o Redis (si ya hay infraestructura).

```python
# SQLite: tabla conversations(user_id, role, content, created_at)
def save_message(user_id, role, content):
    cursor.execute("INSERT INTO messages VALUES (?, ?, ?, ?)", ...)

def load_history(user_id, limit=10):
    cursor.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY created_at DESC LIMIT ?", ...)
```

**Beneficio:** historial persistente entre reinicios, los usuarios no notan caídas del bot.

### 3. Backoff para usuarios que no responden

**Problema:** Si un usuario no abre Telegram o ignora los status checks, el scheduler sigue enviando mensajes indefinidamente con la misma frecuencia.

**Solución:** Rastrear cuántos checks consecutivos no han recibido respuesta (el usuario no escribe después del check). Después de `N` checks sin respuesta, espaciar el intervalo progresivamente (ej. ×2, ×4, ×8) hasta detener los envíos.

```python
# En active_users.json, agregar:
"missed_checks": 3

# En _send_status_check, si missed_checks > 5:
#   interval *= 2  (backoff exponencial)
```

**Beneficio:** reduce mensajes innecesarios a usuarios inactivos. **Riesgo:** un usuario en riesgo podría dejar de recibir seguimiento si no responde; requiere lógica de límite máximo.

### 4. Métricas de scheduler expuestas vía /health

**Problema:** No hay visibilidad del estado del scheduler (cuántos jobs activos, cuántos checks enviados, cola de jobs).

**Solución:** Exponer métricas en el endpoint `/health` de la API.

```python
# En api/main.py
@app.get("/health")
def health():
    return {
        "scheduler_active_jobs": len(_user_status_jobs),  # desde maternas_bot
        "scheduler_checks_sent": ...,  # contador acumulado
        "api_healthy": _api_healthy,
    }
```

**Beneficio:** monitoreo en tiempo real del comportamiento del scheduler.

---

## Variables de Entorno

| Variable | Default | Descripción |
|---|---|---|
| `GROQ_API_KEY` | — | API key de Groq (requerida) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Modelo de Groq para clasificación y generación |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Modelo de embeddings multilingüe |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` si hay GPU, `cpu` si no |
| `FAISS_STORE_PATH` | `./faiss_store` | Ruta del índice FAISS persistido |
| `RAG_TOP_K` | `5` | Fragmentos recuperados por consulta |
| `TELEGRAM_BOT_TOKEN` | — | Token del bot de Telegram (requerido) |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `STATUS_CHECK_INTERVAL_LOW_SECONDS` | `60` | Intervalo para riesgo LOW (segundos) |
| `STATUS_CHECK_INTERVAL_MEDIUM_SECONDS` | `45` | Intervalo para riesgo MEDIUM (segundos) |
| `STATUS_CHECK_INTERVAL_HIGH_SECONDS` | `30` | Intervalo para riesgo HIGH (segundos) |
| `STATUS_CHECK_MESSAGE` | Ver `.env.example` | Texto del mensaje de status check (Markdown) |

---

## Preguntas Frecuentes para Desarrolladores

### ¿Por qué unificar bot y scheduler?

Porque antes se necesitaban dos terminales/procesos, cada uno con su propia conexión a Telegram. El scheduler podía mandar mensajes incluso cuando el bot estaba caído, creando incongruencia. Al unificarlos, un solo proceso maneja todo y el scheduler sabe si el bot está vivo.

### ¿El scheduler funciona aunque nadie haya escrito al bot?

No. `_api_healthy` arranca en `False`. Solo pasa a `True` cuando alguien escribe y la API responde OK. Hasta entonces, el scheduler existe (el sync job corre cada 15s) pero los per-user jobs omiten el envío.

### ¿Qué pasa si cambio las variables de entorno en caliente?

`settings.py` se carga una vez al importar el módulo. Si cambias `.env` mientras el bot corre, no tendrá efecto hasta que reinicies el bot. No hay watch de archivo de configuración.

### ¿Qué pasa si hay muchos usuarios activos?

Cada usuario activo genera 1 job recurrente en APScheduler. Para 1000 usuarios activos, serían 1000 jobs + 1 sync = 1001 jobs. APScheduler maneja esto con un thread pool (default: 10 threads). Si los jobs son muchos más que los threads, se encolan. Esto es aceptable porque los jobs son livianos (solo revisan un flag y envían un mensaje).

### ¿Cómo depurar el scheduler?

Los logs del scheduler están etiquetados con `__name__` (`maternas_bot`). Buscar en la salida estándar:
```
INFO - Scheduler de status check integrado — sync cada 15s
DEBUG - Job creado para 123456789 — riesgo=high, cada 30s
DEBUG - Status check enviado a 123456789
DEBUG - API no disponible — status check omitido
```

---

## Historial de cambios

| Fecha | Cambio |
|---|---|
| Jul 2026 | Status Check Scheduler unificado dentro de `maternas_bot.py` |
| Jul 2026 | Intervalos cambiados de fórmula matemática a mapeo directo por nivel de riesgo (segundos) |
| Jul 2026 | Flag `_api_healthy` para congruencia: no enviar checks si la API no responde |
| Jul 2026 | `status_scheduler.py` marcado como deprecado |
| Jun 2026 | Creación inicial del proyecto |
