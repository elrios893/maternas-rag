# Maternas — Chatbot RAG de Salud Materna

Chatbot conversacional basado en arquitectura RAG orientado a madres gestantes. Clasifica la intención del usuario, evalúa el riesgo clínico y genera respuestas fundamentadas en literatura médica.

> Proyecto de investigación — Convocatoria 890 Minciencias · Institución Universitaria de Envigado

---

## Stack

| Capa | Tecnología |
|---|---|
| Embedding | `intfloat/multilingual-e5-base` (768 dims, ES/EN/ZH) en CPU por defecto |
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
pip install -r requirements.txt

# 2. Configuración
cp .env.example .env          # completar GROQ_API_KEY y rutas de datasets

# 2.1. Si no tienes GPU NVIDIA, deja EMBEDDING_DEVICE=cpu

# 2.2. Si sí tienes GPU NVIDIA, puedes cambiar EMBEDDING_DEVICE=cuda

# 3. Ingestión (una sola vez; en CPU tarda bastante más)
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
query → classify_intent() → detect_risk() → FAISS retrieve() → Groq LLM → respuesta
```

- **Riesgo HIGH** → alerta inmediata + respuesta de urgencia
- **Riesgo MEDIUM** → respuesta con recomendación de consulta médica
- **Riesgo LOW** → respuesta educativa con citas a la fuente

## Siguientes mejoras
- **Integración con telegram** → Chatbot especializado en telegram.
- **Fine-tunning con QLORA**
