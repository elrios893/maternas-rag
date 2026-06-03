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
```

UI disponible en `http://localhost:8501` · API docs en `http://localhost:8080/docs`

## Flujo por turno

```
query → classify_intent() → detect_risk() → FAISS retrieve() → Groq LLM → respuesta
```

- **Riesgo HIGH** → alerta inmediata + respuesta de urgencia
- **Riesgo MEDIUM** → respuesta con recomendación de consulta médica
- **Riesgo LOW** → respuesta educativa con citas a la fuente
