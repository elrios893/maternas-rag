"""
chain.py — Cadena RAG completa del chatbot Maternas.

Orquesta el flujo completo por turno:
  1. classify_intent()   → qué quiere el usuario
  2. detect_risk()       → nivel de riesgo clínico
  3. retrieve()          → fragmentos relevantes del FAISS
  4. build_prompt()      → prompt con contexto + historial
  5. Groq LLM            → respuesta generada
  6. ChatResponse        → objeto de retorno estructurado

Historial conversacional:
  Se pasa como lista de dicts [{"role": "user"|"assistant", "content": "..."}].
  El caller es responsable de mantenerlo entre turnos.

Uso básico:
    from src.rag.chain import chat
    history = []
    response = chat("¿Qué alimentos debo evitar en el embarazo?", history)
    history.append({"role": "user",      "content": "¿Qué alimentos debo evitar?"})
    history.append({"role": "assistant", "content": response.answer})
    print(response.answer)
    print(response.intent, response.risk_level)
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from groq import Groq

from src.classifiers.intent_classifier import classify_intent, IntentResult
from src.classifiers.risk_detector import detect_risk, RiskResult
from src.rag.retriever import retrieve, format_context, source_label
from src.settings import settings
from src.skills import ToolRegistry
import src.skills.notifier  # noqa: F401 — registra tools del notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipo de retorno
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    answer:       str                      # Respuesta generada al usuario
    intent:       str                      # Intención clasificada
    risk_level:   str                      # "low" | "medium" | "high"
    action:       str                      # "educational_answer" | "medical_consultation" | "urgent_care"
    risk_flags:   list[str] = field(default_factory=list)
    sources:      list[dict] = field(default_factory=list)   # docs recuperados (sin text para ahorrar espacio)
    reasoning:    str = ""                 # reasoning del clasificador de riesgo
    tokens_used:  int = 0
    notified:     bool = False             # si se envió notificación por riesgo


# ---------------------------------------------------------------------------
# System prompt base
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = (
    "Eres Maternas, una asistente de salud dedicada a acompanar con calidez a madres "
    "gestantes y en puerperio. Tu objetivo es que cada mujer se sienta escuchada, "
    "apoyada e informada.\n\n"

    "COMO RESPONDER:\n"
    "- Usa un tono calido, cercano y empatico. Nunca frio ni clinico.\n"
    "- Responde de forma CONCISA y DIRECTA. No repitas la pregunta, no hagas "
    "introducciones largas, no agregues despedidas innecesarias.\n"
    "- Usa lenguaje sencillo. Evita tecnicismos salvo que sean imprescindibles "
    "(y si los usas, explicalos brevemente).\n"
    "- Responde siempre en espanol.\n\n"

    "SOBRE LAS FUENTES:\n"
    "- Si un fragmento contiene informacion util para tu respuesta, cita [n] "
    "al final de esa oracion especifica. Solo cita si el fragmento realmente "
    "dice lo que afirmas — nunca cites para aparentar respaldo.\n"
    "- Si los fragmentos no contienen la informacion exacta pero si informacion "
    "relacionada, usala como apoyo y complementa con conocimiento medico general "
    "bien establecido. En ese caso no es necesario aclarar 'no tengo fuentes' — "
    "simplemente responde con naturalidad.\n"
    "- Si los fragmentos no tienen absolutamente nada relevante, responde desde "
    "conocimiento general sin citar [n].\n"
    "- Los fragmentos marcados [caso clinico] son ejemplos de pacientes reales. "
    "Usaos si el caso ilustra o confirma algo relevante para la pregunta.\n\n"

    "LIMITES:\n"
    "- No eres medico y no reemplazas una consulta. Cuando corresponda, "
    "orienta a consultar con su medico o matrona.\n"
    "- Nunca inventes datos clinicos, dosis ni procedimientos."
)

URGENT_SUFFIX = (
    "\n\nALERTA: Este mensaje contiene senales de alarma clinica. "
    "Comienza tu respuesta indicando de forma clara y directa que debe "
    "buscar atencion medica INMEDIATA. Se breve, urgente y empatica. "
    "No des informacion que pueda hacerla postergar ir a urgencias."
)

MEDIUM_SUFFIX = (
    "\n\nNOTA: El mensaje sugiere un sintoma que merece evaluacion medica. "
    "Incluye al final una recomendacion breve de consultar con su medico o matrona."
)


# ---------------------------------------------------------------------------
# Cliente Groq (singleton)
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Construcción del prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(risk: RiskResult) -> str:
    prompt = BASE_SYSTEM_PROMPT
    if risk.level == "high":
        prompt += URGENT_SUFFIX
    elif risk.level == "medium":
        prompt += MEDIUM_SUFFIX
    return prompt


def _build_messages(
    query: str,
    context: str,
    history: list[dict],
    system_prompt: str,
) -> list[dict]:
    """
    Construye la lista de mensajes para el LLM.
    Incluye hasta los últimos 6 turnos del historial para no exceder
    el context window de Groq con historial largo.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Historial reciente (máx 6 turns = 3 pares user/assistant)
    recent_history = history[-6:] if len(history) > 6 else history
    messages.extend(recent_history)

    # Mensaje actual con contexto inyectado
    user_message = (
        f"CONTEXTO DE LA BASE DE CONOCIMIENTO MÉDICO:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"PREGUNTA DEL USUARIO:\n{query}"
    )
    messages.append({"role": "user", "content": user_message})

    return messages


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def chat(
    query: str,
    history: list[dict] | None = None,
    k: int | None = None,
) -> ChatResponse:
    """
    Procesa un turno completo del chatbot.

    Args:
        query:   Mensaje del usuario.
        history: Historial de la conversación (lista de dicts role/content).
                 Se modifica externamente por el caller.
        k:       Número de fragmentos a recuperar (default: settings.rag_top_k).

    Returns:
        ChatResponse con respuesta, intent, risk_level, sources, etc.
    """
    if history is None:
        history = []

    if not query or not query.strip():
        return ChatResponse(
            answer="No recibí ningún mensaje. ¿En qué puedo ayudarte?",
            intent="pregunta_fuera_de_alcance",
            risk_level="low",
            action="educational_answer",
        )

    # 1. Clasificar intención
    intent_result: IntentResult = classify_intent(query, conversation_history=history)
    logger.info(f"[Chain] intent={intent_result.intent} conf={intent_result.confidence:.2f}")

    # 2. Detectar riesgo
    risk_result: RiskResult = detect_risk(query, intent=intent_result.intent)
    logger.info(f"[Chain] risk={risk_result.level} action={risk_result.action}")

    # ---------------------------------------------------------------------------
    # Notificación por riesgo clínico
    # ---------------------------------------------------------------------------

    NOTIFY_PROMPT = (
        "Eres un clasificador medico. Decide si este mensaje de una paciente "
        "amerita notificar a un clinico para revision.\n\n"
        "Contexto:\n"
        f"- Nivel de riesgo: {risk_result.level}\n"
        f"- Intencion: {intent_result.intent}\n"
        f"- Razonamiento: {risk_result.reasoning}\n"
        f"- Banderas: {risk_result.flags}\n\n"
        f"Mensaje de la paciente:\n{query}\n\n"
        "Responde SOLO con 'YES' si un medico debe revisar este caso, "
        "o 'NO' si no es necesario."
    )
    notified = False
    if risk_result.level == "high":
        ToolRegistry.execute("notify_risk", query=query, risk_level=risk_result.level,
                             intent=intent_result.intent, reasoning=risk_result.reasoning,
                             flags=risk_result.flags)
        notified = True
    elif risk_result.level == "medium":
        client = _get_client()
        try:
            resp = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": NOTIFY_PROMPT}],
                temperature=0,
                max_tokens=10,
            )
            decision = resp.choices[0].message.content.strip().upper()
            if "YES" in decision:
                ToolRegistry.execute("notify_risk", query=query, risk_level=risk_result.level,
                                     intent=intent_result.intent, reasoning=risk_result.reasoning,
                                     flags=risk_result.flags)
                notified = True
        except Exception as e:
            logger.warning(f"[Chain] Error en decision de notificacion medium: {e}")

    # 3. Recuperar fragmentos relevantes
    # Para preguntas fuera de alcance recuperamos igualmente por si acaso
    docs = retrieve(query, k=k)
    context = format_context(docs)
    logger.info(f"[Chain] {len(docs)} fragmentos recuperados")

    # 4. Construir prompt y llamar al LLM
    system_prompt = _build_system_prompt(risk_result)
    messages      = _build_messages(query, context, history, system_prompt)

    client = _get_client()
    tokens_used = 0

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        answer      = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        # 5. Detectar citas [n] en la respuesta y construir referencias
        cited = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
        if cited and docs:
            refs: list[str] = []
            for i, doc in enumerate(docs, 1):
                if i in cited:
                    source = doc.get("source_dataset", "desconocido")
                    refs.append(f"[{i}] {source_label(source)}")
            if refs:
                answer += "\n\n---\n" + "\n".join(refs)

    except Exception as e:
        logger.error(f"[Chain] Error generando respuesta: {e}")
        answer = _fallback_answer(risk_result)

    # 5. Formatear fuentes (sin texto completo para no saturar el objeto)
    sources = [
        {k: v for k, v in doc.items() if k != "text"}
        for doc in docs
    ]

    return ChatResponse(
        answer=answer,
        intent=intent_result.intent,
        risk_level=risk_result.level,
        action=risk_result.action,
        risk_flags=risk_result.flags,
        sources=sources,
        reasoning=risk_result.reasoning,
        tokens_used=tokens_used,
        notified=notified,
    )


def _fallback_answer(risk: RiskResult) -> str:
    """Respuesta de emergencia si el LLM falla."""
    if risk.level == "high":
        return (
            "⚠️ He detectado posibles señales de alarma en tu mensaje. "
            "Por favor busca atención médica urgente de inmediato. "
            "No puedo brindarte más información ahora debido a un error técnico."
        )
    return (
        "Lo siento, tuve un problema técnico al generar la respuesta. "
        "Por favor intenta de nuevo. Si tienes alguna urgencia médica, "
        "contacta a tu médico o ve a urgencias."
    )
