from __future__ import annotations

from unittest.mock import patch

import pytest

from src.classifiers.intent_classifier import (
    VALID_INTENTS,
    _fallback_intent,
    _parse_response,
    classify_intent,
)


class TestFallbackIntent:
    def test_signos_de_alarma_detected(self):
        msg = "tengo una hemorragia muy fuerte"
        assert _fallback_intent(msg) == "signos_de_alarma"

    def test_sintomas_embarazo_detected(self):
        msg = "tengo náuseas por las mañanas"
        assert _fallback_intent(msg) == "sintomas_embarazo"

    def test_lactancia_detected(self):
        msg = "me duele el pecho al amamantar"
        assert _fallback_intent(msg) == "lactancia"

    def test_medicamentos_detected(self):
        msg = "¿puedo tomar este medicamento?"
        assert _fallback_intent(msg) == "medicamentos"

    def test_nutricion_detected(self):
        msg = "¿qué alimentos debo comer?"
        assert _fallback_intent(msg) == "nutricion"

    def test_fallback_to_out_of_scope(self):
        msg = "¿cómo cambio una llanta?"
        assert _fallback_intent(msg) == "pregunta_fuera_de_alcance"

    def test_empty_message(self):
        assert _fallback_intent("") == "pregunta_fuera_de_alcance"

    @pytest.mark.parametrize("msg,expected", [
        ("convulsión", "signos_de_alarma"),
        ("desmayo", "signos_de_alarma"),
        ("no se mueve", "signos_de_alarma"),
        ("sangrado", "signos_de_alarma"),
    ])
    def test_parametric_alarm_keywords(self, msg, expected):
        assert _fallback_intent(msg) == expected

    @pytest.mark.parametrize("msg,expected", [
        ("tengo vómito", "sintomas_embarazo"),
        ("hinchazón en los pies", "sintomas_embarazo"),
        ("mucho cansancio", "sintomas_embarazo"),
    ])
    def test_parametric_symptom_keywords(self, msg, expected):
        assert _fallback_intent(msg) == expected

    @pytest.mark.parametrize("msg,expected", [
        ("duele el pecho al dar leche", "lactancia"),
        ("amamantar con mastitis", "lactancia"),
        ("lactar es muy doloroso", "lactancia"),
    ])
    def test_parametric_lactation_keywords(self, msg, expected):
        assert _fallback_intent(msg) == expected


class TestParseResponse:
    def test_valid_json(self):
        raw = '{"intent": "control_prenatal", "confidence": 0.95, "reasoning": "Pregunta sobre ecografía."}'
        result = _parse_response(raw, "¿cuándo me hago la ecografía?")
        assert result.intent == "control_prenatal"
        assert result.confidence == 0.95
        assert result.reasoning == "Pregunta sobre ecografía."
        assert result.raw == raw

    def test_unknown_intent_fallback(self):
        raw = '{"intent": "cardiologia", "confidence": 0.9, "reasoning": "test"}'
        result = _parse_response(raw, "test message")
        assert result.intent in VALID_INTENTS
        assert result.confidence == 0.3

    def test_confidence_clamped_above(self):
        raw = '{"intent": "lactancia", "confidence": 1.5, "reasoning": "test"}'
        result = _parse_response(raw, "test")
        assert result.confidence == 1.0

    def test_confidence_clamped_below(self):
        raw = '{"intent": "lactancia", "confidence": -0.5, "reasoning": "test"}'
        result = _parse_response(raw, "test")
        assert result.confidence == 0.0

    def test_invalid_json_fallback(self):
        raw = "broken json"
        result = _parse_response(raw, "test")
        assert result.confidence == 0.2
        assert result.raw == raw

    def test_regex_fallback_for_malformed_json(self):
        raw = 'some text {"intent": "nutricion", "confidence": 0.5} trailing'
        result = _parse_response(raw, "¿qué puedo comer?")
        assert result.intent == "nutricion"
        assert result.confidence == 0.4

    def test_regex_fallback_invalid_intent(self):
        raw = '{"intent": "invalid_intent_here", "confidence": 0.5}'
        result = _parse_response(raw, "test")
        assert result.intent in VALID_INTENTS
        assert result.confidence == 0.3

    @pytest.mark.parametrize("raw,intent", [
        ('{"intent": "control_prenatal", "confidence": 0.8, "reasoning": "a"}', "control_prenatal"),
        ('{"intent": "signos_de_alarma", "confidence": 0.9, "reasoning": "b"}', "signos_de_alarma"),
        ('{"intent": "postparto", "confidence": 0.7, "reasoning": "c"}', "postparto"),
        ('{"intent": "salud_mental_perinatal", "confidence": 0.6, "reasoning": "d"}', "salud_mental_perinatal"),
        ('{"intent": "planificacion_familiar", "confidence": 0.8, "reasoning": "e"}', "planificacion_familiar"),
        ('{"intent": "consulta_administrativa", "confidence": 0.5, "reasoning": "f"}', "consulta_administrativa"),
        ('{"intent": "actividad_fisica", "confidence": 0.7, "reasoning": "g"}', "actividad_fisica"),
    ])
    def test_parametric_valid_intents(self, raw, intent):
        result = _parse_response(raw, "test")
        assert result.intent == intent


class TestClassifyIntent:
    def test_empty_message(self):
        result = classify_intent("")
        assert result.intent == "pregunta_fuera_de_alcance"
        assert result.confidence == 1.0

    def test_whitespace_message(self):
        result = classify_intent("   ")
        assert result.intent == "pregunta_fuera_de_alcance"
        assert result.confidence == 1.0

    def test_none_message(self):
        result = classify_intent(None)
        assert result.intent == "pregunta_fuera_de_alcance"

    def test_successful_classification(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value.choices[0].message.content = (
            '{"intent": "lactancia", "confidence": 0.92, "reasoning": "Pregunta sobre amamantamiento."}'
        )
        result = classify_intent("dolor al amamantar")
        assert result.intent == "lactancia"
        assert result.confidence == 0.92
        mock_groq_client.chat.completions.create.assert_called_once()

    def test_llm_error_fallback(self, mock_groq_client):
        mock_groq_client.chat.completions.create.side_effect = Exception("API error")
        result = classify_intent("dolor al amamantar")
        assert result.intent == "pregunta_fuera_de_alcance"
        assert result.confidence == 0.0
        assert "Error en clasificación" in result.reasoning

    def test_with_conversation_history(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value.choices[0].message.content = (
            '{"intent": "control_prenatal", "confidence": 0.85, "reasoning": "Pregunta sobre seguimiento."}'
        )
        history = [{"role": "assistant", "content": "Te recomendamos control prenatal mensual."}]
        result = classify_intent("¿cada cuánto debo ir?", conversation_history=history)
        assert result.intent == "control_prenatal"

    def test_invalid_intent_from_llm_fallback(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value.choices[0].message.content = (
            '{"intent": "invalid_intent_xyz", "confidence": 0.9, "reasoning": "test"}'
        )
        result = classify_intent("dolor al amamantar")
        assert result.intent in VALID_INTENTS
        assert result.confidence == 0.3
