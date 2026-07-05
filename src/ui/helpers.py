"""
helpers.py — Funciones de formateo para la interfaz Streamlit.

Independientes de cualquier vista. Reutilizables desde cualquier módulo.
"""


def risk_badge(level: str) -> str:
    labels = {"low": "🟢 Bajo", "medium": "🟡 Medio", "high": "🔴 ALTO"}
    css    = {"low": "badge-low", "medium": "badge-medium", "high": "badge-high"}
    label  = labels.get(level, level)
    klass  = css.get(level, "badge-low")
    return f'<span class="{klass}">{label}</span>'


def intent_label(intent: str) -> str:
    labels = {
        "control_prenatal":       "📅 Control prenatal",
        "signos_de_alarma":       "🚨 Signos de alarma",
        "sintomas_embarazo":      "🤰 Síntomas embarazo",
        "postparto":              "👶 Postparto",
        "lactancia":              "🍼 Lactancia",
        "salud_mental_perinatal": "💙 Salud mental",
        "medicamentos":           "💊 Medicamentos",
        "nutricion":              "🥗 Nutrición",
        "actividad_fisica":       "🏃 Actividad física",
        "planificacion_familiar": "📋 Planificación familiar",
        "consulta_administrativa":"📂 Administrativa",
        "pregunta_fuera_de_alcance": "❓ Fuera de alcance",
    }
    return labels.get(intent, intent)


def source_dataset_label(ds: str) -> str:
    labels = {
        "medmcqa":               "MedMCQA",
        "medqa_us":              "MedQA-US",
        "medqa_taiwan":          "MedQA-TW",
        "medqa_mainland":        "MedQA-ML",
        "multiclinsum_summary":  "Caso clínico (resumen)",
        "multiclinsum_fulltext": "Caso clínico (texto)",
        "textbook":              "Textbook médico",
    }
    return labels.get(ds, ds)
