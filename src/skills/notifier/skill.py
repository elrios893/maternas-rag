from src.skills import Skill, ToolSpec
from src.skills.notifier.tool import notify_risk


class NotifierSkill(Skill):
    name = "notifier"
    description = "Notificaciones por email SMTP ante riesgo clinico alto o medio"

    tools = [
        ToolSpec(
            name="notify_risk",
            description=(
                "Envia una alerta por email cuando se detecta riesgo clinico "
                "(high siempre, medium solo si LLM determina que amerita). "
                "Incluye el mensaje del usuario, nivel de riesgo, intencion "
                "y razonamiento del clasificador."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query":       {"type": "string", "description": "Mensaje del usuario"},
                    "risk_level":  {"type": "string", "enum": ["high", "medium"]},
                    "intent":      {"type": "string", "description": "Intencion clasificada"},
                    "reasoning":   {"type": "string", "description": "Razonamiento del detector"},
                    "flags":       {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query", "risk_level", "intent", "reasoning"],
            },
            required=["query", "risk_level", "intent", "reasoning"],
            fn=notify_risk,
        ),
    ]
