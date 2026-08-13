"""
consent.py — Aviso de tratamiento de información, compartido entre la UI
de Streamlit (src/ui/app.py) y el bot de Telegram (src/bot/maternas_bot.py).

Debe mostrarse al inicio de cada sesión nueva, antes de permitir cualquier
conversación, y requiere aceptación explícita del usuario. Si el usuario no
acepta, la sesión no procesa mensajes; si vuelve a escribir, se le muestra
el aviso nuevamente.

Texto en plano (sin markdown) para que se renderice igual en Streamlit y en
Telegram, evitando errores de parseo de entidades en Telegram.
"""

CONSENT_TEXT = (
    "⚠️ AVISO SOBRE TRATAMIENTO DE INFORMACIÓN\n\n"
    "Este chatbot es un proyecto de INVESTIGACIÓN en FASE EXPERIMENTAL. Usa "
    "inteligencia artificial y NO SUSTITUYE la orientación de un profesional "
    "competente.\n\n"
    "Podemos conservar tu NOMBRE PREFERIDO O ALIAS y un CÓDIGO INTERNO de "
    "identificación. NO escribas tu nombre legal completo, número de "
    "documento, dirección, contraseñas ni datos financieros.\n\n"
    "Tu conversación se procesa para mantener el CONTEXTO del diálogo y los "
    "fines del proyecto de investigación. Si se usa en análisis o "
    "publicaciones, se aplican medidas de ANONIMIZACIÓN o SEUDONIMIZACIÓN.\n\n"
    "Tu participación es VOLUNTARIA: puedes NO responder preguntas sensibles "
    "y puedes SOLICITAR EL RETIRO de tu información en cualquier momento.\n\n"
    "Este es un PROTOTIPO DE INVESTIGACIÓN — aún no apto para uso clínico, "
    "comercial o productivo real.\n\n"
    "¿Aceptas continuar bajo estas condiciones?"
)

ACCEPTED_TEXT = "✅ Gracias — aceptaste el aviso. Ya puedes escribirme tu pregunta."

FAREWELL_TEXT = (
    "Entendido — no se procesará tu conversación. Sesión finalizada. 👋\n"
    "Si cambias de opinión, escríbeme de nuevo y volveré a mostrarte este aviso."
)
