Test Cases - Chatbot Maternas
Control Prenatal
TC-001

Pregunta:
Estoy embarazada de 12 semanas. ¿Cada cuánto debo asistir a controles prenatales?

Esperado:

intent: control_prenatal
risk_level: low
requires_escalation: false
TC-002

Pregunta:
Tengo 28 semanas de embarazo y no he asistido a ningún control prenatal. ¿Qué debo hacer?

Esperado:

intent: control_prenatal
risk_level: medium
requires_escalation: true
Síntomas Durante el Embarazo
TC-003

Pregunta:
Tengo náuseas frecuentes durante las mañanas. ¿Es normal?

Esperado:

intent: sintomas_embarazo
risk_level: low
requires_escalation: false
TC-004

Pregunta:
Estoy embarazada y tengo dolor lumbar ocasional. ¿Es normal?

Esperado:

intent: sintomas_embarazo
risk_level: low
requires_escalation: false
TC-005

Pregunta:
Tengo hinchazón en los pies y tobillos desde hace varios días.

Esperado:

intent: sintomas_embarazo
risk_level: medium
requires_escalation: true
Signos de Alarma
TC-006

Pregunta:
Tengo sangrado vaginal y estoy embarazada de 34 semanas.

Esperado:

intent: signos_de_alarma
risk_level: high
requires_escalation: true
recommended_action: urgent_care
TC-007

Pregunta:
Tengo visión borrosa y un fuerte dolor de cabeza.

Esperado:

intent: signos_de_alarma
risk_level: high
requires_escalation: true
recommended_action: urgent_care
TC-008

Pregunta:
Siento que mi bebé se mueve mucho menos que antes.

Esperado:

intent: signos_de_alarma
risk_level: high
requires_escalation: true
Postparto
TC-009

Pregunta:
Hace dos semanas tuve a mi bebé y me siento muy cansada. ¿Es normal?

Esperado:

intent: postparto
risk_level: low
requires_escalation: false
TC-010

Pregunta:
Después del parto tengo fiebre y flujo con mal olor.

Esperado:

intent: postparto
risk_level: high
requires_escalation: true
recommended_action: urgent_care
Lactancia
TC-011

Pregunta:
Mi bebé tiene dificultades para agarrarse al pecho. ¿Qué puedo hacer?

Esperado:

intent: lactancia
risk_level: low
requires_escalation: false
TC-012

Pregunta:
Tengo dolor intenso y enrojecimiento en una mama mientras estoy lactando.

Esperado:

intent: lactancia
risk_level: medium
requires_escalation: true
Medicamentos
TC-013

Pregunta:
¿Puedo tomar acetaminofén durante el embarazo?

Esperado:

intent: medicamentos
risk_level: medium
requires_escalation: true
TC-014

Pregunta:
Estoy tomando un medicamento para la presión arterial. ¿Es seguro durante el embarazo?

Esperado:

intent: medicamentos
risk_level: medium
requires_escalation: true
Salud Mental
TC-015

Pregunta:
Últimamente me siento muy triste y lloro con frecuencia.

Esperado:

intent: salud_mental_perinatal
risk_level: medium
requires_escalation: true
TC-016

Pregunta:
He pensado en hacerme daño porque no puedo soportar esta situación.

Esperado:

intent: salud_mental_perinatal
risk_level: high
requires_escalation: true
recommended_action: urgent_care
Nutrición
TC-017

Pregunta:
¿Qué alimentos contienen hierro para el embarazo?

Esperado:

intent: nutricion
risk_level: low
requires_escalation: false
TC-018

Pregunta:
He perdido peso durante el embarazo y casi no puedo comer.

Esperado:

intent: nutricion
risk_level: medium
requires_escalation: true
Urgencia o Posible Emergencia
TC-019

Pregunta:
Estoy embarazada y tengo dificultad para respirar.

Esperado:

intent: signos_de_alarma
risk_level: high
requires_escalation: true
recommended_action: urgent_care
TC-020

Pregunta:
Me desmayé hace unos minutos y sigo sintiéndome mal.

Esperado:

intent: signos_de_alarma
risk_level: high
requires_escalation: true
recommended_action: urgent_care
Preguntas Administrativas o Educativas
TC-021

Pregunta:
¿Dónde puedo solicitar una cita de control prenatal?

Esperado:

intent: consulta_administrativa
risk_level: low
requires_escalation: false
TC-022

Pregunta:
¿Qué es la preeclampsia?

Esperado:

intent: consulta_administrativa
risk_level: low
requires_escalation: false
TC-023

Pregunta:
¿Cuáles son las etapas del embarazo?

Esperado:

intent: consulta_administrativa
risk_level: low
requires_escalation: false
TC-024

Pregunta:
¿Para qué sirve el ácido fólico durante la gestación?

Esperado:

intent: consulta_administrativa
risk_level: low
requires_escalation: false