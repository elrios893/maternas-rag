# Segundo entregable

Fecha de primer avance: 28 de mayo de 2026 → 31 de mayo de 2026
Estado: En progreso

Construcción de agente para maternas → Versión preliminar. Resultado final de este segundo entregable no comprende la versión final del desarrollo.

Con los siguientes datasets, LOS CUALES TAMBIEN SE ENCUENTRAN EN EL PC:

1. MEDMCQA [https://huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
2. MEDQA [https://www.kaggle.com/datasets/moaaztameer/medqa-usmle](https://www.kaggle.com/datasets/moaaztameer/medqa-usmle)
3. MULTICLINSUM [https://zenodo.org/records/15517617](https://zenodo.org/records/15517617)

Realizar lo siguiente

- Desarrollar un RAG con un modelo gratis por API (modelo aún por revisar)
    
    El rag debe hacer lo siguiente:
    
    - Recibir preguntas clinicas o educativas de usuarios
    - **CLASIFICAR** intencion y nivel de riesgo
    - La informacion que suministre debe ser desde los DATASETS, limitar el conocmiento del LLM a los datasets. Especificarle al rag en ‘system_prompt’ que toda respuesta sea contrastada en guias medicas
    - Generar respuestas fundamentadas, claras y con citas
    - **DETECTAR** signos de alarmas y recomendar **SIEMPRE** consultar al personal medico cuando corresponda.
    - **EVALUAR** metricas RAG y revision medica
    - El RAG debe ser expuesto como un chatbot, por lo que debe ser conversacional
- Para evaluar el modelo, se deberá realizar prompts de los siguientes temas y ser llevados a un informe para un triage:
    - Control prenatal
    - Sintomas durante el embarazo
    - Signos de alarmas
    - postparto
    - lactancia
    - medicamentos
    - Salud mental
    - Nutricion
    - Urgencia
    - Pregunta administrativa o educativa
    
    Para todos prompts y temas, el RAG DEBE **CLASIFICAR Y DETECTAR** el nivel de riesgo
    
    Niveles de riesgo:
    
    - Riesgo BAJO: pregunta informativa
    - Riesgo MEDIO: requiere recomendación de consulta
    - Riesgo ALTO: requiere indicar
- El agente debe tener skills, como por ejemplo, para la clasificacion de intencion:
    
    ```python
    {
      "skill": "intent_classification",
      "categories": [
        "control_prenatal",
        "signos_de_alarma",
        "sintomas_embarazo",
        "postparto",
        "lactancia",
        "salud_mental_perinatal",
        "medicamentos",
        "nutricion",
        "actividad_fisica",
        "planificacion_familiar",
        "consulta_administrativa",
        "pregunta_fuera_de_alcance"
      ]
    }
    ```
    
- El output del modelo debe tener salidas de tipo
    
    ```python
    {
      "intent": "signos_de_alarma",
      "confidence": 0.92,
      "reason": "La pregunta menciona cefalea intensa durante el embarazo."
    }
    ```
    
    ```python
    {
      "risk_level": "low | medium | high",
      "requires_escalation": true,
      "detected_alert_signs": [],
      "recommended_action": "educational_answer | medical_consultation | urgent_care"
    }
    ```
    

NOTAS: 
- Elegir modelo LLM via api para desarrollar RAG, exponer el desarrollo en una interfaz grafica interactuable. Revisar maneras de COMPARTIR/DESPLEGAR el desarollo.
- Por ahora, utilizar todos los datasets con todo su contenido

Optimización: QLORA, pero para el primer avance no es necesario.
