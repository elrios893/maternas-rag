"""
styles.py — CSS del panel Maternas.

Se inyecta una sola vez, en el entrypoint (app.py), antes del dispatch de
páginas — así aplica a todas ellas (chat, dashboard, documentos, etc.),
no solo a la que lo definía originalmente.
"""

STYLES = """
<style>
/* Burbuja usuario */
.msg-user {
    background: #e8f4fd;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    color: #1a1a2e;
}
/* Burbuja asistente */
.msg-assistant {
    background: #f0f7f0;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    color: #1a1a2e;
}
/* Burbuja clarificación */
.msg-clarification {
    background: #fff8e6;
    border: 1px solid #ffc107;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    color: #1a1a2e;
}
/* Badge de riesgo */
.badge-low    { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }
.badge-medium { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }
.badge-high   { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }
/* Pill de fuente */
.source-pill  { background:#e9ecef; color:#495057; padding:2px 8px; border-radius:8px; font-size:0.78em; margin:2px; display:inline-block; }
</style>
"""
