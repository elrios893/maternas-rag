"""
settings.py — Configuración central del proyecto.
Lee variables desde .env usando pydantic-settings.
Todas las rutas y parámetros del sistema se definen aquí.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # --- Groq ---
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_api_key_2: str = Field("", env="GROQ_API_KEY_2")   # key dedicada para Ragas judge
    groq_model: str = Field("llama-3.1-70b-versatile", env="GROQ_MODEL")

    # --- Embedding ---
    embedding_model: str = Field("intfloat/multilingual-e5-base", env="EMBEDDING_MODEL")
    embedding_device: str = Field("cpu", env="EMBEDDING_DEVICE")

    # --- FAISS ---
    faiss_store_path: Path = Field(Path("./faiss_store"), env="FAISS_STORE_PATH")

    # --- Datasets ---
    dataset_medmcqa_path: Path = Field(Path("./datasets/data"), env="DATASET_MEDMCQA_PATH")
    dataset_medqa_path: Path = Field(Path("./datasets/data_clean/data_clean"), env="DATASET_MEDQA_PATH")
    dataset_multiclinsum_path: Path = Field(
        Path("./datasets/multiclinsum_large-scale_train_es/multiclinsum_large-scale_train_es"),
        env="DATASET_MULTICLINSUM_PATH",
    )

    # --- RAG ---
    rag_top_k: int = Field(5, env="RAG_TOP_K")

    # --- Panel de administración ---
    # Token compartido que protege todos los endpoints /documents* y /admin*.
    # Si queda vacío, el panel se deshabilita por completo (fail-closed):
    # los endpoints responden 503 en vez de quedar abiertos.
    # Generar con:  python -c "import secrets; print(secrets.token_urlsafe(32))"
    admin_api_token: str = Field("", env="ADMIN_API_TOKEN")

    # --- Maternas API (consumida por la UI de Streamlit) ---
    api_url: str = Field("http://localhost:8080", env="API_URL")

    # --- Telegram Bot ---
    telegram_bot_token: str = Field("", env="TELEGRAM_BOT_TOKEN")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # --- OpenRouter ---
    openrouter_key: str = Field("", env="OPENROUTER_KEY")

    # --- Cerebras ---
    cerebras_key: str = Field("", env="CEREBRAS_KEY")

    # --- Notifier (Skill) ---
    notifier_enabled: bool = Field(True, env="NOTIFIER_ENABLED")
    notifier_email_to: str = Field("", env="NOTIFIER_EMAIL_TO")
    notifier_smtp_host: str = Field("smtp.gmail.com", env="NOTIFIER_SMTP_HOST")
    notifier_smtp_port: int = Field(587, env="NOTIFIER_SMTP_PORT")
    notifier_smtp_user: str = Field("", env="NOTIFIER_SMTP_USER")
    notifier_smtp_password: str = Field("", env="NOTIFIER_SMTP_PASSWORD")

    # --- Status Check Scheduler (integrado en maternas_bot.py via JobQueue) ---
    # Frecuencia de los mensajes automaticos de seguimiento, segun el nivel
    # de riesgo acumulado del usuario. Valores por defecto pensados para
    # desarrollo/pruebas — en produccion usar minutos u horas (ver .env.example).
    status_check_interval_low_seconds: float = Field(60.0, env="STATUS_CHECK_INTERVAL_LOW_SECONDS")
    status_check_interval_medium_seconds: float = Field(45.0, env="STATUS_CHECK_INTERVAL_MEDIUM_SECONDS")
    status_check_interval_high_seconds: float = Field(30.0, env="STATUS_CHECK_INTERVAL_HIGH_SECONDS")
    status_check_message: str = Field(
        "🩺 *Check de estado* — ¿Cómo te encuentras hoy? "
        "Cuéntame cualquier molestia o duda que tengas.",
        env="STATUS_CHECK_MESSAGE",
    )

    # --- Privacidad: cifrado del registro de usuarios activos del bot ---
    # active_users.json contiene chat_id de Telegram + nivel de riesgo — se
    # cifra en disco con Fernet. Generar con:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    active_users_encryption_key: str = Field("", env="ACTIVE_USERS_ENCRYPTION_KEY")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Instancia global — importar desde aquí en todos los módulos
settings = Settings()
