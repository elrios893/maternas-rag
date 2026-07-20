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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Instancia global — importar desde aquí en todos los módulos
settings = Settings()
