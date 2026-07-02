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

    # --- Status Check Scheduler (unificado en bot) ---
    # NOTA: Las vars *_minutes se mantienen por compatibilidad con .envs
    # existentes. El scheduler unificado usa las vars *_seconds definidas abajo.
    status_check_base_interval_minutes: float = Field(60.0, env="STATUS_CHECK_BASE_INTERVAL_MINUTES")
    status_check_min_interval_minutes: float = Field(5.0, env="STATUS_CHECK_MIN_INTERVAL_MINUTES")
    status_check_message: str = Field(
        "🩺 *Check de estado* — ¿Cómo te encuentras hoy? "
        "Cuéntame cualquier molestia o duda que tengas.",
        env="STATUS_CHECK_MESSAGE",
    )

    # --- Status Check Scheduler (unificado — intervalos por nivel de riesgo) ---
    # Valores bajos para desarrollo/pruebas. En producción subir a minutos.
    status_check_interval_low_seconds: float = Field(60.0, env="STATUS_CHECK_INTERVAL_LOW_SECONDS")
    status_check_interval_medium_seconds: float = Field(45.0, env="STATUS_CHECK_INTERVAL_MEDIUM_SECONDS")
    status_check_interval_high_seconds: float = Field(30.0, env="STATUS_CHECK_INTERVAL_HIGH_SECONDS")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Instancia global — importar desde aquí en todos los módulos
settings = Settings()
