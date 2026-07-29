"""Cấu hình ứng dụng HealthSense AI Service."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Config
    APP_NAME: str = os.getenv("APP_NAME", "HealthSense AI Service")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # RabbitMQ Config
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "")
    RABBITMQ_QUEUE: str = os.getenv("RABBITMQ_QUEUE", "health.record.processing.queue")

    # AWS S3 / MinIO Config (Không hardcode credentials trên mã nguồn để đảm bảo bảo mật tuyệt đối)
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "")
    AWS_S3_REGION: str = os.getenv("AWS_S3_REGION", "")
    AWS_S3_ENDPOINT_URL: str = os.getenv("AWS_S3_ENDPOINT_URL", "")

    # Core Service Webhook URL
    CORE_CALLBACK_URL: str = os.getenv("CORE_CALLBACK_URL", "")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
