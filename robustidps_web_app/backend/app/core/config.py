"""
Application Configuration
==========================

Load and validate environment variables and application settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # ==========================================
    # Application
    # ==========================================
    APP_NAME: str = "RobustIDPS.ai"
    APP_ENV: str = "production"
    DEBUG: bool = False
    SECRET_KEY: str
    API_VERSION: str = "v1"

    # URLs
    DOMAIN: str = "robustidps.ai"
    FRONTEND_URL: str = "https://robustidps.ai"
    BACKEND_URL: str = "https://robustidps.ai/api"
    WEBSOCKET_URL: str = "wss://robustidps.ai/ws"

    # ==========================================
    # Database
    # ==========================================
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ==========================================
    # Redis
    # ==========================================
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    CACHE_TTL: int = 300

    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ==========================================
    # AI Models
    # ==========================================
    MODEL_CHECKPOINT_PATH: str = "/app/models/unified_ids.pt"
    ENABLE_GPU: bool = True
    GPU_DEVICE: int = 0
    BATCH_SIZE: int = 32
    CONFIDENCE_THRESHOLD: float = 0.85

    ENABLE_TORCHSCRIPT: bool = True
    ENABLE_QUANTIZATION: bool = False
    QUANTIZATION_TYPE: str = "int8"

    # ==========================================
    # Traffic Capture
    # ==========================================
    CAPTURE_INTERFACE: str = "eth0"
    CAPTURE_BUFFER_SIZE: int = 65536
    PCAP_FILTER: str = ""
    MAX_PACKET_SIZE: int = 1518

    ENABLE_PACKET_CAPTURE: bool = True
    CAPTURE_MODE: str = "passive"  # passive, inline, agent
    PROCESSING_THREADS: int = 4

    # ==========================================
    # Detection Engine
    # ==========================================
    DETECTION_WORKERS: int = 4
    MAX_QUEUE_SIZE: int = 10000
    PROCESSING_TIMEOUT: int = 5000

    ENABLED_MODELS: str = "neural_ode,optimal_transport,encrypted_traffic,bayesian"

    @property
    def ENABLED_MODELS_LIST(self) -> List[str]:
        return [m.strip() for m in self.ENABLED_MODELS.split(",")]

    ENABLE_REALTIME: bool = True
    REALTIME_WINDOW_SIZE: int = 100
    STREAMING_BATCH_SIZE: int = 16

    # ==========================================
    # Prevention & Firewall
    # ==========================================
    ENABLE_FIREWALL: bool = True
    FIREWALL_BACKEND: str = "iptables"
    AUTO_BLOCK: bool = True
    BLOCK_DURATION: int = 3600

    WHITELIST_IPS: str = "127.0.0.1,10.0.0.0/8,192.168.0.0/16"

    @property
    def WHITELIST_IPS_LIST(self) -> List[str]:
        return [ip.strip() for ip in self.WHITELIST_IPS.split(",")]

    ENABLE_GEO_BLOCKING: bool = False
    BLOCKED_COUNTRIES: str = ""

    # ==========================================
    # Alerts
    # ==========================================
    # Email
    ENABLE_EMAIL_ALERTS: bool = True
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USE_TLS: bool = True
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    ALERT_FROM_EMAIL: str = ""
    ALERT_TO_EMAILS: str = ""

    # Slack
    ENABLE_SLACK_ALERTS: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None

    # Teams
    ENABLE_TEAMS_ALERTS: bool = False
    TEAMS_WEBHOOK_URL: Optional[str] = None

    # SMS
    ENABLE_SMS_ALERTS: bool = False
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_FROM_NUMBER: Optional[str] = None
    ALERT_TO_NUMBERS: str = ""

    # Webhooks
    ENABLE_WEBHOOK_ALERTS: bool = False
    WEBHOOK_URL: Optional[str] = None
    WEBHOOK_SECRET: Optional[str] = None

    # Thresholds
    ALERT_MIN_CONFIDENCE: float = 0.90
    ALERT_RATE_LIMIT: int = 100
    ALERT_DEDUPLICATION_WINDOW: int = 60

    # ==========================================
    # Authentication
    # ==========================================
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    ENABLE_OAUTH: bool = False
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None

    ENABLE_MFA: bool = False
    MFA_ISSUER: str = "RobustIDPS.ai"

    SESSION_TIMEOUT: int = 3600

    # ==========================================
    # Logging & Monitoring
    # ==========================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "/var/log/robustidps/app.log"
    ENABLE_LOG_ROTATION: bool = True
    LOG_MAX_BYTES: int = 10485760
    LOG_BACKUP_COUNT: int = 10

    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090

    ENABLE_TRACING: bool = False
    JAEGER_AGENT_HOST: str = "jaeger"
    JAEGER_AGENT_PORT: int = 6831

    # ==========================================
    # Performance
    # ==========================================
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000

    CELERY_WORKERS: int = 4

    @property
    def CELERY_BROKER_URL(self) -> str:
        return self.REDIS_URL

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return self.REDIS_URL

    # ==========================================
    # Storage
    # ==========================================
    PCAP_STORAGE_PATH: str = "/data/pcaps"
    PCAP_RETENTION_DAYS: int = 7
    MAX_PCAP_SIZE_GB: int = 100

    ALERT_RETENTION_DAYS: int = 90

    ENABLE_AUTO_BACKUP: bool = True
    BACKUP_SCHEDULE: str = "0 2 * * *"
    BACKUP_PATH: str = "/backups"
    BACKUP_RETENTION_DAYS: int = 30

    # ==========================================
    # Integration
    # ==========================================
    ENABLE_THREAT_INTEL: bool = False
    ALIENVAULT_API_KEY: Optional[str] = None
    VIRUSTOTAL_API_KEY: Optional[str] = None
    SHODAN_API_KEY: Optional[str] = None

    GEOIP_DATABASE_PATH: str = "/data/geoip/GeoLite2-City.mmdb"
    ENABLE_GEOIP: bool = True

    # ==========================================
    # Development
    # ==========================================
    ENABLE_CORS: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    ENABLE_SWAGGER: bool = True
    ENABLE_DEBUG_TOOLBAR: bool = False

    TEST_DATABASE_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
