"""
Configuration for Chat Service with MySQL Database.
"""

import os
import threading
from pathlib import Path
from typing import List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

__all__ = ["Settings", "SettingsManager", "settings_manager", "load_settings_from_env"]


class Settings(BaseModel):
    """Immutable service settings."""

    # Service identity
    service_name: str = "chat-service"
    service_version: str = "1.0.0"

    # Network
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)

    # =========================================================================
    # MySQL Database
    # =========================================================================
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="root")
    mysql_password: str = Field(default="")
    mysql_database: str = Field(default="chat_service")
    mysql_pool_min: int = Field(default=5, ge=1, le=50)
    mysql_pool_max: int = Field(default=20, ge=5, le=100)

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    llm_provider: Literal["openai", "gemini"] = Field(default="openai")

    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=2048, ge=100, le=16000)

    # Gemini settings
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")

    # =========================================================================
    # Search Service Configuration
    # =========================================================================
    search_service_url: str = Field(default="http://localhost:8002")
    search_timeout_seconds: int = Field(default=30, ge=5, le=120)

    # =========================================================================
    # RAG Configuration
    # =========================================================================
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_context_chunks: int = Field(default=5, ge=1, le=10)
    max_context_tokens: int = Field(default=4000, ge=500, le=12000)

    # =========================================================================
    # Response Generation
    # =========================================================================
    require_citations: bool = Field(default=True)
    allow_no_context_response: bool = Field(default=False)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # =========================================================================
    # Session Management
    # =========================================================================
    max_history_turns: int = Field(default=10, ge=0, le=50)
    session_timeout_minutes: int = Field(default=30, ge=5, le=1440)

    # =========================================================================
    # API Configuration
    # =========================================================================
    cors_origins: List[str] = Field(default=["*"])
    enable_streaming: bool = Field(default=True)

    class Config:
        frozen = True


def load_settings_from_env() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        # Service
        host=os.getenv("CHAT_HOST", "0.0.0.0"),
        port=int(os.getenv("CHAT_PORT", "8000")),

        # MySQL
        mysql_host=os.getenv("MYSQL_HOST", "localhost"),
        mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
        mysql_user=os.getenv("MYSQL_USER", "root"),
        mysql_password=os.getenv("MYSQL_PASSWORD", ""),
        mysql_database=os.getenv("MYSQL_DATABASE", "chat_service"),
        mysql_pool_min=int(os.getenv("MYSQL_POOL_MIN", "5")),
        mysql_pool_max=int(os.getenv("MYSQL_POOL_MAX", "20")),

        # LLM
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),

        # Search
        search_service_url=os.getenv("SEARCH_SERVICE_URL", "http://localhost:8002"),
        search_timeout_seconds=int(os.getenv("SEARCH_TIMEOUT_SECONDS", "30")),

        # RAG
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        retrieval_score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3")),
        max_context_chunks=int(os.getenv("MAX_CONTEXT_CHUNKS", "5")),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4000")),

        # Response
        require_citations=os.getenv("REQUIRE_CITATIONS", "true").lower() == "true",
        allow_no_context_response=os.getenv("ALLOW_NO_CONTEXT_RESPONSE", "false").lower() == "true",
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),

        # Session
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "10")),
        session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")),

        # API
        cors_origins=[
            x.strip()
            for x in os.getenv("CORS_ORIGINS", "*").split(",")
            if x.strip()
        ],
        enable_streaming=os.getenv("ENABLE_STREAMING", "true").lower() == "true",
    )


class SettingsManager:
    """Thread-safe settings manager."""

    def __init__(self, initial: Optional[Settings] = None):
        self._settings: Settings = initial or load_settings_from_env()
        self._lock = threading.RLock()

    @property
    def current(self) -> Settings:
        with self._lock:
            return self._settings

    def update(self, new_settings: Settings) -> Settings:
        with self._lock:
            old = self._settings
            self._settings = new_settings
            return old


settings_manager = SettingsManager()


class _SettingsProxy:
    def __getattr__(self, name: str):
        return getattr(settings_manager.current, name)

    def model_dump(self):
        return settings_manager.current.model_dump()


settings = _SettingsProxy()