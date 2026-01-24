"""
Database layer for persistent settings storage.

Uses async SQLAlchemy with MySQL/MariaDB for storing service settings
that can be modified at runtime without service restart.
"""

from __future__ import annotations

import os
import json
import logging
from typing import AsyncIterator, Optional, Any, Dict, Protocol
from contextlib import asynccontextmanager

from sqlalchemy import String, select, delete, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from dotenv import load_dotenv
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

__all__ = [
    "Base",
    "ServiceSetting",
    "DatabaseManager",
    "SettingsRepository",
    "SUPPORTED_KEYS",
]

load_dotenv()

# =============================================================================
# ORM Models
# =============================================================================

class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class ServiceSetting(Base):
    """
    Stores key-value settings per service.

    Allows multiple services to share the same settings table
    with service-scoped keys.
    """
    __tablename__ = "service_settings"

    service: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(String(2048), nullable=False)

    def __repr__(self) -> str:
        return f"<ServiceSetting {self.service}.{self.key}={self.value[:50]}>"


# =============================================================================
# Database Connection Manager
# =============================================================================

class DatabaseManager:
    """
    Manages async database connections with proper lifecycle handling.

    Usage:
        db = DatabaseManager()
        await db.initialize()
        async with db.session() as session:
            # use session
        await db.close()
    """

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self._engine is not None

    def get_database_url(self) -> str:
        """Get database URL from environment."""
        url = os.getenv("EMBEDDING_DB_URL")
        if not url:
            raise RuntimeError(
                "EMBEDDING_DB_URL is required for database settings. "
                "Example: mysql+aiomysql://user:pass@host:3306/db"
            )
        return url

    async def initialize(self) -> None:
        """Initialize database connection and ensure schema exists."""
        if self._initialized:
            return

        try:
            url = self.get_database_url()
            self._engine = create_async_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=5,
                max_overflow=10,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
            )
            self._sessionmaker = async_sessionmaker(
                self._engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )

            # Test connection and create tables
            async with self._engine.begin() as conn:
                # Test connection
                await conn.execute(text("SELECT 1"))
                # Create tables
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self._engine = None
            self._sessionmaker = None
            raise

    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
            self._initialized = False
            logger.info("Database connections closed")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get a database session with automatic cleanup."""
        if not self._initialized or self._sessionmaker is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self._sessionmaker() as session:
            try:
                yield session
            except SQLAlchemyError as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise

    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        if not self.is_initialized:
            return {"status": "not_initialized", "connected": False}

        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"status": "healthy", "connected": True}
        except Exception as e:
            return {"status": "unhealthy", "connected": False, "error": str(e)}


# Global database manager instance
db_manager = DatabaseManager()


# Legacy compatibility - yields sessions from global manager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Get a database session (legacy interface)."""
    async with db_manager.session() as session:
        yield session


# =============================================================================
# Settings Repository
# =============================================================================

# Keys that can be stored/retrieved from database
SUPPORTED_KEYS = frozenset({
    "host", "port", "model_name", "model_cache_dir",
    "device", "normalize_embeddings", "max_batch_size", "cors_origins",
})


class _SettingsLike(Protocol):
    """Protocol for settings-like objects."""

    def model_dump(self) -> Dict[str, Any]: ...


def _parse_value(key: str, raw: str) -> Any:
    """Parse string value to appropriate Python type."""
    if key in {"port", "max_batch_size"}:
        return int(raw)
    if key in {"normalize_embeddings"}:
        return str(raw).lower() in {"1", "true", "yes", "on"}
    if key in {"cors_origins"}:
        s = raw.strip()
        if s.startswith("["):
            return json.loads(s)
        return [x.strip() for x in s.split(",") if x.strip()]
    return raw


def _stringify_value(value: Any) -> str:
    """Convert Python value to string for storage."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class SettingsRepository:
    """
    Repository for CRUD operations on service settings.

    Follows repository pattern to abstract database operations
    from business logic.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name

    async def get_all(self, session: AsyncSession) -> Dict[str, str]:
        """Get all settings for this service as raw strings."""
        stmt = select(ServiceSetting).where(
            ServiceSetting.service == self.service_name
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return {r.key: r.value for r in rows}

    async def get_typed(
            self,
            session: AsyncSession,
            defaults: _SettingsLike
    ) -> Any:
        """
        Get settings merged with defaults, with proper type conversion.

        Args:
            session: Database session
            defaults: Default settings object (provides types and fallbacks)

        Returns:
            New settings object of same type as defaults
        """
        data = defaults.model_dump()
        stored = await self.get_all(session)

        for key, raw_value in stored.items():
            if key in SUPPORTED_KEYS:
                try:
                    data[key] = _parse_value(key, raw_value)
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse setting {key}={raw_value}: {e}")
                    # Keep default value

        return type(defaults)(**data)

    async def get(self, session: AsyncSession, key: str) -> Optional[str]:
        """Get a single setting value."""
        if key not in SUPPORTED_KEYS:
            raise ValueError(f"Unsupported setting key: {key}")

        obj = await session.get(
            ServiceSetting,
            {"service": self.service_name, "key": key}
        )
        return obj.value if obj else None

    async def upsert(self, session: AsyncSession, key: str, value: Any) -> None:
        """Create or update a setting."""
        if key not in SUPPORTED_KEYS:
            raise ValueError(f"Unsupported setting key: {key}. Supported: {SUPPORTED_KEYS}")

        str_value = _stringify_value(value)
        obj = await session.get(
            ServiceSetting,
            {"service": self.service_name, "key": key}
        )

        if obj is None:
            obj = ServiceSetting(
                service=self.service_name,
                key=key,
                value=str_value
            )
            session.add(obj)
            logger.info(f"Created setting: {self.service_name}.{key}")
        else:
            obj.value = str_value
            logger.info(f"Updated setting: {self.service_name}.{key}")

        await session.commit()

    async def bulk_upsert(self, session: AsyncSession, settings_dict: Dict[str, Any]) -> None:
        """Create or update multiple settings in a single transaction."""
        for key, value in settings_dict.items():
            if key not in SUPPORTED_KEYS:
                continue  # Or raise ValueError if you prefer strict validation

            str_value = _stringify_value(value)
            obj = await session.get(
                ServiceSetting,
                {"service": self.service_name, "key": key}
            )

            if obj is None:
                obj = ServiceSetting(
                    service=self.service_name,
                    key=key,
                    value=str_value
                )
                session.add(obj)
            else:
                obj.value = str_value

        await session.commit()
        logger.info(f"Bulk updated {len(settings_dict)} settings for {self.service_name}")

    async def delete(self, session: AsyncSession, key: str) -> bool:
        """Delete a setting. Returns True if setting existed."""
        if key not in SUPPORTED_KEYS:
            raise ValueError(f"Unsupported setting key: {key}")

        stmt = (
            delete(ServiceSetting)
            .where(ServiceSetting.service == self.service_name)
            .where(ServiceSetting.key == key)
        )
        result = await session.execute(stmt)
        await session.commit()

        deleted = (result.rowcount or 0) > 0
        if deleted:
            logger.info(f"Deleted setting: {self.service_name}.{key}")
        return deleted

    async def delete_all(self, session: AsyncSession) -> int:
        """Delete all settings for this service. Returns count deleted."""
        stmt = delete(ServiceSetting).where(
            ServiceSetting.service == self.service_name
        )
        result = await session.execute(stmt)
        await session.commit()
        count = result.rowcount or 0
        logger.info(f"Deleted {count} settings for {self.service_name}")
        return count