"""
MySQL Database Module for Chat Service.

Schema Design:
- sessions: Conversation sessions
- messages: Individual messages (user + assistant)
- message_chunks: Chunks retrieved for each message (for UI display)
- message_citations: Chunks actually cited in the answer
- session_documents: Documents in scope for a session

Key Design Decision:
- Each MESSAGE has its own retrieved chunks (not per session)
- This allows the frontend to show "Sources used" for each response
- Citations link the answer [1], [2] markers to specific chunks
"""

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from enum import Enum

import aiomysql
from aiomysql import Pool

logger = logging.getLogger(__name__)

__all__ = [
    "DatabaseManager",
    "SessionRepository",
    "MessageRepository",
    "get_database_manager",
    "set_database_manager",
]

# =============================================================================
# SQL Schema
# =============================================================================

SCHEMA_SQL = """
             -- Sessions table: Conversation sessions
             CREATE TABLE IF NOT EXISTS sessions \
             ( \
                 id \
                 VARCHAR \
             ( \
                 36 \
             ) PRIMARY KEY,

                 -- User info (optional, for multi-user)
                 user_id VARCHAR \
             ( \
                 64 \
             ),

                 -- Session metadata
                 title VARCHAR \
             ( \
                 256 \
             ),
                 status ENUM \
             ( \
                 'active', \
                 'archived', \
                 'deleted' \
             ) DEFAULT 'active',

                 -- Document scope
                 document_count INT DEFAULT 0,

                 -- Stats
                 message_count INT DEFAULT 0,
                 total_tokens_used INT DEFAULT 0,

                 -- Timestamps
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                 last_message_at TIMESTAMP,
                 expires_at TIMESTAMP,

                 -- Metadata (JSON)
                 metadata JSON, \
                 INDEX idx_user_id \
             ( \
                 user_id \
             ),
                 INDEX idx_status \
             ( \
                 status \
             ),
                 INDEX idx_updated_at \
             ( \
                 updated_at \
             )
                 ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci;


-- Messages table: Individual messages in a session
             CREATE TABLE IF NOT EXISTS messages \
             ( \
                 id \
                 VARCHAR \
             ( \
                 36 \
             ) PRIMARY KEY,
                 session_id VARCHAR \
             ( \
                 36 \
             ) NOT NULL,

                 -- Message content
                 role ENUM \
             ( \
                 'user', \
                 'assistant', \
                 'system' \
             ) NOT NULL,
                 content TEXT NOT NULL,

                 -- For assistant messages
                 confidence FLOAT,
                 status ENUM \
             ( \
                 'success', \
                 'no_relevant_context', \
                 'low_confidence', \
                 'error' \
             ) DEFAULT 'success',

                 -- Stats
                 chunks_retrieved INT DEFAULT 0,
                 chunks_cited INT DEFAULT 0,
                 tokens_used INT DEFAULT 0,
                 processing_time_ms FLOAT,

                 -- Timestamps
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                 -- Metadata
                 metadata JSON, \
                 FOREIGN KEY \
             ( \
                 session_id \
             ) REFERENCES sessions \
             ( \
                 id \
             ) ON DELETE CASCADE,
                 INDEX idx_session_id \
             ( \
                 session_id \
             ),
                 INDEX idx_created_at \
             ( \
                 created_at \
             )
                 ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci;


             -- Message Chunks: Chunks retrieved for a message (all retrieved, not just cited)
-- This enables "Show all sources" in the UI
             CREATE TABLE IF NOT EXISTS message_chunks \
             ( \
                 id \
                 BIGINT \
                 AUTO_INCREMENT \
                 PRIMARY \
                 KEY, \
                 message_id \
                 VARCHAR \
             ( \
                 36 \
             ) NOT NULL,

                 -- Chunk identification (from Qdrant)
                 chunk_id VARCHAR \
             ( \
                 64 \
             ) NOT NULL,
                 document_id VARCHAR \
             ( \
                 36 \
             ) NOT NULL,

                 -- Retrieval info
                 retrieval_rank INT NOT NULL, -- 1, 2, 3... order of retrieval
                 similarity_score FLOAT NOT NULL,

                 -- Citation info (for UI display)
                 source_file VARCHAR \
             ( \
                 512 \
             ) NOT NULL,
                 page_number INT,
                 section_title VARCHAR \
             ( \
                 256 \
             ),

                 -- Content for display
                 content_preview TEXT, -- First ~500 chars
                 content_length INT,

                 -- Was this chunk cited in the answer?
                 was_cited BOOLEAN DEFAULT FALSE,
                 citation_number INT, -- [1], [2], etc. if cited

             -- Timestamps
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, \
                 FOREIGN KEY \
             ( \
                 message_id \
             ) REFERENCES messages \
             ( \
                 id \
             ) ON DELETE CASCADE,
                 INDEX idx_message_id \
             ( \
                 message_id \
             ),
                 INDEX idx_document_id \
             ( \
                 document_id \
             ),
                 INDEX idx_was_cited \
             ( \
                 was_cited \
             )
                 ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci;


-- Session Documents: Documents in scope for a session
             CREATE TABLE IF NOT EXISTS session_documents \
             ( \
                 session_id \
                 VARCHAR \
             ( \
                 36 \
             ) NOT NULL,
                 document_id VARCHAR \
             ( \
                 36 \
             ) NOT NULL,

                 -- Document info (cached from ingestion service)
                 filename VARCHAR \
             ( \
                 512 \
             ),
                 page_count INT,

                 -- When added
                 added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, \
                 PRIMARY KEY \
             ( \
                 session_id, \
                 document_id \
             ),
                 FOREIGN KEY \
             ( \
                 session_id \
             ) REFERENCES sessions \
             ( \
                 id \
             ) ON DELETE CASCADE
                 ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci;


-- Analytics/Feedback (optional but useful)
             CREATE TABLE IF NOT EXISTS message_feedback \
             ( \
                 id \
                 BIGINT \
                 AUTO_INCREMENT \
                 PRIMARY \
                 KEY, \
                 message_id \
                 VARCHAR \
             ( \
                 36 \
             ) NOT NULL,

                 -- Feedback
                 rating TINYINT, -- 1-5 or thumbs up/down
                 feedback_type ENUM \
             ( \
                 'helpful', \
                 'not_helpful', \
                 'incorrect', \
                 'incomplete' \
             ),
                 feedback_text TEXT,

                 -- Timestamps
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, \
                 FOREIGN KEY \
             ( \
                 message_id \
             ) REFERENCES messages \
             ( \
                 id \
             ) ON DELETE CASCADE,
                 INDEX idx_message_id \
             ( \
                 message_id \
             )
                 ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci; \
             """


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """
    MySQL connection manager with pooling.

    Uses aiomysql for async operations.
    """

    def __init__(
            self,
            host: str = "localhost",
            port: int = 3306,
            user: str = "root",
            password: str = "",
            database: str = "chat_service",
            min_size: int = 5,
            max_size: int = 20,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Optional[Pool] = None

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "DatabaseManager":
        """
        Create from MySQL URL.

        Format: mysql://user:password@host:port/database
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") or "chat_service",
            **kwargs,
        )

    async def initialize(self) -> None:
        """Initialize connection pool and create schema."""
        try:
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                minsize=self.min_size,
                maxsize=self.max_size,
                charset='utf8mb4',
                autocommit=True,
            )

            # Create schema
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    # Execute each statement separately
                    for statement in SCHEMA_SQL.split(';'):
                        statement = statement.strip()
                        if statement:
                            try:
                                await cur.execute(statement)
                            except Exception as e:
                                # Ignore "table already exists" errors
                                if "already exists" not in str(e).lower():
                                    logger.warning(f"Schema statement warning: {e}")

            logger.info(f"MySQL pool initialized: {self.host}:{self.port}/{self.database}")

        except Exception as e:
            logger.error(f"Failed to initialize MySQL: {e}")
            raise

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            logger.info("MySQL pool closed")

    @asynccontextmanager
    async def connection(self):
        """Get a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database not initialized")

        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def cursor(self, dict_cursor: bool = True):
        """Get a cursor."""
        async with self.connection() as conn:
            cursor_class = aiomysql.DictCursor if dict_cursor else aiomysql.Cursor
            async with conn.cursor(cursor_class) as cur:
                yield cur

    async def execute(self, query: str, args=None) -> int:
        """Execute a query, return affected rows."""
        async with self.cursor(dict_cursor=False) as cur:
            await cur.execute(query, args)
            return cur.rowcount

    async def fetchone(self, query: str, args=None) -> Optional[Dict]:
        """Fetch single row."""
        async with self.cursor() as cur:
            await cur.execute(query, args)
            return await cur.fetchone()

    async def fetchall(self, query: str, args=None) -> List[Dict]:
        """Fetch all rows."""
        async with self.cursor() as cur:
            await cur.execute(query, args)
            return await cur.fetchall()

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            result = await self.fetchone("SELECT 1 as ok")
            return result is not None
        except Exception:
            return False


# =============================================================================
# Global Instance Management
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> Optional[DatabaseManager]:
    """Get global database manager."""
    return _db_manager


def set_database_manager(manager: DatabaseManager) -> None:
    """Set global database manager."""
    global _db_manager
    _db_manager = manager


# =============================================================================
# Session Repository
# =============================================================================

class SessionRepository:
    """
    Repository for session operations.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def create_session(
            self,
            session_id: str,
            user_id: Optional[str] = None,
            title: Optional[str] = None,
            expires_minutes: int = 30,
    ) -> str:
        """Create a new session."""
        import json

        await self.db.execute(
            """
            INSERT INTO sessions (id, user_id, title, expires_at, metadata)
            VALUES (%s, %s, %s, DATE_ADD(NOW(), INTERVAL %s MINUTE), %s)
            """,
            (session_id, user_id, title, expires_minutes, json.dumps({})),
        )

        logger.debug(f"Created session: {session_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        return await self.db.fetchone(
            "SELECT * FROM sessions WHERE id = %s AND status = 'active'",
            (session_id,),
        )

    async def update_session(
            self,
            session_id: str,
            title: Optional[str] = None,
            message_count: Optional[int] = None,
    ) -> None:
        """Update session."""
        updates = ["updated_at = NOW()", "last_message_at = NOW()"]
        params = []

        if title is not None:
            updates.append("title = %s")
            params.append(title)

        if message_count is not None:
            updates.append("message_count = %s")
            params.append(message_count)

        params.append(session_id)

        await self.db.execute(
            f"UPDATE sessions SET {', '.join(updates)} WHERE id = %s",
            tuple(params),
        )

    async def extend_session(self, session_id: str, minutes: int = 30) -> None:
        """Extend session expiry."""
        await self.db.execute(
            "UPDATE sessions SET expires_at = DATE_ADD(NOW(), INTERVAL %s MINUTE) WHERE id = %s",
            (minutes, session_id),
        )

    async def delete_session(self, session_id: str) -> None:
        """Soft delete session."""
        await self.db.execute(
            "UPDATE sessions SET status = 'deleted' WHERE id = %s",
            (session_id,),
        )

    async def add_document_to_session(
            self,
            session_id: str,
            document_id: str,
            filename: str,
            page_count: int = 0,
    ) -> None:
        """Add document to session scope."""
        await self.db.execute(
            """
            INSERT
            IGNORE INTO session_documents (session_id, document_id, filename, page_count)
            VALUES (
            %s,
            %s,
            %s,
            %s
            )
            """,
            (session_id, document_id, filename, page_count),
        )

        # Update document count
        await self.db.execute(
            """
            UPDATE sessions
            SET document_count = (SELECT COUNT(*)
                                  FROM session_documents
                                  WHERE session_id = %s)
            WHERE id = %s
            """,
            (session_id, session_id),
        )

    async def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get documents in session scope."""
        return await self.db.fetchall(
            "SELECT * FROM session_documents WHERE session_id = %s",
            (session_id,),
        )

    async def list_sessions(
            self,
            user_id: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
    ) -> List[Dict]:
        """List sessions."""
        if user_id:
            return await self.db.fetchall(
                """
                SELECT id, user_id, title, message_count, created_at, updated_at
                FROM sessions
                WHERE user_id = %s
                  AND status = 'active'
                ORDER BY updated_at DESC
                    LIMIT %s
                OFFSET %s
                """,
                (user_id, limit, offset),
            )
        else:
            return await self.db.fetchall(
                """
                SELECT id, user_id, title, message_count, created_at, updated_at
                FROM sessions
                WHERE status = 'active'
                ORDER BY updated_at DESC
                    LIMIT %s
                OFFSET %s
                """,
                (limit, offset),
            )

    async def cleanup_expired(self) -> int:
        """Delete expired sessions."""
        result = await self.db.execute(
            "UPDATE sessions SET status = 'archived' WHERE expires_at < NOW() AND status = 'active'"
        )
        if result > 0:
            logger.info(f"Archived {result} expired sessions")
        return result


# =============================================================================
# Message Repository
# =============================================================================

class MessageRepository:
    """
    Repository for message and citation operations.

    Key insight: Each message has its own set of retrieved chunks.
    This allows the UI to show sources for each response.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def create_message(
            self,
            message_id: str,
            session_id: str,
            role: str,
            content: str,
            confidence: Optional[float] = None,
            status: str = "success",
            chunks_retrieved: int = 0,
            chunks_cited: int = 0,
            tokens_used: int = 0,
            processing_time_ms: float = 0,
            metadata: Optional[Dict] = None,
    ) -> str:
        """Create a new message."""
        import json

        await self.db.execute(
            """
            INSERT INTO messages (id, session_id, role, content, confidence, status,
                                  chunks_retrieved, chunks_cited, tokens_used, processing_time_ms, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                message_id, session_id, role, content, confidence, status,
                chunks_retrieved, chunks_cited, tokens_used, processing_time_ms,
                json.dumps(metadata or {}),
            ),
        )

        # Update session message count
        await self.db.execute(
            """
            UPDATE sessions
            SET message_count   = message_count + 1,
                last_message_at = NOW()
            WHERE id = %s
            """,
            (session_id,),
        )

        logger.debug(f"Created message: {message_id}")
        return message_id

    async def save_retrieved_chunks(
            self,
            message_id: str,
            chunks: List[Dict],
            cited_numbers: List[int] = None,
    ) -> int:
        """
        Save retrieved chunks for a message.

        Args:
            message_id: Message ID
            chunks: List of chunk dicts from search service
            cited_numbers: Which chunk numbers were cited [1, 2, etc.]

        Returns:
            Number of chunks saved
        """
        cited_numbers = cited_numbers or []
        count = 0

        for i, chunk in enumerate(chunks, 1):
            was_cited = i in cited_numbers
            citation_number = i if was_cited else None

            await self.db.execute(
                """
                INSERT INTO message_chunks (message_id, chunk_id, document_id,
                                            retrieval_rank, similarity_score,
                                            source_file, page_number, section_title,
                                            content_preview, content_length,
                                            was_cited, citation_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    message_id,
                    chunk.get("chunk_id", ""),
                    chunk.get("document_id", ""),
                    i,  # rank
                    chunk.get("score", 0.0),
                    chunk.get("filename", "unknown"),
                    chunk.get("page_number"),
                    chunk.get("section_title"),
                    chunk.get("content", ""),
                    len(chunk.get("content", "")),
                    was_cited,
                    citation_number,
                ),
            )
            count += 1

        logger.debug(f"Saved {count} chunks for message {message_id}")
        return count

    async def get_message(self, message_id: str) -> Optional[Dict]:
        """Get message by ID."""
        return await self.db.fetchone(
            "SELECT * FROM messages WHERE id = %s",
            (message_id,),
        )

    async def get_message_chunks(
            self,
            message_id: str,
            cited_only: bool = False,
    ) -> List[Dict]:
        """
        Get chunks for a message.

        Args:
            message_id: Message ID
            cited_only: Only return chunks that were cited
        """
        if cited_only:
            return await self.db.fetchall(
                """
                SELECT *
                FROM message_chunks
                WHERE message_id = %s
                  AND was_cited = TRUE
                ORDER BY citation_number
                """,
                (message_id,),
            )
        else:
            return await self.db.fetchall(
                """
                SELECT *
                FROM message_chunks
                WHERE message_id = %s
                ORDER BY retrieval_rank
                """,
                (message_id,),
            )

    async def get_session_messages(
            self,
            session_id: str,
            limit: int = 100,
            include_chunks: bool = False,
    ) -> List[Dict]:
        """
        Get messages for a session.

        Args:
            session_id: Session ID
            limit: Max messages to return
            include_chunks: Include chunk data for assistant messages
        """
        messages = await self.db.fetchall(
            """
            SELECT *
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at ASC
                LIMIT %s
            """,
            (session_id, limit),
        )

        if include_chunks:
            for msg in messages:
                if msg["role"] == "assistant":
                    msg["chunks"] = await self.get_message_chunks(
                        msg["id"],
                        cited_only=True,
                    )

        return messages

    async def get_message_with_citations(self, message_id: str) -> Optional[Dict]:
        """
        Get message with full citation details.

        This is what the frontend needs to display:
        - The answer text
        - The cited chunks with source info
        """
        message = await self.get_message(message_id)
        if not message:
            return None

        # Get cited chunks
        citations = await self.get_message_chunks(message_id, cited_only=True)

        # Get all retrieved chunks (for "show all sources")
        all_chunks = await self.get_message_chunks(message_id, cited_only=False)

        return {
            **message,
            "citations": citations,
            "all_retrieved_chunks": all_chunks,
        }

    async def add_feedback(
            self,
            message_id: str,
            rating: Optional[int] = None,
            feedback_type: Optional[str] = None,
            feedback_text: Optional[str] = None,
    ) -> None:
        """Add feedback for a message."""
        await self.db.execute(
            """
            INSERT INTO message_feedback (message_id, rating, feedback_type, feedback_text)
            VALUES (%s, %s, %s, %s)
            """,
            (message_id, rating, feedback_type, feedback_text),
        )

    async def get_conversation_for_llm(
            self,
            session_id: str,
            max_turns: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM.

        Returns list of {"role": "user/assistant", "content": "..."}
        """
        messages = await self.db.fetchall(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = %s
              AND role IN ('user', 'assistant')
            ORDER BY created_at DESC
                LIMIT %s
            """,
            (session_id, max_turns * 2),
        )

        # Reverse to get chronological order
        messages = list(reversed(messages))

        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]