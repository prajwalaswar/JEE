"""
memory_store.py — SQLite-backed persistent memory for the Math Mentor.

Stores every pipeline run and user feedback so agents can:
 - Retrieve similar past problems (fuzzy text match)
 - Re-use solution patterns
 - Correct OCR mistakes based on prior human edits
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from backend.config import SQLITE_DB_PATH
from backend.models import MemoryRecord, UserFeedback

logger = logging.getLogger(__name__)

# ── SQL schema ────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory (
    record_id           TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    original_input      TEXT NOT NULL,
    parsed_problem      TEXT,          -- JSON
    retrieved_context   TEXT,          -- JSON
    solution            TEXT,          -- JSON
    verifier_result     TEXT,          -- JSON
    user_feedback       TEXT,          -- "correct" | "incorrect"
    feedback_comment    TEXT,
    created_at          TEXT NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_session ON memory (session_id);
"""


class MemoryStore:
    """
    Thread-safe SQLite memory store.

    Usage:
        store = MemoryStore()
        store.save(record)
        results = store.get_similar("find quadratic roots")
    """

    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialise_db()

    # ── Internals ─────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a new connection and auto-commits."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialise_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
        logger.info("Memory store initialised at: %s", self.db_path)

    @staticmethod
    def _to_json(obj) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return json.dumps(obj)
        return json.dumps(obj.dict() if hasattr(obj, "dict") else obj)

    @staticmethod
    def _from_json(raw: Optional[str]) -> Optional[dict]:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save(self, record: MemoryRecord) -> None:
        """
        Insert or replace a memory record.

        Args:
            record: MemoryRecord instance.
        """
        sql = """
        INSERT OR REPLACE INTO memory
            (record_id, session_id, original_input, parsed_problem,
             retrieved_context, solution, verifier_result,
             user_feedback, feedback_comment, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                record.record_id,
                record.session_id,
                record.original_input,
                self._to_json(record.parsed_problem),
                self._to_json(record.retrieved_context),
                self._to_json(record.solution),
                self._to_json(record.verifier_result),
                record.user_feedback,
                record.feedback_comment,
                record.created_at,
            ))
        logger.debug("Saved memory record: %s", record.record_id)

    def get_by_session(self, session_id: str) -> Optional[MemoryRecord]:
        """Retrieve the most recent record for a given session."""
        sql = """
        SELECT * FROM memory
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """
        with self._conn() as conn:
            row = conn.execute(sql, (session_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def update_feedback(self, feedback: UserFeedback) -> None:
        """
        Attach user feedback to the latest memory record for a session.

        Args:
            feedback: UserFeedback instance.
        """
        sql = """
        UPDATE memory
        SET user_feedback = ?, feedback_comment = ?
        WHERE session_id = ?
          AND record_id = (
              SELECT record_id FROM memory
              WHERE session_id = ?
              ORDER BY created_at DESC
              LIMIT 1
          )
        """
        with self._conn() as conn:
            conn.execute(sql, (
                feedback.feedback_type.value,
                feedback.comment,
                feedback.session_id,
                feedback.session_id,
            ))
        logger.debug("Updated feedback for session: %s", feedback.session_id)

    def get_similar(self, query_text: str, limit: int = 3) -> List[MemoryRecord]:
        """
        Find past records whose original_input contains overlapping keywords.
        This is a lightweight LIKE-based fuzzy match.

        Args:
            query_text: Search query.
            limit:      Maximum records to return.

        Returns:
            List of MemoryRecord sorted by most recent.
        """
        # Extract significant keywords (>3 chars, lowercase)
        keywords = [
            w.lower() for w in query_text.split() if len(w) > 3
        ]

        if not keywords:
            return []

        # Build LIKE clauses
        clauses = " OR ".join(
            "LOWER(original_input) LIKE ?" for _ in keywords
        )
        params  = [f"%{kw}%" for kw in keywords]

        sql = f"""
        SELECT * FROM memory
        WHERE {clauses}
        ORDER BY created_at DESC
        LIMIT ?
        """
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_record(r) for r in rows]

    def get_recent(self, limit: int = 10) -> List[MemoryRecord]:
        """Return the most recent memory records."""
        sql = "SELECT * FROM memory ORDER BY created_at DESC LIMIT ?"
        with self._conn() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ── Helper ────────────────────────────────────────────────────────────────

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            record_id=row["record_id"],
            session_id=row["session_id"],
            original_input=row["original_input"],
            parsed_problem=self._from_json(row["parsed_problem"]),
            retrieved_context=self._from_json(row["retrieved_context"]),
            solution=self._from_json(row["solution"]),
            verifier_result=self._from_json(row["verifier_result"]),
            user_feedback=row["user_feedback"],
            feedback_comment=row["feedback_comment"],
            created_at=row["created_at"],
        )


# Module-level singleton
_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Return (or lazily create) the shared MemoryStore."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
