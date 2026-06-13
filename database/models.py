"""
SQLite database layer.

Tables:
  posts   — generated post variants and their lifecycle status
  digests — analyst digest results (10 topics per run)

posts status lifecycle:
  pending   → sent to admin, waiting for decision
  approved  → admin picked a variant, ready to publish
  published → successfully posted to the channel
  rejected  → admin rejected all variants
  editing   → admin is typing a custom edit
  error     → publish failed

digests status lifecycle:
  pending   → sent to admin as topic menu, waiting for clicks
  processed → at least one topic was sent to WriterCrew
  rejected  → admin rejected all topics
"""

import json
import logging
import sqlite3
from datetime import datetime

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create schema if it doesn't exist and apply any pending migrations."""
    conn = _connect()

    # --- Create table (v1 baseline) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source_urls      TEXT    NOT NULL DEFAULT '[]',
            raw_news         TEXT    NOT NULL DEFAULT '',
            variants         TEXT    NOT NULL DEFAULT '[]',
            selected_variant INTEGER,
            final_text       TEXT,
            status           TEXT    NOT NULL DEFAULT 'pending',
            tg_message_id    INTEGER,
            created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
            published_at     TEXT
        )
    """)

    # --- Migrations: add columns introduced in later versions ---
    # Each migration is idempotent: ignored if the column already exists.
    _add_column_if_missing(conn, "posts", "image_prompt", "TEXT")
    _add_column_if_missing(conn, "posts", "researcher_summary", "TEXT")
    _add_column_if_missing(conn, "posts", "media_path", "TEXT")
    _add_column_if_missing(conn, "posts", "media_type", "TEXT")

    # --- digests table (added in voronka refactor) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS digests (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            date           TEXT    NOT NULL,
            topics_json    TEXT    NOT NULL DEFAULT '[]',
            status         TEXT    NOT NULL DEFAULT 'pending',
            tg_message_id  INTEGER,
            created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, col_type: str
) -> None:
    """ALTER TABLE … ADD COLUMN if the column doesn't exist yet."""
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})")
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        logger.info("DB migration: added column %s.%s", table, column)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def create_post(
    source_urls: list[str],
    raw_news: str,
    variants: list[str],
    image_prompt: str | None = None,
    researcher_summary: str = "",
    media_path: str | None = None,
    media_type: str | None = None,
) -> int:
    """Insert a new post record and return its ID."""
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO posts
           (source_urls, raw_news, variants, image_prompt, researcher_summary,
            media_path, media_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            json.dumps(source_urls, ensure_ascii=False),
            raw_news,
            json.dumps(variants, ensure_ascii=False),
            image_prompt,
            researcher_summary,
            media_path,
            media_type,
        ),
    )
    post_id: int = cur.lastrowid  # type: ignore[assignment]
    conn.commit()
    conn.close()
    return post_id


def set_tg_message_id(post_id: int, message_id: int) -> None:
    """Store the Telegram message_id of the approval message sent to admin."""
    _exec("UPDATE posts SET tg_message_id = ? WHERE id = ?", (message_id, post_id))


def update_status(post_id: int, status: str, **extra_fields) -> None:
    """Update post status and any additional columns passed as kwargs."""
    parts = ["status = ?"]
    values: list = [status]

    for col, val in extra_fields.items():
        parts.append(f"{col} = ?")
        values.append(val)

    if status == "published":
        parts.append("published_at = ?")
        values.append(datetime.now().isoformat(timespec="seconds"))

    values.append(post_id)
    _exec(f"UPDATE posts SET {', '.join(parts)} WHERE id = ?", values)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_post(post_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return _deserialize(dict(row))


def get_recent_posts(limit: int = 10) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM posts ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [_deserialize(dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _exec(sql: str, params=()) -> None:
    conn = _connect()
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def _deserialize(row: dict) -> dict:
    for field in ("source_urls", "variants"):
        if isinstance(row.get(field), str):
            try:
                row[field] = json.loads(row[field])
            except json.JSONDecodeError:
                row[field] = []
    return row


# ---------------------------------------------------------------------------
# Digest write operations
# ---------------------------------------------------------------------------

def create_digest(date: str, topics: list[dict]) -> int:
    """
    Insert a new digest record and return its ID.
    topics — list of dicts serialisable to JSON (DigestTopic.__dict__).
    """
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO digests (date, topics_json) VALUES (?, ?)",
        (date, json.dumps(topics, ensure_ascii=False)),
    )
    digest_id: int = cur.lastrowid  # type: ignore[assignment]
    conn.commit()
    conn.close()
    return digest_id


def set_digest_tg_message_id(digest_id: int, message_id: int) -> None:
    """Store the Telegram message_id of the digest menu message sent to admin."""
    _exec(
        "UPDATE digests SET tg_message_id = ? WHERE id = ?",
        (message_id, digest_id),
    )


def update_digest_status(digest_id: int, status: str) -> None:
    """Update digest status (pending / processed / rejected)."""
    _exec("UPDATE digests SET status = ? WHERE id = ?", (status, digest_id))


# ---------------------------------------------------------------------------
# Digest read operations
# ---------------------------------------------------------------------------

def get_digest(digest_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM digests WHERE id = ?", (digest_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return _deserialize_digest(dict(row))


def get_recent_digests(limit: int = 5) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM digests ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [_deserialize_digest(dict(r)) for r in rows]


def _deserialize_digest(row: dict) -> dict:
    if isinstance(row.get("topics_json"), str):
        try:
            row["topics"] = json.loads(row["topics_json"])
        except json.JSONDecodeError:
            row["topics"] = []
    return row


def get_recently_seen_topics(days: int = 3) -> list[dict]:
    """
    Return a flat list of {title, url} dicts for all topics that appeared
    in digests created within the last `days` days.

    Used by scheduler.py to pass dedup context to DigestCrew so the analyst
    avoids recommending the same topics twice.
    """
    from datetime import datetime, timedelta  # noqa: PLC0415

    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    conn = _connect()
    rows = conn.execute(
        "SELECT topics_json FROM digests WHERE created_at >= ?",
        (cutoff,),
    ).fetchall()
    conn.close()

    seen: list[dict] = []
    for row in rows:
        try:
            topics = json.loads(row[0] or "[]")
        except json.JSONDecodeError:
            continue
        for t in topics:
            title = t.get("title", "")
            url   = t.get("url", "")
            if title or url:
                seen.append({"title": title, "url": url})

    return seen
