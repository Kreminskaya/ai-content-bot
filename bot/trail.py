"""
TRAIL audit logging — append-only JSONL trail for ai-telegram-assistant.

Format per trail-spec:
  timestamp, trace_id, action (send/edit/delete), content_id (telegram:message:N),
  source, actor, params, result

Functions:
  trail_append(entry)   — append a trail entry
  is_used(content_id)   — bool: has this content_id been acted on?
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_TRAIL_DIR = Path(__file__).parent.parent / "data"
_TRAIL_FILE = _TRAIL_DIR / "trail.jsonl"

# ---------------------------------------------------------------------------
# Low-level append
# ---------------------------------------------------------------------------

def _ensure_trail_dir() -> None:
    _TRAIL_DIR.mkdir(parents=True, exist_ok=True)


def trail_append(
    action: str,
    content_id: str,
    *,
    trace_id: str | None = None,
    source: str = "ai-telegram-assistant",
    actor: str = "bot",
    params: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Append an append-only JSONL entry to trail.jsonl.

    Returns the written entry (useful for testing / verification).
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "action": action,
        "content_id": content_id,
        "source": source,
        "actor": actor,
        "params": params or {},
        "result": result or {},
    }

    _ensure_trail_dir()
    with open(_TRAIL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def is_used(content_id: str) -> bool:
    """
    Return True if any entry with the given content_id exists in the trail.
    Scans the file — suitable for deduplication checks before publish.
    """
    if not _TRAIL_FILE.exists():
        return False

    with open(_TRAIL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("content_id") == content_id:
                    return True
            except json.JSONDecodeError:
                continue

    return False


def last_action(content_id: str) -> dict[str, Any] | None:
    """
    Return the most recent entry for content_id, or None if not found.
    """
    if not _TRAIL_FILE.exists():
        return None

    last: dict[str, Any] | None = None
    with open(_TRAIL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("content_id") == content_id:
                    last = entry
            except json.JSONDecodeError:
                continue

    return last