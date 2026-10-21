from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


class CollectionCache:
    """Collection history is committed only after the corresponding index succeeds."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS collections (cache_key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str):
        with self.connect() as c:
            row = c.execute("SELECT value FROM collections WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict):
        value = {**value, "indexed_at": datetime.now(timezone.utc).isoformat()}
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO collections VALUES (?, ?)",
                      (key, json.dumps(value, ensure_ascii=False)))
