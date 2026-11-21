"""SQLite persistence: time series, events and the learnt model (survives restarts).

Writes go through a background thread so a slow disk never stalls the video pipeline.
One file, no server: easy to open in DB Browser, pandas or a notebook.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts REAL PRIMARY KEY,
    fish_count INTEGER,
    activity REAL, cohesion REAL, nnd REAL, polarization REAL,
    mean_y REAL, mean_z REAL, burst_ratio REAL, rest_ratio REAL, surface_ratio REAL,
    activity_z REAL, cohesion_z REAL,
    mood TEXT
);
CREATE TABLE IF NOT EXISTS fish_samples (
    ts REAL, fish_id INTEGER, x REAL, y REAL, z REAL, speed_bls REAL, state TEXT
);
CREATE INDEX IF NOT EXISTS fish_samples_ts ON fish_samples(ts);
CREATE INDEX IF NOT EXISTS fish_samples_id ON fish_samples(fish_id, ts);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, kind TEXT, severity REAL, message TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated REAL);
"""


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue[tuple[str, Any] | None] = queue.Queue(maxsize=10_000)
        self._read = self._connect()
        self._read_lock = threading.Lock()
        self._thread = threading.Thread(target=self._writer, daemon=True, name="sqlite-writer")
        self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        conn.commit()
        return conn

    # ------------------------------------------------------------ writes (async)
    def write_snapshot(self, snap: dict[str, Any]) -> None:
        self._put("snapshot", snap)

    def write_events(self, events: list[dict[str, Any]]) -> None:
        if events:
            self._put("events", events)

    def save_state(self, key: str, value: Any) -> None:
        self._put("kv", (key, json.dumps(value)))

    def _put(self, kind: str, payload: Any) -> None:
        try:
            self._q.put_nowait((kind, payload))
        except queue.Full:
            pass  # never block the video loop; losing a sample is acceptable

    def _writer(self) -> None:
        conn = self._connect()
        while True:
            item = self._q.get()
            if item is None:
                break
            batch = [item]
            while len(batch) < 500:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._q.put(None)
                    break
                batch.append(nxt)
            try:
                with conn:
                    for kind, payload in batch:
                        self._apply(conn, kind, payload)
            except sqlite3.Error:
                time.sleep(0.5)
        conn.close()

    @staticmethod
    def _apply(conn: sqlite3.Connection, kind: str, p: Any) -> None:
        if kind == "snapshot":
            m, z = p["metrics"], p.get("zscores", {})
            conn.execute(
                "INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (p["t"], p["fish_count"], m.get("activity"), m.get("cohesion"), m.get("nnd"),
                 m.get("polarization"), m.get("mean_y"), m.get("mean_z"), m.get("burst_ratio"),
                 m.get("rest_ratio"), m.get("surface_ratio"), z.get("activity"),
                 z.get("cohesion"), p.get("mood", {}).get("label")),
            )
            conn.executemany(
                "INSERT INTO fish_samples VALUES (?,?,?,?,?,?,?)",
                [(p["t"], f["id"], f["x"], f["y"], f["z"], f["speed_bls"], f["state"])
                 for f in p.get("fish", [])],
            )
        elif kind == "events":
            conn.executemany(
                "INSERT INTO events (ts, kind, severity, message, payload) VALUES (?,?,?,?,?)",
                [(e["timestamp"], e["kind"], e["severity"], e["message"],
                  json.dumps({"fish": e.get("fish", []), **e.get("data", {})})) for e in p],
            )
        elif kind == "kv":
            key, value = p
            conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (key, value, time.time()))

    def flush(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.05)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)
        self._read.close()

    # ------------------------------------------------------------ reads
    def load_state(self, key: str) -> Any | None:
        with self._read_lock:
            row = self._read.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def history(self, minutes: float = 60.0, bucket_s: float | None = None) -> list[dict[str, Any]]:
        since = time.time() - minutes * 60
        bucket = bucket_s or max(1.0, minutes * 60 / 600)  # ~600 points max
        with self._read_lock:
            rows = self._read.execute(
                """SELECT CAST(ts / ? AS INTEGER) * ? AS b, AVG(fish_count), AVG(activity),
                          AVG(cohesion), AVG(polarization), AVG(mean_y), AVG(mean_z),
                          AVG(activity_z), AVG(rest_ratio), AVG(burst_ratio)
                   FROM samples WHERE ts >= ? GROUP BY b ORDER BY b""",
                (bucket, bucket, since),
            ).fetchall()
        keys = ("t", "fish_count", "activity", "cohesion", "polarization", "mean_y", "mean_z",
                "activity_z", "rest_ratio", "burst_ratio")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def events(self, limit: int = 50, since: float | None = None) -> list[dict[str, Any]]:
        with self._read_lock:
            rows = self._read.execute(
                "SELECT ts, kind, severity, message, payload FROM events WHERE ts >= ? "
                "ORDER BY ts DESC LIMIT ?", (since or 0.0, limit)).fetchall()
        return [{"timestamp": r[0], "kind": r[1], "severity": r[2], "message": r[3],
                 **json.loads(r[4] or "{}")} for r in rows]

    def daily_summary(self, days: int = 7) -> list[dict[str, Any]]:
        since = time.time() - days * 86400
        with self._read_lock:
            rows = self._read.execute(
                """SELECT date(ts, 'unixepoch', 'localtime') AS d, AVG(activity), MAX(fish_count),
                          AVG(cohesion), AVG(mean_z), COUNT(*)
                   FROM samples WHERE ts >= ? GROUP BY d ORDER BY d""", (since,)).fetchall()
            ev = dict(self._read.execute(
                """SELECT date(ts, 'unixepoch', 'localtime') AS d, COUNT(*) FROM events
                   WHERE ts >= ? GROUP BY d""", (since,)).fetchall())
        return [{"date": r[0], "activity": r[1], "max_fish": r[2], "cohesion": r[3],
                 "mean_z": r[4], "samples": r[5], "events": ev.get(r[0], 0)} for r in rows]
