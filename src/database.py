"""SQLite log of detection events - feeds the dashboard and the AI agent."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,          -- ISO-8601 local time
    source        TEXT NOT NULL,          -- webcam / image / video
    label         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    is_violation  INTEGER NOT NULL,
    snapshot      TEXT                    -- path to saved frame, if any
);
CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts);
CREATE TABLE IF NOT EXISTS alerts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    kind     TEXT NOT NULL,              -- low_compliance / persistent_violations / summary
    message  TEXT NOT NULL
);
"""


_local = threading.local()   # one connection per thread (sqlite connections aren't thread-safe)


def _connection() -> sqlite3.Connection:
    path = str(DB_PATH)
    conn = getattr(_local, "conn", None)
    if conn is None or _local.path != path:
        conn = sqlite3.connect(path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")      # readers (dashboard) don't block the writer
        conn.execute("PRAGMA synchronous=NORMAL")    # safe with WAL, far fewer disk flushes
        conn.executescript(SCHEMA)                   # schema checked once per connection
        _local.conn, _local.path = conn, path
    return conn


@contextmanager
def connect():
    conn = _connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close() -> None:
    """Close this thread's connection (e.g. before deleting the database file)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


def log_detections(rows: list[tuple[str, float, bool, str, str | None]]) -> None:
    """Batch insert (label, confidence, is_violation, source, snapshot) rows in one transaction."""
    ts = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        conn.executemany(
            "INSERT INTO detections (ts, source, label, confidence, is_violation, snapshot) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(ts, source, label, round(conf, 4), int(viol), snap)
             for label, conf, viol, source, snap in rows],
        )


def log_detection(label: str, confidence: float, is_violation: bool,
                  source: str = "webcam", snapshot: str | None = None) -> None:
    log_detections([(label, confidence, is_violation, source, snapshot)])


def log_alert(kind: str, message: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO alerts (ts, kind, message) VALUES (?, ?, ?)",
                     (datetime.now().isoformat(timespec="seconds"), kind, message))


def recent_alerts(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def _since(hours: float | None) -> str:
    if hours is None:
        return "0000"
    return (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")


def summary(hours: float | None = 24) -> dict:
    """Totals and compliance rate over the last `hours` (None = all time)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n, AVG(confidence) AS avg_conf, "
            "SUM(is_violation) AS violations FROM detections WHERE ts >= ? GROUP BY label",
            (_since(hours),),
        ).fetchall()
    by_label = {r["label"]: {"count": r["n"], "avg_confidence": round(r["avg_conf"], 3)}
                for r in rows}
    total = sum(r["n"] for r in rows)
    violations = sum(r["violations"] for r in rows)
    return {
        "window_hours": hours,
        "total_detections": total,
        "violations": violations,
        "compliance_rate": round(100 * (total - violations) / total, 1) if total else None,
        "by_label": by_label,
    }


def hourly(hours: float | None = 24) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT substr(ts, 1, 13) AS hour, COUNT(*) AS total, "
            "SUM(is_violation) AS violations FROM detections WHERE ts >= ? "
            "GROUP BY hour ORDER BY hour",
            (_since(hours),),
        ).fetchall()
    return [dict(r) for r in rows]


def recent(limit: int = 50, violations_only: bool = False) -> list[dict]:
    where = "WHERE is_violation = 1" if violations_only else ""
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM detections {where} ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear() -> None:
    with connect() as conn:
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM alerts")
