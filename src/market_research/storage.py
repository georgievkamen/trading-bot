from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Candle


class MarketStore:
    def __init__(self, path: str | Path = "data/market_research.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS candles (
                pair TEXT NOT NULL, interval INTEGER NOT NULL, timestamp INTEGER NOT NULL,
                open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
                close REAL NOT NULL, volume REAL NOT NULL,
                PRIMARY KEY(pair, interval, timestamp))""")
            db.execute("""CREATE TABLE IF NOT EXISTS paper_state (
                state_id TEXT PRIMARY KEY, payload TEXT NOT NULL)""")

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save_candles(self, candles: list[Candle]) -> int:
        if not candles:
            return 0
        with self.connect() as db:
            before = db.total_changes
            db.executemany("""INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair, interval, timestamp) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume""",
                [(c.pair, c.interval, c.timestamp, c.open, c.high, c.low, c.close, c.volume) for c in candles])
            return db.total_changes - before

    def get_candles(self, interval: int, pairs: list[str] | None = None) -> dict[str, list[Candle]]:
        query = "SELECT pair, interval, timestamp, open, high, low, close, volume FROM candles WHERE interval=?"
        params: list[object] = [interval]
        if pairs:
            placeholders = ",".join("?" for _ in pairs)
            query += f" AND pair IN ({placeholders})"
            params.extend(pair.upper() for pair in pairs)
        query += " ORDER BY pair, timestamp"
        grouped: dict[str, list[Candle]] = {}
        with self.connect() as db:
            for row in db.execute(query, params):
                candle = Candle(*row)
                grouped.setdefault(candle.pair, []).append(candle)
        return grouped

    def get_paper_state(self, account_id: str) -> dict[str, object] | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM paper_state WHERE state_id=?", (account_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_paper_state(self, payload: dict[str, object], account_id: str) -> None:
        text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        with self.connect() as db:
            db.execute("""INSERT INTO paper_state(state_id, payload) VALUES(?, ?)
                ON CONFLICT(state_id) DO UPDATE SET payload=excluded.payload""", (account_id, text))

    def clear_paper_state(self, account_id: str | None = None) -> None:
        with self.connect() as db:
            if account_id is None:
                db.execute("DELETE FROM paper_state")
            else:
                db.execute("DELETE FROM paper_state WHERE state_id=?", (account_id,))
