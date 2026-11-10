from __future__ import annotations

import json
import sqlite3
from math import isfinite
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterator

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
            db.execute("""CREATE TABLE IF NOT EXISTS corporate_actions (
                pair TEXT NOT NULL, timestamp INTEGER NOT NULL,
                dividend REAL NOT NULL DEFAULT 0, split_ratio REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(pair, timestamp))""")
            db.execute("""CREATE TABLE IF NOT EXISTS ai_observations (
                state_id TEXT NOT NULL, pair TEXT NOT NULL, decision_timestamp INTEGER NOT NULL,
                close REAL NOT NULL, regime TEXT NOT NULL, confidence REAL NOT NULL,
                news_json TEXT NOT NULL, resolved_timestamp INTEGER, net_return_pct REAL,
                PRIMARY KEY(state_id, pair, decision_timestamp))""")
            db.execute("""CREATE INDEX IF NOT EXISTS ai_observations_resolved_idx
                ON ai_observations(state_id, resolved_timestamp, regime)""")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

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

    def save_corporate_actions(self, actions: list[dict[str, object]]) -> int:
        if not actions:
            return 0
        rows = []
        for item in actions:
            dividend = float(item.get("dividend", 0))
            split_ratio = float(item.get("split_ratio", 0))
            if not isfinite(dividend) or not isfinite(split_ratio) or dividend < 0 or split_ratio < 0:
                raise ValueError("Corporate-action amounts must be finite and non-negative")
            if dividend > 0 or split_ratio > 0:
                rows.append((str(item["pair"]).upper(), int(item["timestamp"]), dividend, split_ratio))
        if not rows:
            return 0
        with self.connect() as db:
            before = db.total_changes
            db.executemany("""INSERT INTO corporate_actions(pair, timestamp, dividend, split_ratio)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(pair, timestamp) DO UPDATE SET
                dividend=excluded.dividend, split_ratio=excluded.split_ratio""", rows)
            return db.total_changes - before

    def get_corporate_actions(
        self, pairs: list[str], from_timestamp: int | None = None,
        through_timestamp: int | None = None,
    ) -> dict[str, dict[int, dict[str, float]]]:
        if not pairs:
            return {}
        placeholders = ",".join("?" for _ in pairs)
        query = f"SELECT pair, timestamp, dividend, split_ratio FROM corporate_actions WHERE pair IN ({placeholders})"
        params: list[object] = [pair.upper() for pair in pairs]
        if from_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(from_timestamp)
        if through_timestamp is not None:
            query += " AND timestamp <= ?"
            params.append(through_timestamp)
        query += " ORDER BY pair, timestamp"
        grouped: dict[str, dict[int, dict[str, float]]] = {}
        with self.connect() as db:
            for pair, timestamp, dividend, split_ratio in db.execute(query, params):
                grouped.setdefault(str(pair), {})[int(timestamp)] = {
                    "dividend": float(dividend), "split_ratio": float(split_ratio),
                }
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

    def record_ai_observations(self, state_id: str, observations: list[dict[str, Any]]) -> int:
        """Keep at most one overlapping four-hour label per pair in the learning set."""
        if not observations:
            return 0
        minimum_spacing_seconds = 4 * 60 * 60
        rows = [
            (
                state_id,
                str(item["pair"]).upper(),
                int(item["decision_timestamp"]),
                float(item["close"]),
                str(item["regime"]),
                float(item["confidence"]),
                json.dumps(item.get("news", []), separators=(",", ":"), allow_nan=False),
            )
            for item in observations
            if item.get("regime") in {
                "trend", "range", "breakout", "relative_momentum", "cash",
                "high_volatility", "unknown",
            }
        ]
        inserted = 0
        with self.connect() as db:
            for row in rows:
                _, pair, decision_timestamp, *_ = row
                overlapping = db.execute("""SELECT 1 FROM ai_observations
                    WHERE state_id=? AND pair=? AND decision_timestamp>? AND decision_timestamp<=?
                    LIMIT 1""",
                    (state_id, pair, decision_timestamp - minimum_spacing_seconds, decision_timestamp),
                ).fetchone()
                if overlapping:
                    continue
                cursor = db.execute("""INSERT OR IGNORE INTO ai_observations
                    (state_id, pair, decision_timestamp, close, regime, confidence, news_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""", row)
                inserted += cursor.rowcount
        return inserted

    def resolve_ai_observations(
        self,
        state_id: str,
        candles: dict[str, list[Candle]],
        as_of_timestamp: int,
        interval_minutes: int,
        fee_bps: float,
        slippage_bps: float,
        horizon_minutes: int = 240,
    ) -> int:
        """Resolve only examples whose full forward horizon is available by the event time."""
        if interval_minutes <= 0 or horizon_minutes % interval_minutes:
            raise ValueError("Candle interval must divide the four-hour feedback horizon exactly.")
        horizon_bars = horizon_minutes // interval_minutes
        indexes = {
            pair: {bar.timestamp: index for index, bar in enumerate(bars)}
            for pair, bars in candles.items()
        }
        updates: list[tuple[int, float, str, str, int]] = []
        fee_rate = fee_bps / 10_000
        slippage_rate = slippage_bps / 10_000
        with self.connect() as db:
            pending = db.execute("""SELECT pair, decision_timestamp FROM ai_observations
                WHERE state_id=? AND resolved_timestamp IS NULL AND decision_timestamp < ?
                ORDER BY decision_timestamp""", (state_id, as_of_timestamp)).fetchall()
            for pair, decision_timestamp in pending:
                bars = candles.get(pair, [])
                index = indexes.get(pair, {}).get(int(decision_timestamp))
                if index is None or index + horizon_bars >= len(bars):
                    continue
                entry = bars[index + 1]
                outcome = bars[index + horizon_bars]
                expected_entry = int(decision_timestamp) + interval_minutes * 60
                expected_outcome = int(decision_timestamp) + horizon_minutes * 60
                if (
                    entry.timestamp != expected_entry
                    or outcome.timestamp != expected_outcome
                    or outcome.timestamp > as_of_timestamp
                    or entry.open <= 0
                    or outcome.close <= 0
                ):
                    continue
                entry_fill = entry.open * (1 + slippage_rate)
                exit_fill = outcome.close * (1 - slippage_rate)
                net_return_pct = ((exit_fill * (1 - fee_rate)) / (entry_fill * (1 + fee_rate)) - 1) * 100
                updates.append((outcome.timestamp, net_return_pct, state_id, pair, int(decision_timestamp)))
            if updates:
                db.executemany("""UPDATE ai_observations
                    SET resolved_timestamp=?, net_return_pct=?
                    WHERE state_id=? AND pair=? AND decision_timestamp=?
                    AND resolved_timestamp IS NULL""", updates)
        return len(updates)

    def get_ai_feedback(
        self,
        state_id: str,
        as_of_timestamp: int,
        minimum_regime_samples: int = 25,
        max_samples: int = 2000,
    ) -> dict[str, Any]:
        """Summarize outcomes with distinct four-hour windows as samples.

        Coin outcomes from the same window are correlated, so they are
        collapsed to one median return per regime and window before applying
        the minimum sample gate.
        """
        with self.connect() as db:
            rows = db.execute("""SELECT decision_timestamp, regime, net_return_pct FROM ai_observations
                WHERE state_id=? AND resolved_timestamp IS NOT NULL AND resolved_timestamp <= ?
                ORDER BY resolved_timestamp DESC LIMIT ?""",
                (state_id, as_of_timestamp, max_samples)).fetchall()
        horizon_seconds = 4 * 60 * 60
        grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        all_windows: set[int] = set()
        for decision_timestamp, regime, net_return_pct in rows:
            window = int(decision_timestamp) // horizon_seconds
            all_windows.add(window)
            grouped[str(regime)][window].append(float(net_return_pct))
        by_regime = {}
        for regime, windows in grouped.items():
            # Each window contributes one robust cross-sectional result,
            # regardless of how many coins received the same regime label.
            results = [median(window_results) for window_results in windows.values()]
            if len(results) < minimum_regime_samples:
                continue
            by_regime[regime] = {
                "samples": len(results),
                "asset_observations": sum(len(window_results) for window_results in windows.values()),
                "mean_4h_net_return_pct": round(fmean(results), 4),
                "median_4h_net_return_pct": round(median(results), 4),
                "win_rate_pct": round(sum(result > 0 for result in results) / len(results) * 100, 2),
            }
        return {
            "target": "next_4h_long_return_after_fees_and_slippage",
            "resolved_samples": len(rows),
            "independent_windows": len(all_windows),
            "minimum_samples_per_regime": minimum_regime_samples,
            "sample_unit": "distinct 4-hour windows; per-window asset outcomes are aggregated by median",
            "by_regime": by_regime,
            "ready": bool(by_regime),
            "status": "ready" if by_regime else "collecting_more_resolved_examples",
        }

    def clear_paper_state(self, account_id: str | None = None) -> None:
        with self.connect() as db:
            if account_id is None:
                db.execute("DELETE FROM paper_state")
            else:
                db.execute("DELETE FROM paper_state WHERE state_id=?", (account_id,))
