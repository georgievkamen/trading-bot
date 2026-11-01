from __future__ import annotations

import json
import math
import sqlite3
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .kraken import NON_CRYPTO_BASES, is_crypto_usd_pair


BAR_SECONDS = 15 * 60
HORIZON_BARS = 16
EVENT_SECONDS = BAR_SECONDS * HORIZON_BARS
UNIVERSE_SIZE = 30
MAX_POSITIONS = 6
POSITION_FRACTION = 0.10
INITIAL_CASH = 1000.0
BASE_FEE_BPS = 26.0
BASE_SLIPPAGE_BPS = 5.0
HIGH_FEE_BPS = 52.0
HIGH_SLIPPAGE_BPS = 10.0

FEATURES = (
    "return_15m", "return_1h", "return_4h", "return_24h",
    "volatility_4h", "volatility_24h", "range_4h",
    "turnover_change", "log_turnover_24h", "trend_spread_10_30",
    "mean_reversion_z20", "btc_return_4h", "btc_return_24h",
    "relative_return_4h", "relative_return_24h",
)

_ALIASES = {
    "XBT": "BTC", "XXBT": "BTC", "XXBTZ": "BTC",
    "XDG": "DOGE", "XXDG": "DOGE", "XXDGZ": "DOGE",
    "XETH": "ETH", "XETHZ": "ETH", "XXETH": "ETH", "XXETHZ": "ETH",
    "XLTC": "LTC", "XLTCZ": "LTC", "XXMR": "XMR", "XXMRZ": "XMR",
    "XXRP": "XRP", "XXRPZ": "XRP", "XXLM": "XLM", "XXLMZ": "XLM",
}

_FOLDS = (
    ("2025-H1", "2025-01-01T00:00:00+00:00", "2025-07-01T00:00:00+00:00"),
    ("2025-H2", "2025-07-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    ("2026-H1", "2026-01-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00"),
    ("2026-Q3", "2026-07-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00"),
)
_VALIDATION_STARTS = {
    "2025-H1": "2024-07-01T00:00:00+00:00",
    "2025-H2": "2025-01-01T00:00:00+00:00",
    "2026-H1": "2025-07-01T00:00:00+00:00",
    "2026-Q3": "2026-01-01T00:00:00+00:00",
}


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def _pair_from_member(name: str) -> str | None:
    leaf = name.rsplit("/", 1)[-1]
    if not leaf.endswith("_15.csv"):
        return None
    code = leaf[:-7].upper()
    if not code.endswith("USD"):
        return None
    base = code[:-3]
    normalized = _ALIASES.get(base, base)
    if base in NON_CRYPTO_BASES or normalized in NON_CRYPTO_BASES:
        return None
    return f"{normalized}/USD"


def _archive_members(archive: zipfile.ZipFile) -> dict[str, str]:
    candidates: dict[str, list[str]] = {}
    for name in archive.namelist():
        pair = _pair_from_member(name)
        if pair:
            candidates.setdefault(pair, []).append(name)
    result: dict[str, str] = {}
    for pair, names in candidates.items():
        canonical_code = pair.split("/", 1)[0]
        result[pair] = min(
            names,
            key=lambda name: (
                0 if name.rsplit("/", 1)[-1][:-7].upper() == canonical_code + "USD" else 1,
                name,
            ),
        )
    return result


def _read_csv_member(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as source:
        rows = np.loadtxt(source, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.size == 0 or rows.ndim != 2 or rows.shape[1] < 7:
        return np.empty((0, 7), dtype=np.float64)
    rows = rows[:, :7]
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    rows = rows[(rows[:, 0] > 0) & (rows[:, 1:5].min(axis=1) > 0) & (rows[:, 5] >= 0)]
    if len(rows) > 1:
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        _, first = np.unique(rows[:, 0], return_index=True)
        rows = rows[np.sort(first)]
    return rows


def _read_pair(
    full: zipfile.ZipFile,
    quarterly: zipfile.ZipFile,
    full_name: str | None,
    quarterly_name: str | None,
    live_rows: np.ndarray | None = None,
) -> np.ndarray:
    parts = []
    if full_name:
        parts.append(_read_csv_member(full, full_name))
    if quarterly_name:
        parts.append(_read_csv_member(quarterly, quarterly_name))
    if live_rows is not None and len(live_rows):
        parts.append(live_rows)
    parts = [part for part in parts if len(part)]
    if not parts:
        return np.empty((0, 7), dtype=np.float64)
    return _merge_rows(parts)


def _merge_rows(parts: list[np.ndarray]) -> np.ndarray:
    parts = [part for part in parts if len(part)]
    if not parts:
        return np.empty((0, 7), dtype=np.float64)
    rows = np.concatenate(parts, axis=0)
    if len(rows) > 1:
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        _, reverse_first = np.unique(rows[::-1, 0], return_index=True)
        last = len(rows) - 1 - reverse_first
        rows = rows[np.sort(last)]
    return rows


def _read_live_database(path: Path) -> tuple[dict[str, np.ndarray], int]:
    if not path.is_file():
        return {}, 0
    grouped: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    count = 0
    with sqlite3.connect(path) as live_db:
        exists = live_db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='candles'"
        ).fetchone()
        if not exists:
            return {}, 0
        for row in live_db.execute(
            "SELECT pair, timestamp, open, high, low, close, volume "
            "FROM candles WHERE interval=15 ORDER BY pair, timestamp"
        ):
            raw_pair = str(row[0]).upper()
            if not is_crypto_usd_pair(raw_pair):
                continue
            base = raw_pair.split("/", 1)[0]
            pair = f"{_ALIASES.get(base, base)}/USD"
            grouped[pair].append((float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), 0.0))
            count += 1
    return {pair: np.asarray(rows, dtype=np.float64) for pair, rows in grouped.items()}, count


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < window:
        return result
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result[window - 1:] = prefix[window:] - prefix[:-window]
    return result


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_sum(values, window) / window


def _continuous_prefix(timestamps: np.ndarray) -> np.ndarray:
    broken = np.zeros(len(timestamps), dtype=np.int32)
    if len(timestamps) > 1:
        broken[1:] = (np.diff(timestamps) != BAR_SECONDS).astype(np.int32)
    return np.cumsum(broken, dtype=np.int64)


def _schema(connection: sqlite3.Connection) -> None:
    columns = ",\n".join(f"{feature} REAL NOT NULL" for feature in FEATURES)
    connection.executescript(
        f"""
        DROP TABLE IF EXISTS selected;
        DROP TABLE IF EXISTS events;
        CREATE TABLE events (
            ts INTEGER NOT NULL,
            pair TEXT NOT NULL,
            turnover REAL NOT NULL,
            gross_ratio REAL,
            entry_open REAL,
            exit_open REAL,
            trend_score REAL NOT NULL,
            mean_reversion_score REAL NOT NULL,
            {columns},
            PRIMARY KEY (ts, pair)
        );
        """
    )


def _insert_pair_events(
    connection: sqlite3.Connection,
    pair: str,
    rows: np.ndarray,
    btc_timestamps: np.ndarray,
    btc_closes: np.ndarray,
    btc_prefix: np.ndarray,
) -> tuple[int, int]:
    if len(rows) < 114:
        return 0, 0
    timestamps = rows[:, 0].astype(np.int64)
    opens, highs, lows, closes, volumes = (rows[:, i] for i in range(1, 6))
    n = len(rows)
    past_prefix = _continuous_prefix(timestamps)

    notional = closes * volumes
    turnover_4 = _rolling_sum(notional, 4)
    turnover_24 = _rolling_sum(notional, 96)
    ret = np.full(n, np.nan, dtype=np.float64)
    logret = np.full(n, np.nan, dtype=np.float64)
    ret[1:] = closes[1:] / closes[:-1] - 1.0
    logret[1:] = np.log(closes[1:] / closes[:-1])
    logret_safe = np.nan_to_num(logret, nan=0.0)
    vol4 = np.sqrt(np.maximum(0.0, _rolling_mean(logret_safe * logret_safe, 16) - _rolling_mean(logret_safe, 16) ** 2))
    vol24 = np.sqrt(np.maximum(0.0, _rolling_mean(logret_safe * logret_safe, 96) - _rolling_mean(logret_safe, 96) ** 2))
    ranges = (highs - lows) / closes
    range4 = _rolling_mean(ranges, 16)
    fast10 = _rolling_mean(closes, 10)
    slow30 = _rolling_mean(closes, 30)
    trend = fast10 / slow30 - 1.0
    mean20 = _rolling_mean(closes, 20)
    std20 = np.sqrt(np.maximum(0.0, _rolling_mean(closes * closes, 20) - mean20 * mean20))
    z20 = np.divide(closes - mean20, std20, out=np.zeros(n, dtype=np.float64), where=std20 > 0)
    meanrev_strength = np.maximum(0.0, -2.0 - z20)
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_change = np.log((turnover_4 / 4.0 + 1e-12) / (turnover_24 / 96.0 + 1e-12))
        log_turnover = np.log1p(turnover_24)

    event_indices = np.flatnonzero((timestamps % EVENT_SECONDS) == (EVENT_SECONDS - BAR_SECONDS))
    event_indices = event_indices[(event_indices >= 96) & (event_indices + 1 < n)]
    if len(event_indices) == 0:
        return 0, 0

    # Require a continuous past for each feature window. Future gaps are left
    # as unfilled labels; they never affect the point-in-time universe ranking.
    past_continuous = (past_prefix[event_indices] - past_prefix[event_indices - 96]) == 0
    event_indices = event_indices[past_continuous]
    if len(event_indices) == 0:
        return 0, 0

    btc_indices = np.searchsorted(btc_timestamps, timestamps[event_indices])
    btc_exact = btc_indices < len(btc_timestamps)
    btc_exact[btc_exact] &= btc_timestamps[btc_indices[btc_exact]] == timestamps[event_indices[btc_exact]]
    btc_valid = btc_exact.copy()
    btc_valid[btc_valid] &= btc_indices[btc_valid] >= 96
    valid_positions = np.flatnonzero(btc_valid)
    if len(valid_positions):
        btc_valid[valid_positions] &= (
            btc_prefix[btc_indices[valid_positions]] - btc_prefix[btc_indices[valid_positions] - 96]
        ) == 0
    event_indices = event_indices[btc_valid]
    btc_indices = btc_indices[btc_valid]
    if len(event_indices) == 0:
        return 0, 0

    r1 = ret[event_indices]
    r4 = closes[event_indices] / closes[event_indices - 4] - 1.0
    r16 = closes[event_indices] / closes[event_indices - 16] - 1.0
    r96 = closes[event_indices] / closes[event_indices - 96] - 1.0
    btc_r16 = btc_closes[btc_indices] / btc_closes[btc_indices - 16] - 1.0
    btc_r96 = btc_closes[btc_indices] / btc_closes[btc_indices - 96] - 1.0
    feature_matrix = np.column_stack((
        r1, r4, r16, r96,
        vol4[event_indices], vol24[event_indices], range4[event_indices],
        volume_change[event_indices], log_turnover[event_indices],
        trend[event_indices], z20[event_indices], btc_r16, btc_r96,
        r16 - btc_r16, r96 - btc_r96,
    ))
    turnover = turnover_24[event_indices]
    feature_ok = np.all(np.isfinite(feature_matrix), axis=1) & np.isfinite(turnover) & (turnover > 0)
    event_indices = event_indices[feature_ok]
    feature_matrix = feature_matrix[feature_ok]
    turnover = turnover[feature_ok]
    btc_indices = btc_indices[feature_ok]
    if len(event_indices) == 0:
        return 0, 0

    entry_indices = event_indices + 1
    exit_indices = event_indices + 17
    has_future = exit_indices < n
    safe_positions = np.flatnonzero(has_future)
    if len(safe_positions):
        valid_exit = (
            (timestamps[entry_indices[safe_positions]] == timestamps[event_indices[safe_positions]] + BAR_SECONDS)
            & (timestamps[exit_indices[safe_positions]] == timestamps[entry_indices[safe_positions]] + EVENT_SECONDS)
            & ((past_prefix[exit_indices[safe_positions]] - past_prefix[entry_indices[safe_positions]]) == 0)
        )
        has_future[safe_positions] = valid_exit

    gross_ratio = np.full(len(event_indices), np.nan, dtype=np.float64)
    entry_open = np.full(len(event_indices), np.nan, dtype=np.float64)
    exit_open = np.full(len(event_indices), np.nan, dtype=np.float64)
    good = np.flatnonzero(has_future)
    if len(good):
        entry_open[good] = opens[entry_indices[good]]
        exit_open[good] = opens[exit_indices[good]]
        gross_ratio[good] = exit_open[good] / entry_open[good]

    insert = "INSERT OR REPLACE INTO events VALUES (" + ",".join("?" for _ in range(23)) + ")"
    payload = []
    for j, index in enumerate(event_indices):
        payload.append((
            int(timestamps[index + 1]), pair, float(turnover[j]),
            None if not has_future[j] else float(gross_ratio[j]),
            None if not has_future[j] else float(entry_open[j]),
            None if not has_future[j] else float(exit_open[j]),
            float(max(0.0, trend[index])), float(meanrev_strength[index]),
            *(float(value) for value in feature_matrix[j]),
        ))
    connection.executemany(insert, payload)
    return len(payload), int(np.count_nonzero(~has_future))


def _prepare_events(full_path: Path, quarterly_path: Path, live_db_path: Path, db_path: Path) -> dict[str, int]:
    if not full_path.is_file() or not quarterly_path.is_file():
        raise FileNotFoundError("The verified Kraken full archive and Q3 2026 update must be present under data/kraken_ohlcvt_archive.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-150000")
    _schema(connection)

    live_series, live_candle_rows = _read_live_database(live_db_path)
    with zipfile.ZipFile(full_path) as full, zipfile.ZipFile(quarterly_path) as quarterly:
        full_members = _archive_members(full)
        quarterly_members = _archive_members(quarterly)
        pairs = sorted(set(full_members) | set(quarterly_members) | set(live_series))
        btc_pair = "BTC/USD"
        if btc_pair not in pairs:
            raise RuntimeError("Could not identify a Bitcoin/USD 15-minute archive member.")
        btc = _read_pair(full, quarterly, full_members.get(btc_pair), quarterly_members.get(btc_pair), live_series.get(btc_pair))
        btc_timestamps = btc[:, 0].astype(np.int64)
        btc_closes = btc[:, 4]
        btc_prefix = _continuous_prefix(btc_timestamps)
        total_rows = missing_labels = pairs_loaded = 0
        for pair in pairs:
            rows = btc if pair == btc_pair else _read_pair(
                full, quarterly, full_members.get(pair), quarterly_members.get(pair), live_series.get(pair),
            )
            count, missing = _insert_pair_events(connection, pair, rows, btc_timestamps, btc_closes, btc_prefix)
            total_rows += count
            missing_labels += missing
            pairs_loaded += int(count > 0)
            if pairs_loaded % 50 == 0:
                connection.commit()
        connection.commit()
    selected_columns = "ts, pair, turnover, gross_ratio, entry_open, exit_open, trend_score, mean_reversion_score, " + ", ".join(FEATURES)
    connection.executescript(
        f"""
        DROP TABLE IF EXISTS selected;
        CREATE TABLE selected AS
        SELECT {selected_columns}
        FROM (
            SELECT events.*, ROW_NUMBER() OVER (PARTITION BY ts ORDER BY turnover DESC, pair ASC) AS liquidity_rank
            FROM events
        )
        WHERE liquidity_rank <= {UNIVERSE_SIZE};
        CREATE INDEX selected_ts_idx ON selected(ts);
        CREATE INDEX selected_pair_ts_idx ON selected(pair, ts);
        """
    )
    selected_rows = connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    decision_events = connection.execute("SELECT COUNT(DISTINCT ts) FROM selected").fetchone()[0]
    connection.commit()
    connection.close()
    return {
        "archive_pairs_with_features": pairs_loaded,
        "point_in_time_universe_rows": int(selected_rows),
        "decision_events": int(decision_events),
        "rows_without_continuous_4h_label": missing_labels,
        "forward_candles_from_local_paper_database": live_candle_rows,
    }


def _cost_multiplier(fee_bps: float, slippage_bps: float) -> float:
    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0
    return ((1.0 - fee) * (1.0 - slip)) / ((1.0 + fee) * (1.0 + slip))


def _net_return(gross_ratio: float, fee_bps: float, slippage_bps: float) -> float:
    return gross_ratio * _cost_multiplier(fee_bps, slippage_bps) - 1.0


def _load_matrix(connection: sqlite3.Connection, where: str, args: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, ...]]]:
    projection = "ts, pair, gross_ratio, entry_open, exit_open, trend_score, mean_reversion_score, " + ", ".join(FEATURES)
    rows = connection.execute(f"SELECT {projection} FROM selected WHERE {where} ORDER BY ts, pair", args).fetchall()
    if not rows:
        return np.empty((0, len(FEATURES)), dtype=np.float32), np.empty(0, dtype=np.float64), rows
    x = np.asarray([row[7:] for row in rows], dtype=np.float32)
    gross = np.asarray([np.nan if row[2] is None else row[2] for row in rows], dtype=np.float64)
    return x, gross, rows


def _new_regressor() -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=100,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=False, random_state=42,
    )


def _fit_for_period(connection: sqlite3.Connection, end_timestamp: int) -> tuple[Any | None, int]:
    x, _, rows = _load_matrix(connection, "ts <= ? AND gross_ratio IS NOT NULL", (end_timestamp,))
    if len(rows) < 5000:
        return None, len(rows)
    y = np.asarray([_net_return(float(row[2]), BASE_FEE_BPS, BASE_SLIPPAGE_BPS) for row in rows], dtype=np.float64)
    model = _new_regressor()
    model.fit(x, y)
    return model, len(rows)


def _choose_validation_threshold(
    connection: sqlite3.Connection,
    start: int,
    end: int,
) -> tuple[float | None, dict[str, Any]]:
    validation_train_end = start - EVENT_SECONDS
    validation_model, train_count = _fit_for_period(connection, validation_train_end)
    if validation_model is None:
        return None, {"status": "insufficient_training_data", "training_samples": train_count, "candidates": []}
    validation_x, _, validation_rows = _load_matrix(connection, "ts >= ? AND ts < ?", (start, end))
    if not validation_rows:
        return None, {"status": "no_validation_rows", "training_samples": train_count, "candidates": []}
    predictions = validation_model.predict(validation_x)
    candidates = []
    thresholds = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
    for threshold in thresholds:
        metrics = _metrics(validation_rows, predictions, "model", BASE_FEE_BPS, BASE_SLIPPAGE_BPS, threshold)
        pf = metrics["profit_factor"]
        eligible = (
            metrics["closed_trades"] >= 30
            and metrics["return_pct"] > 0
            and (pf is None or pf >= 1.05)
            and metrics["max_drawdown_pct"] <= 25.0
        )
        candidates.append({"threshold_pct": round(threshold * 100.0, 3), "eligible": eligible, **metrics})
    eligible_candidates = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible_candidates:
        return None, {"status": "abstain_no_positive_validation_edge", "training_samples": train_count, "candidates": candidates}
    best = max(eligible_candidates, key=lambda candidate: (candidate["return_pct"], -candidate["max_drawdown_pct"]))
    return float(best["threshold_pct"]) / 100.0, {
        "status": "threshold_selected_on_prior_period",
        "training_samples": train_count,
        "selected": best,
        "candidates": candidates,
    }


def _metrics(
    rows: list[tuple[Any, ...]],
    predictions: np.ndarray | None,
    mode: str,
    fee_bps: float,
    slippage_bps: float,
    model_threshold: float = 0.0,
) -> dict[str, Any]:
    equity = INITIAL_CASH
    peak = equity
    max_drawdown = 0.0
    closed = wins = unfilled = 0
    gross_profit = gross_loss = 0.0
    event_equities: list[float] = []
    start = 0
    while start < len(rows):
        end = start + 1
        event_ts = rows[start][0]
        while end < len(rows) and rows[end][0] == event_ts:
            end += 1
        candidates = rows[start:end]
        scored: list[tuple[float, tuple[Any, ...]]] = []
        for offset, row in enumerate(candidates):
            if mode == "model":
                score = float(predictions[start + offset]) if predictions is not None else -math.inf
                eligible = score > model_threshold
            elif mode == "trend":
                score = float(row[5])
                eligible = score > 0
            elif mode == "mean_reversion":
                score = float(row[6])
                eligible = score > 0
            else:
                raise ValueError(f"Unknown strategy: {mode}")
            if eligible:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], str(item[1][1])))
        pnl = 0.0
        for _, row in scored[:MAX_POSITIONS]:
            if row[2] is None:
                unfilled += 1
                continue
            result = _net_return(float(row[2]), fee_bps, slippage_bps)
            position_pnl = equity * POSITION_FRACTION * result
            pnl += position_pnl
            closed += 1
            if result > 0:
                wins += 1
                gross_profit += position_pnl
            else:
                gross_loss -= position_pnl
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
        event_equities.append(equity)
        start = end
    return {
        "ending_equity": round(equity, 2),
        "return_pct": round((equity / INITIAL_CASH - 1.0) * 100.0, 3),
        "max_drawdown_pct": round(max_drawdown * 100.0, 3),
        "closed_trades": closed,
        "unfilled_selected_signals": unfilled,
        "win_rate_pct": round(wins / closed * 100.0, 2) if closed else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "decision_events": len(event_equities),
    }


def _buy_and_hold_btc(rows: list[tuple[Any, ...]], fee_bps: float, slippage_bps: float) -> dict[str, Any]:
    btc_rows = [row for row in rows if row[1] == "BTC/USD" and row[3] is not None and row[4] is not None]
    if len(btc_rows) < 2:
        return {"status": "unavailable", "reason": "BTC was not present in the selected historical universe."}
    exposure = POSITION_FRACTION * MAX_POSITIONS
    budget = INITIAL_CASH * exposure
    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0
    entry = float(btc_rows[0][3])
    exit_price = float(btc_rows[-1][4])
    quantity = budget / (entry * (1.0 + slip) * (1.0 + fee))
    cash = INITIAL_CASH - budget
    equities = [cash + quantity * float(row[3]) for row in btc_rows]
    ending = cash + quantity * exit_price * (1.0 - slip) * (1.0 - fee)
    peak = INITIAL_CASH
    drawdown = 0.0
    for mark in equities:
        peak = max(peak, mark)
        if peak > 0:
            drawdown = max(drawdown, (peak - mark) / peak)
    return {
        "ending_equity": round(ending, 2),
        "return_pct": round((ending / INITIAL_CASH - 1.0) * 100.0, 3),
        "max_drawdown_pct": round(drawdown * 100.0, 3),
        "closed_trades": 1,
        "position_fraction": exposure,
    }


def _fit_walk_forward(connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], Any, int]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("Install the optional learning dependencies with: .venv\\Scripts\\python.exe -m pip install -e .[learning]") from exc

    report: list[dict[str, Any]] = []
    trained_samples = 0
    latest_model = None
    for name, start_text, end_text in _FOLDS:
        start, end = _epoch(start_text), _epoch(end_text)
        cutoff = start - EVENT_SECONDS
        validation_start = _epoch(_VALIDATION_STARTS[name])
        threshold, validation = _choose_validation_threshold(connection, validation_start, start)
        train_x, _, train_rows = _load_matrix(connection, "ts <= ? AND gross_ratio IS NOT NULL", (cutoff,))
        if len(train_rows) < 5000:
            report.append({"period": name, "status": "skipped", "reason": f"Only {len(train_rows)} purged training samples.", "threshold_validation": validation})
            continue
        train_y = np.asarray([
            _net_return(float(row[2]), BASE_FEE_BPS, BASE_SLIPPAGE_BPS)
            for row in train_rows
        ], dtype=np.float64)
        model = _new_regressor()
        model.fit(train_x, train_y)
        test_x, _, test_rows = _load_matrix(connection, "ts >= ? AND ts < ?", (start, end))
        predictions = model.predict(test_x) if len(test_rows) else np.empty(0)
        active_threshold = math.inf if threshold is None else threshold
        base_metrics = {
            "learned_model": _metrics(test_rows, predictions, "model", BASE_FEE_BPS, BASE_SLIPPAGE_BPS, active_threshold),
            "trend_baseline": _metrics(test_rows, None, "trend", BASE_FEE_BPS, BASE_SLIPPAGE_BPS),
            "mean_reversion_baseline": _metrics(test_rows, None, "mean_reversion", BASE_FEE_BPS, BASE_SLIPPAGE_BPS),
            "btc_buy_and_hold": _buy_and_hold_btc(test_rows, BASE_FEE_BPS, BASE_SLIPPAGE_BPS),
            "cash": {"ending_equity": INITIAL_CASH, "return_pct": 0.0, "max_drawdown_pct": 0.0, "closed_trades": 0, "profit_factor": None},
        }
        high_threshold = max(active_threshold, _cost_multiplier(BASE_FEE_BPS, BASE_SLIPPAGE_BPS) / _cost_multiplier(HIGH_FEE_BPS, HIGH_SLIPPAGE_BPS) - 1.0)
        high_metrics = {
            "learned_model": _metrics(test_rows, predictions, "model", HIGH_FEE_BPS, HIGH_SLIPPAGE_BPS, high_threshold),
            "trend_baseline": _metrics(test_rows, None, "trend", HIGH_FEE_BPS, HIGH_SLIPPAGE_BPS),
            "mean_reversion_baseline": _metrics(test_rows, None, "mean_reversion", HIGH_FEE_BPS, HIGH_SLIPPAGE_BPS),
            "btc_buy_and_hold": _buy_and_hold_btc(test_rows, HIGH_FEE_BPS, HIGH_SLIPPAGE_BPS),
        }
        report.append({
            "period": name,
            "status": "evaluated",
            "training_samples": len(train_rows),
            "test_candidates": len(test_rows),
            "threshold_validation": validation,
            "selected_model_threshold_pct": None if threshold is None else round(threshold * 100.0, 3),
            "base_costs": {"fee_bps_per_side": BASE_FEE_BPS, "slippage_bps_per_side": BASE_SLIPPAGE_BPS, "results": base_metrics},
            "higher_cost_sensitivity": {"fee_bps_per_side": HIGH_FEE_BPS, "slippage_bps_per_side": HIGH_SLIPPAGE_BPS, "model_min_predicted_return_pct": None if not math.isfinite(high_threshold) else round(high_threshold * 100.0, 3), "results": high_metrics},
        })
        trained_samples = max(trained_samples, len(train_rows))
        latest_model = model

    return report, latest_model, trained_samples


def _fit_final_model(connection: sqlite3.Connection) -> tuple[Any, int, int]:
    x, _, rows = _load_matrix(connection, "gross_ratio IS NOT NULL", ())
    if len(rows) < 5000:
        raise RuntimeError(f"Only {len(rows)} complete samples are available; refusing to fit a small, unstable candidate.")
    y = np.asarray([_net_return(float(row[2]), BASE_FEE_BPS, BASE_SLIPPAGE_BPS) for row in rows], dtype=np.float64)
    model = _new_regressor()
    model.fit(x, y)
    return model, len(rows), max(int(row[0]) for row in rows)


def run_historical_learning(
    full_archive: str | Path = "data/kraken_ohlcvt_archive/Kraken_OHLCVT_Full.zip",
    quarterly_archive: str | Path = "data/kraken_ohlcvt_archive/Kraken_OHLCVT_2026Q3.zip",
    live_db: str | Path = "data/market_research.sqlite3",
    research_db: str | Path = "data/historical_learning.sqlite3",
    report_path: str | Path = "data/historical_learning_report.json",
    model_path: str | Path = "data/models/return_4h_hist_gradient.joblib",
) -> dict[str, Any]:
    full_archive = Path(full_archive)
    quarterly_archive = Path(quarterly_archive)
    live_db = Path(live_db)
    research_db = Path(research_db)
    report_path = Path(report_path)
    model_path = Path(model_path)

    preparation = _prepare_events(full_archive, quarterly_archive, live_db, research_db)
    connection = sqlite3.connect(research_db)
    folds, _, _ = _fit_walk_forward(connection)
    final_model, final_samples, trained_through = _fit_final_model(connection)

    # The final fit is an offline research artifact for a later paper-only test.
    # It is not wired into the running paper accounts.
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("The learning installation is incomplete: joblib is missing.") from exc
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    metadata_path = model_path.with_suffix(".json")
    metadata = {
        "purpose": "offline research candidate; not connected to live or paper order decisions",
        "target": "net return from next 15-minute open to open 4 hours later",
        "feature_names": list(FEATURES),
        "trained_through_entry_timestamp_utc": datetime.fromtimestamp(trained_through, tz=timezone.utc).isoformat(),
        "training_samples": final_samples,
        "fee_bps_per_side": BASE_FEE_BPS,
        "slippage_bps_per_side": BASE_SLIPPAGE_BPS,
        "universe": f"Top {UNIVERSE_SIZE} USD crypto pairs by trailing 24-hour quote turnover at each decision time",
        "horizon_bars": HORIZON_BARS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    connection.close()

    evaluated = [fold for fold in folds if fold["status"] == "evaluated"]
    base_positive = sum(fold["base_costs"]["results"]["learned_model"]["return_pct"] > 0 for fold in evaluated)
    high_positive = sum(fold["higher_cost_sensitivity"]["results"]["learned_model"]["return_pct"] > 0 for fold in evaluated)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Historical research and offline model fit only. No model is connected to order execution.",
        "data": {
            "full_archive": str(full_archive),
            "quarterly_archive": str(quarterly_archive),
            "forward_paper_database": str(live_db),
            **preparation,
            "training_samples_for_final_fit": final_samples,
        },
        "label": "Net return after entry and exit fees and adverse slippage; entry is next 15-minute open, exit is 16 bars after entry.",
        "validation": {
            "method": "Expanding chronological walk-forward; each training set ends at least four hours before its test period.",
            "periods": folds,
            "learned_model_positive_periods_base_costs": base_positive,
            "learned_model_positive_periods_higher_costs": high_positive,
            "research_gate_passed": bool(evaluated) and base_positive == len(evaluated) and high_positive >= max(2, math.ceil(len(evaluated) * 0.75)),
            "gate_definition": "Positive model return in every evaluated time period at base costs and at least 75% of periods at doubled illustrative fees/slippage. Passing is only a paper-test candidate gate, not evidence of guaranteed profit.",
        },
        "offline_model": {"path": str(model_path), "metadata_path": str(metadata_path), "connected_to_bot": False},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
