from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timezone
import json
import math
import re
import time as time_module
import warnings
from typing import Any
from zoneinfo import ZoneInfo

from .local_model import LocalOllamaAdvisor
from .models import Candle, EquityPoint, Position, SimulationResult, Trade
from .paper import NoNewCandles, paper_summary
from .simulator import Costs
from .storage import MarketStore
from .strategies import STRATEGIES, MeanReversion, Strategy, TrendFollowing


US_EASTERN = ZoneInfo("America/New_York")
DAILY_INTERVAL_MINUTES = 1440
PAPER_ACCOUNT_ID = "stocks_llama_regime"
DEFAULT_STOCK_SYMBOLS = (
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN",
    "INTC", "TSLA", "SPCX",
)
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
MARKET_CLOSE_CHECK = time(16, 15)


def _symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"Invalid US equity ticker: {value!r}")
    return symbol


def _session_date_to_timestamp(value: Any) -> int:
    session_date = value.date() if hasattr(value, "date") else value
    session = datetime.combine(session_date, time.min, tzinfo=US_EASTERN)
    return int(session.timestamp())


def fetch_equity_history(
    symbols: list[str] | tuple[str, ...] = DEFAULT_STOCK_SYMBOLS,
    *, period: str = "5y", timeout: int = 30,
) -> tuple[dict[str, list[Candle]], list[dict[str, object]]]:
    """Fetch daily, un-dividend-adjusted US equity candles and action events."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install the stock data extra with: python -m pip install -e ".[stocks]"') from exc

    normalized = list(dict.fromkeys(_symbol(symbol) for symbol in symbols))
    if not normalized:
        raise ValueError("At least one ticker is required")
    now_et = datetime.now(US_EASTERN)
    may_include_today = now_et.weekday() < 5 and now_et.time() >= MARKET_CLOSE_CHECK
    with warnings.catch_warnings():
        try:
            from pandas.errors import Pandas4Warning
            warnings.simplefilter("ignore", Pandas4Warning)
        except ImportError:
            pass
        frame = yf.download(
            tickers=normalized,
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=True,
            prepost=False,
            repair=True,
            progress=False,
            threads=True,
            group_by="ticker",
            multi_level_index=True,
            timeout=timeout,
        )
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo Finance returned no daily bars")

    from pandas import MultiIndex

    candles: dict[str, list[Candle]] = {}
    actions: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for symbol in normalized:
        if isinstance(frame.columns, MultiIndex):
            levels = [frame.columns.get_level_values(level) for level in range(frame.columns.nlevels)]
            if symbol in levels[0]:
                rows = frame.xs(symbol, axis=1, level=0, drop_level=True)
            elif symbol in levels[-1]:
                rows = frame.xs(symbol, axis=1, level=frame.columns.nlevels - 1, drop_level=True)
            else:
                errors.append({"symbol": symbol, "error": "ticker missing from provider response"})
                continue
        else:
            rows = frame
        rows = rows.sort_index()
        bars: list[Candle] = []
        for index, row in rows.iterrows():
            session_date = index.date() if hasattr(index, "date") else index
            if isinstance(session_date, datetime):
                session_date = session_date.date()
            if session_date > now_et.date() or (session_date == now_et.date() and not may_include_today):
                continue
            try:
                open_price = float(row["Open"])
                high = float(row["High"])
                low = float(row["Low"])
                close = float(row["Close"])
                volume_value = float(row["Volume"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (open_price, high, low, close)):
                continue
            volume = volume_value if math.isfinite(volume_value) and volume_value >= 0 else 0.0
            timestamp = _session_date_to_timestamp(session_date)
            bars.append(Candle(symbol, DAILY_INTERVAL_MINUTES, timestamp, open_price, high, low, close, volume))

            dividend = row.get("Dividends", 0.0)
            split_ratio = row.get("Stock Splits", 0.0)
            try:
                dividend = float(dividend) if math.isfinite(float(dividend)) else 0.0
                split_ratio = float(split_ratio) if math.isfinite(float(split_ratio)) else 0.0
            except (TypeError, ValueError):
                dividend, split_ratio = 0.0, 0.0
            if dividend > 0 or split_ratio > 0:
                actions.append({
                    "pair": symbol, "timestamp": timestamp,
                    "dividend": dividend, "split_ratio": split_ratio,
                })
        if bars:
            candles[symbol] = bars
        else:
            errors.append({"symbol": symbol, "error": "no completed daily bars in provider response"})
    if not candles:
        raise RuntimeError("No completed daily equity bars were available")
    return candles, errors + actions  # actions are tagged with pair/timestamp plus action fields


def refresh_equity_data(
    store: MarketStore, symbols: list[str] | tuple[str, ...], *, period: str,
) -> dict[str, object]:
    candles, mixed_rows = fetch_equity_history(symbols, period=period)
    errors = [row for row in mixed_rows if "error" in row]
    actions = [row for row in mixed_rows if "error" not in row]
    saved_by_symbol = {
        symbol: store.save_candles(rows)
        for symbol, rows in candles.items()
    }
    store.save_corporate_actions(actions)
    return {
        "provider": "Yahoo Finance via yfinance",
        "interval_minutes": DAILY_INTERVAL_MINUTES,
        "fetched_symbols": len(candles),
        "symbols": sorted(candles),
        "saved_rows_by_symbol": saved_by_symbol,
        "errors": errors,
        "bars_are_dividend_unadjusted": True,
        "today_session_included_only_after_1615_et": True,
    }


class EquityLlamaRegimeSwitch:
    """Daily-bar regime classifier; Llama selects a fixed long-only baseline."""

    name = PAPER_ACCOUNT_ID
    minimum_confidence = 0.60
    batch_size = 3

    def __init__(self, model: str = LocalOllamaAdvisor.DEFAULT_MODEL) -> None:
        self.model = model
        self.advisor = LocalOllamaAdvisor(model)
        self.trend = TrendFollowing()
        self.mean_reversion = MeanReversion()
        self._labels: dict[tuple[str, int], str] = {}
        self._notes: dict[str, dict[str, object]] = {}

    @property
    def notes(self) -> dict[str, dict[str, object]]:
        return {pair: dict(note) for pair, note in self._notes.items()}

    def prepare_batch(self, histories: dict[str, list[Candle]]) -> None:
        eligible = {pair: bars for pair, bars in histories.items() if len(bars) >= 30}
        pairs = list(eligible)
        for offset in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[offset:offset + self.batch_size]
            batch = {pair: eligible[pair] for pair in batch_pairs}
            try:
                notes = self.advisor.classify_batch(
                    {pair: [bar.close for bar in bars[-61:]] for pair, bars in batch.items()},
                    timeout=90,
                )
            except RuntimeError as exc:
                for pair, bars in batch.items():
                    last = bars[-1]
                    self._labels[(pair, last.timestamp)] = "unavailable"
                    self._notes[pair] = {
                        "regime": "unavailable", "confidence": 0.0,
                        "rationale": f"Local model unavailable; fail-closed. {str(exc)[:160]}",
                        "decision_timestamp": last.timestamp, "decision_close": last.close,
                    }
                continue
            for pair, note in notes.items():
                last = batch[pair][-1]
                label = note.regime if note.confidence >= self.minimum_confidence else "unknown"
                self._labels[(pair, last.timestamp)] = label
                self._notes[pair] = {
                    "regime": label, "predicted_regime": note.regime,
                    "confidence": note.confidence, "rationale": note.rationale,
                    "decision_timestamp": last.timestamp, "decision_close": last.close,
                    "timeframe": "daily", "news": [],
                }

    def _label(self, history: list[Candle]) -> str:
        if len(history) < 30:
            return "unknown"
        last = history[-1]
        return self._labels.get((last.pair, last.timestamp), "unknown")

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        label = self._label(history)
        if label == "trend":
            return self.trend.desired_long(history, currently_long)
        if label == "range":
            return self.mean_reversion.desired_long(history, currently_long)
        if label in {"high_volatility", "unavailable"}:
            return False
        return currently_long

    def entry_strength(self, history: list[Candle]) -> float:
        label = self._label(history)
        if label == "trend":
            return self.trend.entry_strength(history)
        if label == "range":
            return self.mean_reversion.entry_strength(history)
        return 0.0


class EquitySimulator:
    """Long-only equity simulator using close fills and explicit cash dividends."""

    def __init__(
        self, initial_cash: float = 1000.0, costs: Costs = Costs(0.0, 5.0),
        max_position_fraction: float = 0.10, max_positions: int = 6,
    ) -> None:
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        if not 0 < max_position_fraction <= 1 or max_positions < 1:
            raise ValueError("Invalid position limits")
        self.initial_cash = initial_cash
        self.costs = costs
        self.max_position_fraction = max_position_fraction
        self.max_positions = max_positions

    def run(
        self, candles_by_symbol: dict[str, list[Candle]], strategy: Strategy,
        actions: dict[str, dict[int, dict[str, float]]] | None = None,
        *, from_timestamp: int | None = None, through_timestamp: int | None = None,
    ) -> SimulationResult:
        normalized = {
            symbol: sorted(rows, key=lambda bar: bar.timestamp)
            for symbol, rows in candles_by_symbol.items() if rows
        }
        if not normalized:
            raise ValueError("No equity candles available")
        events: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for symbol, bars in normalized.items():
            for index in range(1, len(bars)):
                timestamp = bars[index].timestamp
                if from_timestamp is not None and timestamp < from_timestamp:
                    continue
                if through_timestamp is not None and timestamp > through_timestamp:
                    continue
                events[timestamp].append((symbol, index))
        if not events:
            raise ValueError("No equity bar events fall within the requested period")

        cash = self.initial_cash
        positions: dict[str, Position] = {}
        open_trades: dict[str, Trade] = {}
        completed: list[Trade] = []
        marks: dict[str, float] = {}
        curve: list[EquityPoint] = []
        fee, slip = self.costs.fee_rate, self.costs.slippage_rate
        for timestamp in sorted(events):
            rows = sorted(events[timestamp], key=lambda item: item[0])
            for symbol, index in rows:
                candle = normalized[symbol][index]
                action = ((actions or {}).get(symbol, {})).get(timestamp, {})
                position = positions.get(symbol)
                if position:
                    # Yahoo's historical OHLC has split-adjusted prices; do not
                    # apply historical split ratios a second time in backtests.
                    cash += position.quantity * float(action.get("dividend", 0.0))
                marks[symbol] = candle.close

            for symbol, index in rows:
                position = positions.get(symbol)
                if position and not strategy.desired_long(normalized[symbol][:index], currently_long=True):
                    base = normalized[symbol][index].close
                    fill = base * (1 - slip)
                    proceeds = position.quantity * fill * (1 - fee)
                    cash += proceeds
                    trade = open_trades.pop(symbol)
                    trade.exit_timestamp = timestamp
                    trade.exit_price = fill
                    trade.exit_proceeds = proceeds
                    trade.pnl = proceeds - position.entry_cost
                    completed.append(trade)
                    del positions[symbol]

            candidates: list[tuple[float, str, int]] = []
            for symbol, index in rows:
                if symbol in positions:
                    continue
                history = normalized[symbol][:index]
                if strategy.desired_long(history, currently_long=False):
                    candidates.append((strategy.entry_strength(history), symbol, index))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _, symbol, index in candidates:
                if len(positions) >= self.max_positions or cash <= 0:
                    break
                base = normalized[symbol][index].close
                fill = base * (1 + slip)
                equity_before = cash + sum(
                    position.quantity * marks.get(position.pair, position.entry_price)
                    for position in positions.values()
                )
                budget = min(cash, equity_before * self.max_position_fraction)
                quantity = budget / (fill * (1 + fee))
                entry_cost = quantity * fill * (1 + fee)
                if quantity <= 0 or entry_cost > cash:
                    continue
                cash -= entry_cost
                positions[symbol] = Position(symbol, quantity, timestamp, fill, entry_cost)
                open_trades[symbol] = Trade(symbol, timestamp, None, quantity, fill, None, entry_cost, None, None)

            equity = cash + sum(
                position.quantity * marks.get(symbol, position.entry_price)
                for symbol, position in positions.items()
            )
            curve.append(EquityPoint(timestamp, equity))

        for symbol, bars in normalized.items():
            eligible = [bar for bar in bars if through_timestamp is None or bar.timestamp <= through_timestamp]
            if eligible:
                marks[symbol] = eligible[-1].close
        ending_equity = cash + sum(
            position.quantity * marks.get(symbol, position.entry_price)
            for symbol, position in positions.items()
        )
        closed_pnls = [float(trade.pnl or 0.0) for trade in completed]
        gross_profit = sum(pnl for pnl in closed_pnls if pnl > 0)
        gross_loss = -sum(pnl for pnl in closed_pnls if pnl < 0)
        peak, max_drawdown = self.initial_cash, 0.0
        for point in curve:
            peak = max(peak, point.equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - point.equity) / peak)
        open_positions = [
            {"pair": pos.pair, "quantity": pos.quantity, "entry_timestamp": pos.entry_timestamp,
             "entry_price": pos.entry_price, "mark_price": marks.get(pos.pair, pos.entry_price),
             "unrealized_pnl": pos.quantity * marks.get(pos.pair, pos.entry_price) - pos.entry_cost}
            for pos in positions.values()
        ]
        count = len(completed)
        return SimulationResult(
            strategy=strategy.name, initial_cash=self.initial_cash, ending_equity=ending_equity,
            return_pct=(ending_equity / self.initial_cash - 1) * 100,
            max_drawdown_pct=max_drawdown * 100, closed_trades=count,
            win_rate_pct=(sum(pnl > 0 for pnl in closed_pnls) / count * 100) if count else 0.0,
            profit_factor=(gross_profit / gross_loss) if gross_loss else None,
            cash=cash, open_positions=open_positions, trades=completed, equity_curve=curve,
        )


def start_equity_paper_account(
    store: MarketStore, *, symbols: list[str], starting_cash: float = 1000.0,
    costs: Costs = Costs(0.0, 5.0), max_position_fraction: float = 0.10,
    max_positions: int = 6, model: str = LocalOllamaAdvisor.DEFAULT_MODEL,
    reset: bool = False,
) -> dict[str, Any]:
    existing = store.get_paper_state(PAPER_ACCOUNT_ID)
    if existing and existing.get("active") and not reset:
        raise ValueError("The stock paper account already exists; use status/watch to resume it, or pass --reset to replace it")
    data = store.get_candles(DAILY_INTERVAL_MINUTES, symbols)
    if not data:
        raise ValueError("No stock candles found. Run `stocks fetch` first.")
    EquitySimulator(starting_cash, costs, max_position_fraction, max_positions)
    pairs = sorted(data)
    cursor = max(bars[-1].timestamp for bars in data.values() if bars)
    marks = {pair: bars[-1].close for pair, bars in data.items() if bars}
    state: dict[str, Any] = {
        "active": True, "market": "US equities", "data_provider": "Yahoo Finance via yfinance",
        "interval": DAILY_INTERVAL_MINUTES, "pairs": pairs,
        "strategy": PAPER_ACCOUNT_ID, "model": model,
        "starting_cash": starting_cash, "cash": starting_cash,
        "fee_bps": costs.fee_bps, "slippage_bps": costs.slippage_bps,
        "max_position_fraction": max_position_fraction, "max_positions": max_positions,
        "cursor": cursor, "start_timestamp": cursor,
        "positions": {}, "open_trades": {}, "closed_trades": [],
        "equity_curve": [], "last_marks": marks, "dividend_income": 0.0,
        "last_processing_mode": "waiting_for_forward_session",
        "last_check_timestamp": int(time_module.time()),
        "watcher_heartbeat_timestamp": int(time_module.time()),
        "market_status": {"requested_markets": len(pairs), "successful_markets": len(marks),
                           "failed_markets": 0, "state": "waiting_for_next_us_close"},
    }
    store.save_paper_state(state, PAPER_ACCOUNT_ID)
    return state


def extend_equity_paper_universe(store: MarketStore, symbols: list[str]) -> dict[str, Any]:
    """Add fetched symbols to the active forward account without resetting it."""
    state = store.get_paper_state(PAPER_ACCOUNT_ID)
    if not state or not state.get("active"):
        raise ValueError("No active stock paper account. Run stocks paper start first.")
    requested = list(dict.fromkeys(_symbol(symbol) for symbol in symbols))
    if not requested:
        raise ValueError("Provide at least one ticker to add")
    existing = set(state["pairs"])
    additions = [symbol for symbol in requested if symbol not in existing]
    if not additions:
        return {**paper_summary(state), "added_symbols": []}

    data = store.get_candles(DAILY_INTERVAL_MINUTES, additions)
    missing = [symbol for symbol in additions if not data.get(symbol)]
    if missing:
        raise ValueError("Fetch daily data before adding these symbols: " + ", ".join(missing))
    cursor = int(state["cursor"])
    stale = [symbol for symbol in additions if data[symbol][-1].timestamp < cursor]
    if stale:
        raise ValueError("Latest stored bars are older than the account cursor for: " + ", ".join(stale))

    pairs = sorted(existing | set(additions))
    marks = dict(state.get("last_marks", {}))
    marks.update({symbol: data[symbol][-1].close for symbol in additions})
    state["pairs"] = pairs
    state["last_marks"] = marks
    state["last_processing_mode"] = "universe_extended_waiting_for_next_session"
    state["last_check_timestamp"] = int(time_module.time())
    state["market_status"] = {
        "requested_markets": len(pairs), "successful_markets": len(marks),
        "failed_markets": max(0, len(pairs) - len(marks)),
        "state": "universe_extended_waiting_for_next_session",
    }
    state["universe_updated_at"] = int(time_module.time())
    store.save_paper_state(state, PAPER_ACCOUNT_ID)
    return {**paper_summary(state), "added_symbols": additions}


def advance_equity_paper_account(store: MarketStore) -> dict[str, Any]:
    state = store.get_paper_state(PAPER_ACCOUNT_ID)
    if not state or not state.get("active"):
        raise ValueError("No active stock paper account. Run `stocks paper start` first.")
    pairs = list(state["pairs"])
    data = store.get_candles(DAILY_INTERVAL_MINUTES, pairs)
    cursor = int(state["cursor"])
    new_timestamps = sorted({
        bar.timestamp for bars in data.values() for bar in bars if bar.timestamp > cursor
    })
    if not new_timestamps:
        raise NoNewCandles("No new completed US session is available")
    timestamp = max(new_timestamps)
    rows: dict[str, tuple[int, Candle]] = {}
    for pair, bars in data.items():
        for index in range(len(bars) - 1, -1, -1):
            if bars[index].timestamp == timestamp:
                rows[pair] = (index, bars[index])
                break
    if not rows:
        raise NoNewCandles("No synchronized completed US session is available")

    actions = store.get_corporate_actions(pairs, through_timestamp=timestamp)
    action_today = {pair: action_rows.get(timestamp, {}) for pair, action_rows in actions.items()}
    positions: dict[str, dict[str, Any]] = state["positions"]
    open_trades: dict[str, dict[str, Any]] = state["open_trades"]
    marks: dict[str, float] = state["last_marks"]
    cash = float(state["cash"])
    fee_rate = float(state["fee_bps"]) / 10_000
    slip_rate = float(state["slippage_bps"]) / 10_000

    # Apply splits and dividends before any close-time exits or new entries.
    dividend_today = 0.0
    for pair, (index, candle) in rows.items():
        position = positions.get(pair)
        if not position:
            continue
        action = action_today.get(pair, {})
        split_ratio = float(action.get("split_ratio", 0.0))
        if split_ratio > 0:
            position["quantity"] = float(position["quantity"]) * split_ratio
            position["entry_price"] = float(position["entry_price"]) / split_ratio
            open_trades[pair]["quantity"] = float(open_trades[pair]["quantity"]) * split_ratio
            open_trades[pair]["entry_price"] = float(open_trades[pair]["entry_price"]) / split_ratio
        dividend_per_share = float(action.get("dividend", 0.0))
        if dividend_per_share > 0:
            income = float(position["quantity"]) * dividend_per_share
            cash += income
            dividend_today += income
        marks[pair] = candle.close

    missed = sorted(value for value in new_timestamps if value < timestamp)
    pending_targets = state.get("pending_targets", {})
    allow_pending_fills = bool(state.pop("allow_pending_fills", True))
    stale_pending = bool(pending_targets) and (bool(missed) or not allow_pending_fills)

    # Classifications are produced after today's close. They become eligible
    # for the following session only. If the watcher missed a session, discard
    # stale targets rather than pretending they filled while it was down.
    if not stale_pending:
        for pair, (index, candle) in sorted(rows.items()):
            target = pending_targets.get(pair, {})
            wants_long = bool(target.get("long", False)) if isinstance(target, dict) else bool(target)
            position = positions.get(pair)
            if position and not wants_long:
                fill = candle.close * (1 - slip_rate)
                proceeds = float(position["quantity"]) * fill * (1 - fee_rate)
                cash += proceeds
                trade = open_trades.pop(pair)
                trade.update({"exit_timestamp": timestamp, "exit_price": fill,
                              "exit_proceeds": proceeds, "pnl": proceeds - float(position["entry_cost"]),
                              "exit_reason": "prior_close_signal"})
                state["closed_trades"].append(trade)
                del positions[pair]

    candidates = []
    if not stale_pending:
        for pair, (index, candle) in rows.items():
            target = pending_targets.get(pair, {})
            wants_long = bool(target.get("long", False)) if isinstance(target, dict) else bool(target)
            if pair not in positions and wants_long:
                strength = float(target.get("strength", 0.0)) if isinstance(target, dict) else 0.0
                candidates.append((strength, pair, candle.close))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, pair, base_price in candidates:
        if len(positions) >= int(state["max_positions"]) or cash <= 0:
            break
        fill = base_price * (1 + slip_rate)
        equity_before = cash + sum(
            float(position["quantity"]) * float(marks.get(name, position["entry_price"]))
            for name, position in positions.items()
        )
        budget = min(cash, equity_before * float(state["max_position_fraction"]))
        quantity = budget / (fill * (1 + fee_rate))
        entry_cost = quantity * fill * (1 + fee_rate)
        if quantity <= 0 or entry_cost > cash:
            continue
        cash -= entry_cost
        positions[pair] = {"pair": pair, "quantity": quantity, "entry_timestamp": timestamp,
                           "entry_price": fill, "entry_cost": entry_cost}
        open_trades[pair] = {"pair": pair, "entry_timestamp": timestamp, "exit_timestamp": None,
                             "quantity": quantity, "entry_price": fill, "exit_price": None,
                             "entry_cost": entry_cost, "exit_proceeds": None, "pnl": None}

    # Today's completed prices form tomorrow's targets. The model never sees
    # an unfinished candle or fills against the close that created its signal.
    strategy = EquityLlamaRegimeSwitch(str(state.get("model") or LocalOllamaAdvisor.DEFAULT_MODEL))
    histories = {pair: bars[:index + 1] for pair, (index, _) in rows.items()}
    strategy.prepare_batch({pair: history for pair, history in histories.items() if len(history) >= 30})
    next_targets: dict[str, dict[str, Any]] = {}
    for pair, history in histories.items():
        wants_long = strategy.desired_long(history, currently_long=pair in positions)
        next_targets[pair] = {
            "long": wants_long,
            "strength": strategy.entry_strength(history) if wants_long and pair not in positions else 0.0,
            "signal_timestamp": timestamp,
        }
    for note in strategy.notes.values():
        note["target_applies_after_timestamp"] = timestamp

    equity = cash + sum(
        float(position["quantity"]) * float(marks.get(pair, position["entry_price"]))
        for pair, position in positions.items()
    )
    previous_cursor = cursor
    state["cash"] = cash
    state["cursor"] = timestamp
    state["pending_targets"] = next_targets
    state["dividend_income"] = float(state.get("dividend_income", 0.0)) + dividend_today
    state["last_check_timestamp"] = int(time_module.time())
    state["last_processing_mode"] = "live_forward_daily_close_skipped_stale_targets" if stale_pending else "live_forward_daily_close"
    state["market_status"] = {
        "requested_markets": len(pairs), "successful_markets": len(rows),
        "failed_markets": max(0, len(pairs) - len(rows)), "state": "processed_completed_session",
    }
    if missed:
        state["last_downtime_gap"] = {
            "mode": "skipped_missed_sessions", "from_cursor": previous_cursor,
            "resumed_at": timestamp, "skipped_session_timestamps": missed,
        }
    if not allow_pending_fills:
        state["last_downtime_gap"] = {
            "mode": "skipped_stale_target_after_watcher_downtime",
            "resumed_at": timestamp,
        }
    state["model_notes"] = strategy.notes
    state["equity_curve"].append({"timestamp": timestamp, "equity": equity})
    store.save_paper_state(state, PAPER_ACCOUNT_ID)
    return paper_summary(state)


def watch_equity_paper_account(
    store: MarketStore, poll_seconds: int = 60, once: bool = False,
) -> None:
    if poll_seconds < 30:
        raise ValueError("Poll interval must be at least 30 seconds")
    state = store.get_paper_state(PAPER_ACCOUNT_ID)
    if not state or not state.get("active"):
        raise ValueError("No active stock paper account. Run `stocks paper start` first.")
    symbols = list(state["pairs"])
    last_attempt_date: date | None = None
    while True:
        now_et = datetime.now(US_EASTERN)
        if now_et.weekday() < 5 and now_et.time() >= MARKET_CLOSE_CHECK and last_attempt_date != now_et.date():
            try:
                now_timestamp = int(time_module.time())
                heartbeat = int(state.get("watcher_heartbeat_timestamp", 0))
                continuity_ok = heartbeat > 0 and now_timestamp - heartbeat <= max(180, poll_seconds * 3)
                fetch_result = refresh_equity_data(store, symbols, period="10d")
                last_attempt_date = now_et.date()
                state = store.get_paper_state(PAPER_ACCOUNT_ID) or state
                state["allow_pending_fills"] = continuity_ok
                state["last_check_timestamp"] = int(time_module.time())
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": fetch_result["fetched_symbols"],
                    "failed_markets": len(fetch_result["errors"]), "state": "daily_fetch_complete",
                    "errors": fetch_result["errors"],
                }
                store.save_paper_state(state, PAPER_ACCOUNT_ID)
                try:
                    summary = advance_equity_paper_account(store)
                    print(json.dumps(summary, allow_nan=False), flush=True)
                except NoNewCandles:
                    current = store.get_paper_state(PAPER_ACCOUNT_ID) or state
                    current["market_status"]["state"] = "no_new_us_session"
                    store.save_paper_state(current, PAPER_ACCOUNT_ID)
            except Exception as exc:
                state = store.get_paper_state(PAPER_ACCOUNT_ID) or state
                state["last_check_timestamp"] = int(time_module.time())
                state["feed_errors"] = [{"error": str(exc)}]
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": 0,
                    "failed_markets": len(symbols), "state": "provider_error",
                }
                store.save_paper_state(state, PAPER_ACCOUNT_ID)
                print(json.dumps({"equity_data_error": str(exc)}), flush=True)
        else:
            state = store.get_paper_state(PAPER_ACCOUNT_ID) or state
            state["last_check_timestamp"] = int(time_module.time())
            state["watcher_heartbeat_timestamp"] = int(time_module.time())
            if now_et.weekday() >= 5 or now_et.time() < MARKET_CLOSE_CHECK:
                state["market_status"] = {
                    "requested_markets": len(symbols),
                    "successful_markets": len(state.get("last_marks", {})),
                    "failed_markets": 0, "state": "waiting_for_us_market_close",
                }
            store.save_paper_state(state, PAPER_ACCOUNT_ID)
        if once:
            return
        time_module.sleep(poll_seconds)


def _result_summary(result: SimulationResult) -> dict[str, object]:
    return {
        "strategy": result.strategy, "ending_equity": result.ending_equity,
        "pnl": result.ending_equity - result.initial_cash, "return_pct": result.return_pct,
        "max_drawdown_pct": result.max_drawdown_pct, "closed_trades": result.closed_trades,
        "open_positions": len(result.open_positions),
    }


def run_equity_backtests(
    store: MarketStore, symbols: list[str], *, initial_cash: float = 1000.0,
    costs: Costs = Costs(0.0, 5.0), max_position_fraction: float = 0.10,
    max_positions: int = 6,
) -> dict[str, Any]:
    data = store.get_candles(DAILY_INTERVAL_MINUTES, symbols)
    if not data:
        raise ValueError("No stored equity candles. Run `stocks fetch` first.")
    actions = store.get_corporate_actions(list(data))
    simulator = EquitySimulator(initial_cash, costs, max_position_fraction, max_positions)
    start = max(bars[0].timestamp for bars in data.values() if bars)
    end = min(bars[-1].timestamp for bars in data.values() if bars)
    if start >= end:
        raise ValueError("At least two overlapping market sessions are needed for a comparison")
    results: list[dict[str, object]] = [
        {"strategy": "cash", "ending_equity": initial_cash, "pnl": 0.0,
         "return_pct": 0.0, "max_drawdown_pct": 0.0, "closed_trades": 0, "open_positions": 0},
        _buy_hold(data, actions, sorted(data), start, end, initial_cash, costs),
    ]
    if "SPY" in data:
        results.append(_buy_hold(data, actions, ["SPY"], start, end, initial_cash, costs) | {"strategy": "SPY_buy_and_hold"})
    else:
        results.append({"strategy": "SPY_buy_and_hold", "status": "missing SPY data; run stocks fetch with SPY"})
    results.extend(
        _result_summary(simulator.run(
            data, strategy, actions, from_timestamp=start, through_timestamp=end,
        ))
        for strategy in STRATEGIES.values()
    )
    return {
        "provider": "Yahoo Finance via yfinance", "timeframe": "daily",
        "common_period": {
            "start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        },
        "symbols": sorted(data), "costs_bps_per_side": {"fee": costs.fee_bps, "slippage": costs.slippage_bps},
        "risk_limits": {"max_position_fraction": max_position_fraction, "max_positions": max_positions},
        "results": results,
        "limitations": [
            "The fixed universe is selected today and may contain survivorship bias in historical results.",
            "Historical OHLC is split-adjusted by the provider; ex-date cash dividends are credited to held positions.",
            "Signals use prior completed daily bars and simulate fills at the following session close with adverse slippage.",
            "This is a research backtest, not evidence of future performance.",
        ],
    }


def _buy_hold(
    data: dict[str, list[Candle]], actions: dict[str, dict[int, dict[str, float]]],
    symbols: list[str], start: int, end: int, initial_cash: float, costs: Costs,
) -> dict[str, Any]:
    eligible: list[tuple[str, list[Candle]]] = []
    for symbol in symbols:
        bars = [bar for bar in data.get(symbol, []) if start <= bar.timestamp <= end]
        if bars:
            eligible.append((symbol, bars))
    if not eligible:
        raise ValueError("No benchmark candles overlap the stock paper period")
    allocation = initial_cash / len(eligible)
    cash = initial_cash
    holdings: dict[str, dict[str, Any]] = {}
    timeline: dict[int, list[tuple[str, Candle]]] = defaultdict(list)
    for symbol, bars in eligible:
        for bar in bars:
            timeline[bar.timestamp].append((symbol, bar))
    marks: dict[str, float] = {}
    equity_curve = [initial_cash]
    for timestamp in sorted(timeline):
        for symbol, bar in sorted(timeline[timestamp], key=lambda item: item[0]):
            holding = holdings.get(symbol)
            if holding is None:
                entry_fill = bar.close * (1 + costs.slippage_rate)
                quantity = allocation / (entry_fill * (1 + costs.fee_rate))
                entry_cost = quantity * entry_fill * (1 + costs.fee_rate)
                if entry_cost > cash:
                    quantity = cash / (entry_fill * (1 + costs.fee_rate))
                    entry_cost = quantity * entry_fill * (1 + costs.fee_rate)
                cash -= entry_cost
                holding = holdings[symbol] = {
                    "quantity": quantity, "entry_timestamp": timestamp,
                }
            else:
                dividend = float(actions.get(symbol, {}).get(timestamp, {}).get("dividend", 0.0))
                if dividend > 0:
                    cash += float(holding["quantity"]) * dividend
            marks[symbol] = bar.close
        equity_curve.append(cash + sum(
            float(holding["quantity"]) * marks.get(symbol, 0.0)
            for symbol, holding in holdings.items()
        ))
    ending_equity = cash + sum(
        float(holding["quantity"]) * marks.get(symbol, 0.0)
        * (1 - costs.slippage_rate) * (1 - costs.fee_rate)
        for symbol, holding in holdings.items()
    )
    equity_curve.append(ending_equity)
    peak = initial_cash
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return {
        "strategy": "equal_weight_buy_and_hold", "symbols": [row[0] for row in eligible],
        "ending_equity": ending_equity, "pnl": ending_equity - initial_cash,
        "return_pct": (ending_equity / initial_cash - 1) * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "closed_trades": len(holdings), "open_positions": 0,
    }


def compare_equity_paper_account(store: MarketStore) -> dict[str, Any]:
    state = store.get_paper_state(PAPER_ACCOUNT_ID)
    if not state:
        raise ValueError("No stock paper account exists")
    curve = state.get("equity_curve", [])
    if not curve:
        raise ValueError("The stock account has no forward equity points yet")
    # The initial curve point is the first after-close observation (which only
    # creates tomorrow's signal). Compare strategies from the next session.
    start = int(curve[0]["timestamp"]) + 1
    end = int(state["cursor"])
    symbols = list(state["pairs"])
    if start >= end:
        return {
            "account": PAPER_ACCOUNT_ID, "status": "insufficient_forward_history",
            "message": "At least one session after the first signal is needed for a fair forward comparison.",
            "pending_targets": state.get("pending_targets", {}),
            "equity_points": len(curve),
        }
    data = store.get_candles(DAILY_INTERVAL_MINUTES, symbols)
    actions = store.get_corporate_actions(list(data), through_timestamp=end)
    costs = Costs(float(state["fee_bps"]), float(state["slippage_bps"]))
    simulator = EquitySimulator(
        float(state["starting_cash"]), costs,
        float(state["max_position_fraction"]), int(state["max_positions"]),
    )
    baselines = [
        _result_summary(simulator.run(data, strategy, actions, from_timestamp=start, through_timestamp=end))
        for strategy in STRATEGIES.values()
    ]
    summary = paper_summary(state)
    actual = {
        "strategy": PAPER_ACCOUNT_ID, "ending_equity": summary["equity"], "pnl": summary["pnl"],
        "return_pct": summary["return_pct"], "closed_trades": summary["closed_trades"],
        "open_positions": len(summary["open_positions"]), "source": "forward paper ledger",
    }
    cash = {"strategy": "cash", "ending_equity": float(state["starting_cash"]), "pnl": 0.0, "return_pct": 0.0}
    benchmark = _buy_hold(data, actions, ["SPY"], start, end, float(state["starting_cash"]), costs)
    equal_weight = _buy_hold(data, actions, symbols, start, end, float(state["starting_cash"]), costs)
    return {
        "account": PAPER_ACCOUNT_ID,
        "period": {"start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                   "end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat()},
        "timeframe": "daily", "symbols": sorted(data),
        "costs_bps_per_side": {"fee": costs.fee_bps, "slippage": costs.slippage_bps},
        "results": [actual, cash, benchmark, equal_weight, *baselines],
        "notes": [
            "Post-close signals are staged for the next session; the first after-close observation is excluded from performance comparisons because it creates targets but cannot fill them.",
            "The paper account and fixed baselines use prior completed-day information and simulated next-session-close fills with the same costs and risk limits.",
            "The S&P 500 ETF benchmark uses SPY; equal-weight buy-and-hold covers the selected symbols.",
            "The selected historical universe is subject to survivorship bias; results are descriptive only.",
        ],
    }
