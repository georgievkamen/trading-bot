from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import re
import time as time_module
import warnings
from typing import Any
from zoneinfo import ZoneInfo

from .local_model import LocalOllamaAdvisor
from .models import Candle, EquityPoint, Position, SimulationResult, Trade
from .news import EquityNewsFeed
from .paper import NoNewCandles, paper_summary
from .simulator import Costs
from .storage import MarketStore
from .strategies import (
    STRATEGIES, MeanReversion, RelativeMomentum, Strategy, TrendFollowing,
    VolatilityAdjustedBreakout, prepare_universe, should_remain_long,
)


US_EASTERN = ZoneInfo("America/New_York")
DAILY_INTERVAL_MINUTES = 1440
INTRADAY_INTERVAL_MINUTES = 15
PAPER_ACCOUNT_ID = "stocks_llama_regime"
STOCK_PAPER_STRATEGIES = (*STRATEGIES.keys(), "llama_regime")


def stock_paper_account_id(strategy_name: str) -> str:
    if strategy_name not in STOCK_PAPER_STRATEGIES:
        raise ValueError(f"Unknown stock paper strategy: {strategy_name}")
    return PAPER_ACCOUNT_ID if strategy_name == "llama_regime" else f"stocks_{strategy_name}"


STOCK_PAPER_ACCOUNT_IDS = tuple(stock_paper_account_id(name) for name in STOCK_PAPER_STRATEGIES)
DEFAULT_STOCK_SYMBOLS = (
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN",
    "INTC", "TSLA", "SPCX",
)
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
MARKET_CLOSE_CHECK = time(16, 15)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
INTRADAY_CLOSE_GRACE = time(16, 15)
_EQUITY_NEWS = EquityNewsFeed()


def _symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"Invalid US equity ticker: {value!r}")
    return symbol


def _session_date_to_timestamp(value: Any) -> int:
    session_date = value.date() if hasattr(value, "date") else value
    session = datetime.combine(session_date, time.min, tzinfo=US_EASTERN)
    return int(session.timestamp())


def _bar_timestamp_to_epoch(value: Any) -> int:
    stamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=US_EASTERN)
    return int(stamp.astimezone(timezone.utc).timestamp())


def _action_timestamp_for_bar(timestamp: int, interval_minutes: int) -> int:
    if interval_minutes >= DAILY_INTERVAL_MINUTES:
        return timestamp
    session_date = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(US_EASTERN).date()
    return _session_date_to_timestamp(session_date)


def _api_interval(interval_minutes: int) -> str:
    if interval_minutes == DAILY_INTERVAL_MINUTES:
        return "1d"
    if interval_minutes == INTRADAY_INTERVAL_MINUTES:
        return "15m"
    raise ValueError("US equity paper data supports daily bars or 15-minute bars")


def fetch_equity_history(
    symbols: list[str] | tuple[str, ...] = DEFAULT_STOCK_SYMBOLS,
    *, period: str = "5y", interval_minutes: int = DAILY_INTERVAL_MINUTES,
    timeout: int = 30,
) -> tuple[dict[str, list[Candle]], list[dict[str, object]]]:
    """Fetch regular-session Yahoo Finance candles and daily corporate actions."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install the stock data extra with: python -m pip install -e ".[stocks]"') from exc

    normalized = list(dict.fromkeys(_symbol(symbol) for symbol in symbols))
    if not normalized:
        raise ValueError("At least one ticker is required")
    interval = _api_interval(interval_minutes)
    if interval_minutes == INTRADAY_INTERVAL_MINUTES and period not in {"1d", "5d", "10d", "1mo", "3mo"}:
        raise ValueError("Yahoo Finance intraday history is limited to recent periods; use 10d or less")
    now_et = datetime.now(US_EASTERN)
    now_utc = datetime.now(timezone.utc).timestamp()
    may_include_today = interval_minutes == DAILY_INTERVAL_MINUTES and now_et.weekday() < 5 and now_et.time() >= MARKET_CLOSE_CHECK
    with warnings.catch_warnings():
        try:
            from pandas.errors import Pandas4Warning
            warnings.simplefilter("ignore", Pandas4Warning)
        except ImportError:
            pass
        frame = yf.download(
            tickers=normalized,
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=interval_minutes == DAILY_INTERVAL_MINUTES,
            prepost=False,
            repair=True,
            progress=False,
            threads=True,
            group_by="ticker",
            multi_level_index=True,
            timeout=timeout,
        )
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no {interval} bars")

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
            if interval_minutes == DAILY_INTERVAL_MINUTES:
                session_date = index.date() if hasattr(index, "date") else index
                if isinstance(session_date, datetime):
                    session_date = session_date.date()
                if session_date > now_et.date() or (session_date == now_et.date() and not may_include_today):
                    continue
                timestamp = _session_date_to_timestamp(session_date)
            else:
                stamp = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=US_EASTERN)
                session_stamp = stamp.astimezone(US_EASTERN)
                if session_stamp.weekday() >= 5 or not (MARKET_OPEN <= session_stamp.time() < MARKET_CLOSE):
                    continue
                timestamp = _bar_timestamp_to_epoch(index)
                if timestamp + interval_minutes * 60 > now_utc:
                    continue
                end_et = session_stamp + timedelta(minutes=interval_minutes)
                if end_et.time() > MARKET_CLOSE:
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
            bars.append(Candle(symbol, interval_minutes, timestamp, open_price, high, low, close, volume))

            if interval_minutes != DAILY_INTERVAL_MINUTES:
                continue
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
            errors.append({"symbol": symbol, "error": f"no completed {interval} bars in provider response"})
    if not candles:
        raise RuntimeError(f"No completed {interval} equity bars were available")
    return candles, errors + actions  # actions are tagged with pair/timestamp plus action fields


def refresh_equity_data(
    store: MarketStore, symbols: list[str] | tuple[str, ...], *, period: str,
    interval_minutes: int = DAILY_INTERVAL_MINUTES,
) -> dict[str, object]:
    candles, mixed_rows = fetch_equity_history(symbols, period=period, interval_minutes=interval_minutes)
    errors = [row for row in mixed_rows if "error" in row]
    actions = [row for row in mixed_rows if "error" not in row]
    saved_by_symbol = {
        symbol: store.save_candles(rows)
        for symbol, rows in candles.items()
    }
    store.save_corporate_actions(actions)
    return {
        "provider": "Yahoo Finance via yfinance",
        "interval_minutes": interval_minutes,
        "fetched_symbols": len(candles),
        "symbols": sorted(candles),
        "saved_rows_by_symbol": saved_by_symbol,
        "errors": errors,
        "bars_are_dividend_unadjusted": True,
        "today_session_included_only_after_1615_et": interval_minutes == DAILY_INTERVAL_MINUTES,
        "incomplete_intraday_bars_dropped": interval_minutes == INTRADAY_INTERVAL_MINUTES,
    }


class EquityLlamaRegimeSwitch:
    """Regime classifier; Llama selects an approved deterministic rule or cash."""

    name = PAPER_ACCOUNT_ID
    minimum_confidence = 0.60
    batch_size = 3

    def __init__(self, model: str = LocalOllamaAdvisor.DEFAULT_MODEL) -> None:
        self.model = model
        self.advisor = LocalOllamaAdvisor(model)
        self.strategies: dict[str, Strategy] = {
            "trend": TrendFollowing(),
            "range": MeanReversion(),
            "breakout": VolatilityAdjustedBreakout(),
            "relative_momentum": RelativeMomentum(),
        }
        self._labels: dict[tuple[str, int], str] = {}
        self._notes: dict[str, dict[str, object]] = {}

    @property
    def notes(self) -> dict[str, dict[str, object]]:
        return {pair: dict(note) for pair, note in self._notes.items()}

    def prepare_universe(self, histories: dict[str, list[Candle]], timestamp: int) -> None:
        for strategy in self.strategies.values():
            prepare_universe(strategy, histories, timestamp)

    def eligible_for_classification(self, history: list[Candle], currently_long: bool) -> bool:
        return currently_long or any(
            strategy.desired_long(history, currently_long=False)
            for strategy in self.strategies.values()
        )

    def prepare_batch(self, histories: dict[str, list[Candle]]) -> None:
        eligible = {pair: bars for pair, bars in histories.items() if len(bars) >= 30}
        pairs = list(eligible)
        news_by_pair = _EQUITY_NEWS.fetch_many(pairs)
        for offset in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[offset:offset + self.batch_size]
            batch = {pair: eligible[pair] for pair in batch_pairs}
            try:
                notes = self.advisor.classify_batch(
                    {pair: [bar.close for bar in bars[-61:]] for pair, bars in batch.items()},
                    news_by_pair={pair: news_by_pair.get(pair, ()) for pair in batch_pairs},
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
                    "timeframe": "15m", "news": [item.as_dict() for item in note.news],
                    "news_status": _EQUITY_NEWS.status_for(pair),
                }

    def _label(self, history: list[Candle]) -> str:
        if len(history) < 30:
            return "unknown"
        last = history[-1]
        return self._labels.get((last.pair, last.timestamp), "unknown")

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        label = self._label(history)
        strategy = self.strategies.get(label)
        if strategy is not None:
            return strategy.desired_long(history, currently_long)
        if label in {"cash", "high_volatility", "unavailable"}:
            return False
        return currently_long

    def desired_long_for_position(self, history: list[Candle], position: object) -> bool:
        label = self._label(history)
        strategy = self.strategies.get(label)
        if strategy is not None:
            return should_remain_long(strategy, history, position)
        return label not in {"cash", "high_volatility", "unavailable"}

    def entry_strength(self, history: list[Candle]) -> float:
        label = self._label(history)
        strategy = self.strategies.get(label)
        if strategy is not None:
            return strategy.entry_strength(history)
        return 0.0


class EquitySimulator:
    """Long-only equity simulator using next-open fills and cash dividends."""

    def __init__(
        self, initial_cash: float = 1000.0, costs: Costs = Costs(0.0, 5.0),
        max_position_fraction: float = 0.10, max_positions: int = 6,
    ) -> None:
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        if not 0 < max_position_fraction <= 0.10 or not 1 <= max_positions <= 6:
            raise ValueError("Position limits cannot exceed 10% of equity or six concurrent positions")
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
        reset = getattr(strategy, "reset", None)
        if callable(reset):
            reset()
        timestamps_by_symbol = {
            symbol: [bar.timestamp for bar in bars] for symbol, bars in normalized.items()
        }
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
        applied_actions: set[tuple[str, int]] = set()
        fee, slip = self.costs.fee_rate, self.costs.slippage_rate
        for timestamp in sorted(events):
            rows = sorted(events[timestamp], key=lambda item: item[0])
            completed_histories = {
                symbol: bars[:bisect_left(timestamps_by_symbol[symbol], timestamp)]
                for symbol, bars in normalized.items()
            }
            prepare_universe(strategy, completed_histories, timestamp)
            for symbol, index in rows:
                candle = normalized[symbol][index]
                action_timestamp = _action_timestamp_for_bar(timestamp, candle.interval)
                action = ((actions or {}).get(symbol, {})).get(action_timestamp, {})
                position = positions.get(symbol)
                action_key = (symbol, action_timestamp)
                if position and action_key not in applied_actions:
                    # Yahoo's historical OHLC has split-adjusted prices; do not
                    # apply historical split ratios a second time in backtests.
                    cash += position.quantity * float(action.get("dividend", 0.0))
                applied_actions.add(action_key)
                marks[symbol] = candle.open

            for symbol, index in rows:
                position = positions.get(symbol)
                if position and not should_remain_long(strategy, normalized[symbol][:index], position):
                    base = normalized[symbol][index].open
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
                base = normalized[symbol][index].open
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

            for symbol, index in rows:
                marks[symbol] = normalized[symbol][index].close
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
    reset: bool = False, interval_minutes: int = DAILY_INTERVAL_MINUTES,
    strategy_name: str = "llama_regime",
) -> dict[str, Any]:
    account_id = stock_paper_account_id(strategy_name)
    existing = store.get_paper_state(account_id)
    if existing and existing.get("active") and not reset:
        raise ValueError("That stock paper account already exists; use status/watch to resume it, or pass --reset to replace it")
    _api_interval(interval_minutes)
    data = store.get_candles(interval_minutes, symbols)
    if not data:
        raise ValueError(f"No stock candles found. Fetch {_api_interval(interval_minutes)} bars first.")
    EquitySimulator(starting_cash, costs, max_position_fraction, max_positions)
    pairs = sorted(data)
    cursor = max(bars[-1].timestamp for bars in data.values() if bars)
    marks = {pair: bars[-1].close for pair, bars in data.items() if bars}
    state: dict[str, Any] = {
        "active": True, "market": "US equities", "data_provider": "Yahoo Finance via yfinance",
        "interval": interval_minutes, "pairs": pairs,
        "strategy": account_id, "engine_strategy": strategy_name,
        "model": model if strategy_name == "llama_regime" else None,
        "starting_cash": starting_cash, "cash": starting_cash,
        "fee_bps": costs.fee_bps, "slippage_bps": costs.slippage_bps,
        "max_position_fraction": max_position_fraction, "max_positions": max_positions,
        "cursor": cursor, "start_timestamp": cursor,
        "positions": {}, "open_trades": {}, "closed_trades": [],
        "equity_curve": [], "last_marks": marks, "dividend_income": 0.0,
        "last_processing_mode": "waiting_for_next_completed_15m_bar" if interval_minutes == INTRADAY_INTERVAL_MINUTES else "waiting_for_forward_session",
        "last_check_timestamp": int(time_module.time()),
        "watcher_heartbeat_timestamp": int(time_module.time()),
        "market_status": {"requested_markets": len(pairs), "successful_markets": len(marks),
                           "failed_markets": 0,
                           "state": "waiting_for_next_completed_15m_bar" if interval_minutes == INTRADAY_INTERVAL_MINUTES else "waiting_for_next_us_close"},
    }
    store.save_paper_state(state, account_id)
    return state


def activate_intraday_equity_paper(store: MarketStore) -> dict[str, Any]:
    """Move the existing stock paper ledger to 15-minute bars without resetting it."""
    state = store.get_paper_state(PAPER_ACCOUNT_ID)
    if not state or not state.get("active"):
        raise ValueError("No active stock paper account. Run `stocks paper start` first.")
    if int(state.get("interval", DAILY_INTERVAL_MINUTES)) == INTRADAY_INTERVAL_MINUTES:
        return paper_summary(state)
    symbols = list(state["pairs"])
    fetched = refresh_equity_data(
        store, symbols, period="10d", interval_minutes=INTRADAY_INTERVAL_MINUTES,
    )
    data = store.get_candles(INTRADAY_INTERVAL_MINUTES, symbols)
    missing = [symbol for symbol in symbols if not data.get(symbol)]
    if missing:
        raise ValueError("No completed 15-minute bars for: " + ", ".join(missing))
    cursor = max(bars[-1].timestamp for bars in data.values())
    marks = dict(state.get("last_marks", {}))
    marks.update({symbol: bars[-1].close for symbol, bars in data.items() if bars})
    state["interval"] = INTRADAY_INTERVAL_MINUTES
    state["cursor"] = cursor
    state["pairs"] = sorted(data)
    state["last_marks"] = marks
    state["pending_targets"] = {}
    state["model_notes"] = {}
    state["last_processing_mode"] = "switched_to_15m_forward_paper"
    state["last_check_timestamp"] = int(time_module.time())
    state["watcher_heartbeat_timestamp"] = int(time_module.time())
    state["intraday_activated_at"] = int(time_module.time())
    state["market_status"] = {
        "requested_markets": len(symbols), "successful_markets": len(data),
        "failed_markets": len(fetched["errors"]), "state": "waiting_for_next_completed_15m_bar",
        "errors": fetched["errors"],
    }
    store.save_paper_state(state, PAPER_ACCOUNT_ID)
    return {**paper_summary(state), "preserved_existing_ledger": True, "interval_minutes": INTRADAY_INTERVAL_MINUTES}


def extend_equity_paper_universe(
    store: MarketStore, symbols: list[str], strategy_name: str = "llama_regime",
) -> dict[str, Any]:
    """Add fetched symbols to the active forward account without resetting it."""
    account_id = stock_paper_account_id(strategy_name)
    state = store.get_paper_state(account_id)
    if not state or not state.get("active"):
        raise ValueError("No active stock paper account. Run stocks paper start first.")
    requested = list(dict.fromkeys(_symbol(symbol) for symbol in symbols))
    if not requested:
        raise ValueError("Provide at least one ticker to add")
    existing = set(state["pairs"])
    additions = [symbol for symbol in requested if symbol not in existing]
    if not additions:
        return {**paper_summary(state), "added_symbols": []}

    interval = int(state.get("interval", DAILY_INTERVAL_MINUTES))
    data = store.get_candles(interval, additions)
    missing = [symbol for symbol in additions if not data.get(symbol)]
    if missing:
        raise ValueError(f"Fetch {_api_interval(interval)} data before adding these symbols: " + ", ".join(missing))
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
    store.save_paper_state(state, account_id)
    return {**paper_summary(state), "added_symbols": additions}


def advance_equity_paper_account(
    store: MarketStore, strategy_name: str = "llama_regime",
) -> dict[str, Any]:
    account_id = stock_paper_account_id(strategy_name)
    state = store.get_paper_state(account_id)
    if not state or not state.get("active"):
        raise ValueError("No active stock paper account. Run `stocks paper start` first.")
    EquitySimulator(
        float(state.get("starting_cash", 1000.0)),
        Costs(float(state.get("fee_bps", 0.0)), float(state.get("slippage_bps", 5.0))),
        float(state.get("max_position_fraction", 0.10)),
        int(state.get("max_positions", 6)),
    )
    engine_strategy = str(state.get("engine_strategy") or strategy_name)
    pairs = list(state["pairs"])
    interval = int(state.get("interval", DAILY_INTERVAL_MINUTES))
    data = store.get_candles(interval, pairs)
    cursor = int(state["cursor"])
    new_timestamps = sorted({
        bar.timestamp for bars in data.values() for bar in bars if bar.timestamp > cursor
    })
    if not new_timestamps:
        raise NoNewCandles("No new completed US equity candle is available")
    timestamp = max(new_timestamps)
    rows: dict[str, tuple[int, Candle]] = {}
    for pair, bars in data.items():
        for index in range(len(bars) - 1, -1, -1):
            if bars[index].timestamp == timestamp:
                rows[pair] = (index, bars[index])
                break
    if not rows:
        raise NoNewCandles("No completed US equity candle is available")

    actions = store.get_corporate_actions(pairs, through_timestamp=timestamp)
    action_timestamp = _action_timestamp_for_bar(timestamp, interval)
    action_today = {pair: action_rows.get(action_timestamp, {}) for pair, action_rows in actions.items()}
    applied_action_dates = state.setdefault("applied_action_dates", {})
    positions: dict[str, dict[str, Any]] = state["positions"]
    open_trades: dict[str, dict[str, Any]] = state["open_trades"]
    marks: dict[str, float] = state["last_marks"]
    cash = float(state["cash"])
    fee_rate = float(state["fee_bps"]) / 10_000
    slip_rate = float(state["slippage_bps"]) / 10_000

    # Apply splits and dividends before next-open exits or entries.
    dividend_today = 0.0
    for pair, (index, candle) in rows.items():
        if interval != DAILY_INTERVAL_MINUTES and applied_action_dates.get(pair) == action_timestamp:
            action_today[pair] = {}
        elif interval != DAILY_INTERVAL_MINUTES:
            applied_action_dates[pair] = action_timestamp
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
        marks[pair] = candle.open

    missed = sorted(value for value in new_timestamps if value < timestamp)
    pending_targets = state.get("pending_targets", {})
    allow_pending_fills = bool(state.pop("allow_pending_fills", True))
    stale_pending = bool(pending_targets) and (bool(missed) or not allow_pending_fills)

    # The previous completed bar created these targets. Simulated fills are
    # recorded at this bar's open with adverse slippage. If bars
    # were missed, discard stale targets instead of inventing a fill.
    if not stale_pending:
        for pair, (index, candle) in sorted(rows.items()):
            target = pending_targets.get(pair, {})
            wants_long = bool(target.get("long", False)) if isinstance(target, dict) else bool(target)
            position = positions.get(pair)
            if position and not wants_long:
                fill = candle.open * (1 - slip_rate)
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
                candidates.append((strength, pair, candle.open))
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

    # Only completed bars inform the regime decision. Targets become eligible
    # after this bar and cannot fill against the price that created the signal.
    histories = {pair: bars[:index + 1] for pair, (index, _) in rows.items()}
    universe_histories = {
        pair: [bar for bar in bars if bar.timestamp <= timestamp]
        for pair, bars in data.items()
    }
    strategy: Strategy = (
        EquityLlamaRegimeSwitch(str(state.get("model") or LocalOllamaAdvisor.DEFAULT_MODEL))
        if engine_strategy == "llama_regime" else STRATEGIES[engine_strategy]
    )
    prepare_universe(strategy, universe_histories, timestamp + interval * 60)
    if isinstance(strategy, EquityLlamaRegimeSwitch):
        strategy.prepare_batch({
            pair: history for pair, history in histories.items()
            if len(history) >= 30 and strategy.eligible_for_classification(history, pair in positions)
        })
    next_targets: dict[str, dict[str, Any]] = {}
    for pair, history in histories.items():
        wants_long = (
            should_remain_long(strategy, history, positions[pair])
            if pair in positions else strategy.desired_long(history, currently_long=False)
        )
        next_targets[pair] = {
            "long": wants_long,
            "strength": strategy.entry_strength(history) if wants_long and pair not in positions else 0.0,
            "signal_timestamp": timestamp,
        }
    if isinstance(strategy, EquityLlamaRegimeSwitch):
        for note in strategy.notes.values():
            note["target_applies_after_timestamp"] = timestamp
            note["timeframe"] = "15m" if interval == INTRADAY_INTERVAL_MINUTES else "daily"
    for pair, (_, candle) in rows.items():
        marks[pair] = candle.close
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
    state["watcher_heartbeat_timestamp"] = int(time_module.time())
    if interval == INTRADAY_INTERVAL_MINUTES:
        state["last_processing_mode"] = "live_forward_15m_skipped_stale_targets" if stale_pending else "live_forward_15m"
    else:
        state["last_processing_mode"] = "live_forward_daily_close_skipped_stale_targets" if stale_pending else "live_forward_daily_close"
    state["market_status"] = {
        "requested_markets": len(pairs), "successful_markets": len(rows),
        "failed_markets": max(0, len(pairs) - len(rows)),
        "state": "processed_completed_15m_bar" if interval == INTRADAY_INTERVAL_MINUTES else "processed_completed_session",
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
    state["equity_curve"].append({"timestamp": timestamp, "equity": equity})
    state["model_notes"] = strategy.notes if isinstance(strategy, EquityLlamaRegimeSwitch) else {}
    store.save_paper_state(state, account_id)
    return paper_summary(state)


def watch_equity_paper_account(
    store: MarketStore, poll_seconds: int = 60, once: bool = False,
    strategy_name: str = "llama_regime",
) -> None:
    if poll_seconds < 30:
        raise ValueError("Poll interval must be at least 30 seconds")
    account_id = stock_paper_account_id(strategy_name)
    state = store.get_paper_state(account_id)
    if not state or not state.get("active"):
        raise ValueError("No active stock paper account. Run `stocks paper start` first.")
    symbols = list(state["pairs"])
    last_attempt_date: date | None = None
    while True:
        now_et = datetime.now(US_EASTERN)
        now_timestamp = int(time_module.time())
        interval = int(state.get("interval", DAILY_INTERVAL_MINUTES))
        intraday = interval == INTRADAY_INTERVAL_MINUTES
        if intraday and now_et.time() >= time(9, 45):
            quarter = (now_et.minute // INTRADAY_INTERVAL_MINUTES) * INTRADAY_INTERVAL_MINUTES
            expected_bar = now_et.replace(minute=quarter, second=0, microsecond=0) - timedelta(minutes=INTRADAY_INTERVAL_MINUTES)
            expected_bar_timestamp = int(expected_bar.astimezone(timezone.utc).timestamp())
        else:
            expected_bar_timestamp = 0
        intraday_fetch_window = (
            now_et.weekday() < 5 and time(9, 45) <= now_et.time() < INTRADAY_CLOSE_GRACE
            and expected_bar_timestamp > int(state.get("cursor", 0))
        )

        if intraday and (intraday_fetch_window or once):
            try:
                heartbeat = int(state.get("watcher_heartbeat_timestamp", 0))
                continuity_ok = heartbeat > 0 and now_timestamp - heartbeat <= max(180, poll_seconds * 3)
                fetch_result = refresh_equity_data(
                    store, symbols, period="10d", interval_minutes=INTRADAY_INTERVAL_MINUTES,
                )
                state = store.get_paper_state(account_id) or state
                state["allow_pending_fills"] = continuity_ok
                state["last_check_timestamp"] = now_timestamp
                state["watcher_heartbeat_timestamp"] = now_timestamp
                state.pop("feed_errors", None)
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": fetch_result["fetched_symbols"],
                    "failed_markets": len(fetch_result["errors"]),
                    "state": "fetching_15m_bars" if fetch_result["fetched_symbols"] else "intraday_feed_empty",
                    "errors": fetch_result["errors"],
                }
                store.save_paper_state(state, account_id)
                try:
                    summary = advance_equity_paper_account(store, strategy_name)
                    print(json.dumps(summary, allow_nan=False), flush=True)
                except NoNewCandles:
                    current = store.get_paper_state(account_id) or state
                    current["market_status"]["state"] = "waiting_for_next_completed_15m_bar"
                    store.save_paper_state(current, account_id)
            except Exception as exc:
                state = store.get_paper_state(account_id) or state
                state["last_check_timestamp"] = now_timestamp
                state["watcher_heartbeat_timestamp"] = now_timestamp
                state["feed_errors"] = [{"error": str(exc)}]
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": 0,
                    "failed_markets": len(symbols), "state": "provider_error",
                }
                store.save_paper_state(state, account_id)
                print(json.dumps({"equity_data_error": str(exc)}), flush=True)
        elif not intraday and now_et.weekday() < 5 and now_et.time() >= MARKET_CLOSE_CHECK and last_attempt_date != now_et.date():
            try:
                heartbeat = int(state.get("watcher_heartbeat_timestamp", 0))
                continuity_ok = heartbeat > 0 and now_timestamp - heartbeat <= max(180, poll_seconds * 3)
                fetch_result = refresh_equity_data(store, symbols, period="10d", interval_minutes=DAILY_INTERVAL_MINUTES)
                last_attempt_date = now_et.date()
                state = store.get_paper_state(account_id) or state
                state["allow_pending_fills"] = continuity_ok
                state["last_check_timestamp"] = now_timestamp
                state["watcher_heartbeat_timestamp"] = now_timestamp
                state.pop("feed_errors", None)
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": fetch_result["fetched_symbols"],
                    "failed_markets": len(fetch_result["errors"]), "state": "daily_fetch_complete",
                    "errors": fetch_result["errors"],
                }
                store.save_paper_state(state, account_id)
                try:
                    summary = advance_equity_paper_account(store, strategy_name)
                    print(json.dumps(summary, allow_nan=False), flush=True)
                except NoNewCandles:
                    current = store.get_paper_state(account_id) or state
                    current["market_status"]["state"] = "no_new_us_session"
                    store.save_paper_state(current, account_id)
            except Exception as exc:
                state = store.get_paper_state(account_id) or state
                state["last_check_timestamp"] = now_timestamp
                state["watcher_heartbeat_timestamp"] = now_timestamp
                state["feed_errors"] = [{"error": str(exc)}]
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": 0,
                    "failed_markets": len(symbols), "state": "provider_error",
                }
                store.save_paper_state(state, account_id)
                print(json.dumps({"equity_data_error": str(exc)}), flush=True)
        else:
            state = store.get_paper_state(account_id) or state
            state["last_check_timestamp"] = now_timestamp
            state["watcher_heartbeat_timestamp"] = now_timestamp
            if intraday:
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": len(state.get("last_marks", {})),
                    "failed_markets": 0,
                    "state": "waiting_for_next_completed_15m_bar" if now_et.weekday() < 5 and time(9, 30) <= now_et.time() < INTRADAY_CLOSE_GRACE else "waiting_for_us_regular_session",
                }
            elif now_et.weekday() >= 5 or now_et.time() < MARKET_CLOSE_CHECK:
                state["market_status"] = {
                    "requested_markets": len(symbols), "successful_markets": len(state.get("last_marks", {})),
                    "failed_markets": 0, "state": "waiting_for_us_market_close",
                }
            store.save_paper_state(state, account_id)
        if once:
            return
        time_module.sleep(poll_seconds)


def _result_summary(result: SimulationResult) -> dict[str, object]:
    return {
        "strategy": result.strategy, "ending_equity": result.ending_equity,
        "pnl": result.ending_equity - result.initial_cash, "return_pct": result.return_pct,
        "max_drawdown_pct": result.max_drawdown_pct, "closed_trades": result.closed_trades,
        "win_rate_pct": result.win_rate_pct, "profit_factor": result.profit_factor,
        "open_positions": len(result.open_positions),
    }


def run_equity_backtests(
    store: MarketStore, symbols: list[str], *, initial_cash: float = 1000.0,
    costs: Costs = Costs(0.0, 5.0), max_position_fraction: float = 0.10,
    max_positions: int = 6, test_fraction: float = 0.30,
) -> dict[str, Any]:
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("Out-of-sample fraction must be between 0.10 and 0.50")
    data = store.get_candles(DAILY_INTERVAL_MINUTES, symbols)
    if not data:
        raise ValueError("No stored equity candles. Run `stocks fetch` first.")
    actions = store.get_corporate_actions(list(data))
    simulator = EquitySimulator(initial_cash, costs, max_position_fraction, max_positions)
    # Instruments in the fixed universe have different listing histories.
    # Start at the earliest available session so a recently listed symbol such
    # as SPCX does not truncate every other instrument's five-year evaluation.
    start = min(bars[0].timestamp for bars in data.values() if bars)
    end = min(bars[-1].timestamp for bars in data.values() if bars)
    if start >= end:
        raise ValueError("At least two overlapping market sessions are needed for a comparison")
    common_timestamps = sorted({
        bar.timestamp for bars in data.values() for bar in bars if start <= bar.timestamp <= end
    })
    split_index = int(len(common_timestamps) * (1 - test_fraction))
    if split_index < 1 or split_index >= len(common_timestamps):
        raise ValueError("Not enough overlapping sessions for the requested chronological split")
    training_end = common_timestamps[split_index - 1]
    out_of_sample_start = common_timestamps[split_index]
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
    out_of_sample_results: list[dict[str, object]] = [
        {"strategy": "cash", "ending_equity": initial_cash, "pnl": 0.0,
         "return_pct": 0.0, "max_drawdown_pct": 0.0, "closed_trades": 0,
         "win_rate_pct": None, "profit_factor": None, "open_positions": 0},
        _buy_hold(data, actions, sorted(data), out_of_sample_start, end, initial_cash, costs),
    ]
    if "SPY" in data:
        out_of_sample_results.append(
            _buy_hold(data, actions, ["SPY"], out_of_sample_start, end, initial_cash, costs)
            | {"strategy": "SPY_buy_and_hold"}
        )
    else:
        out_of_sample_results.append({"strategy": "SPY_buy_and_hold", "status": "missing SPY data"})
    out_of_sample_results.extend(
        _result_summary(simulator.run(
            data, strategy, actions,
            from_timestamp=out_of_sample_start, through_timestamp=end,
        ))
        for strategy in STRATEGIES.values()
    )
    return {
        "provider": "Yahoo Finance via yfinance", "timeframe": "daily",
        "common_period": {
            "start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        },
        "chronological_split": {
            "training_end_utc": datetime.fromtimestamp(training_end, timezone.utc).isoformat(),
            "out_of_sample_start_utc": datetime.fromtimestamp(out_of_sample_start, timezone.utc).isoformat(),
            "out_of_sample_fraction": test_fraction,
            "out_of_sample_sessions": len(common_timestamps) - split_index,
        },
        "symbols": sorted(data), "costs_bps_per_side": {"fee": costs.fee_bps, "slippage": costs.slippage_bps},
        "risk_limits": {"max_position_fraction": max_position_fraction, "max_positions": max_positions},
        "results": results,
        "out_of_sample_results": out_of_sample_results,
        "limitations": [
            "The fixed universe is selected today and may contain survivorship bias in historical results.",
            "Historical OHLC is split-adjusted by the provider; ex-date cash dividends are credited to held positions.",
            "Signals use prior completed daily bars and simulate fills at the following session open with adverse slippage.",
            "The out-of-sample split is chronological; strategy parameters are fixed in source and were not selected on that test segment.",
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
    applied_actions: set[tuple[str, int]] = set()
    for timestamp in sorted(timeline):
        for symbol, bar in sorted(timeline[timestamp], key=lambda item: item[0]):
            holding = holdings.get(symbol)
            action_timestamp = _action_timestamp_for_bar(timestamp, bar.interval)
            action_key = (symbol, action_timestamp)
            if holding is None:
                entry_fill = bar.open * (1 + costs.slippage_rate)
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
                dividend = float(actions.get(symbol, {}).get(action_timestamp, {}).get("dividend", 0.0))
                if dividend > 0 and action_key not in applied_actions:
                    cash += float(holding["quantity"]) * dividend
            applied_actions.add(action_key)
            marks[symbol] = bar.close
        equity_curve.append(cash + sum(
            float(holding["quantity"]) * marks.get(symbol, 0.0)
            for symbol, holding in holdings.items()
        ))
    ending_equity = cash + sum(
        float(holding["quantity"]) * marks.get(symbol, 0.0)
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
        "closed_trades": 0, "win_rate_pct": None, "profit_factor": None,
        "open_positions": len(holdings),
    }


def compare_equity_paper_account(
    store: MarketStore, strategy_name: str = "llama_regime",
) -> dict[str, Any]:
    account_id = stock_paper_account_id(strategy_name)
    state = store.get_paper_state(account_id)
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
            "account": account_id, "status": "insufficient_forward_history",
            "message": "At least one completed bar after the first signal is needed for a fair forward comparison.",
            "pending_targets": state.get("pending_targets", {}),
            "equity_points": len(curve),
        }
    interval = int(state.get("interval", DAILY_INTERVAL_MINUTES))
    data = store.get_candles(interval, symbols)
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
        "strategy": account_id, "ending_equity": summary["equity"], "pnl": summary["pnl"],
        "return_pct": summary["return_pct"], "closed_trades": summary["closed_trades"],
        "open_positions": len(summary["open_positions"]), "source": "forward paper ledger",
    }
    cash = {"strategy": "cash", "ending_equity": float(state["starting_cash"]), "pnl": 0.0, "return_pct": 0.0}
    benchmark = _buy_hold(data, actions, ["SPY"], start, end, float(state["starting_cash"]), costs)
    equal_weight = _buy_hold(data, actions, symbols, start, end, float(state["starting_cash"]), costs)
    return {
        "account": account_id,
        "period": {"start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                   "end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat()},
        "timeframe": "15m" if interval == INTRADAY_INTERVAL_MINUTES else "daily", "symbols": sorted(data),
        "costs_bps_per_side": {"fee": costs.fee_bps, "slippage": costs.slippage_bps},
        "results": [actual, cash, benchmark, equal_weight, *baselines],
        "notes": [
            "Completed-bar signals are staged for the next bar; the first forward observation is excluded because it creates targets but cannot fill them.",
            "The paper account and fixed baselines use prior completed-bar information and next-open fills with adverse slippage and the same costs and risk limits.",
            "The S&P 500 ETF benchmark uses SPY; equal-weight buy-and-hold covers the selected symbols.",
            "The selected historical universe is subject to survivorship bias; results are descriptive only.",
        ],
    }
