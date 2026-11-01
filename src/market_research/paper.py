from __future__ import annotations

from collections import defaultdict
import json
import time
from typing import Any

from .ai_strategy import LlamaRegimeSwitch
from .local_model import LocalOllamaAdvisor
from .models import Candle
from .kraken import fetch_ohlc, is_crypto_usd_pair
from .simulator import Costs, Simulator
from .strategies import STRATEGIES, Strategy
from .storage import MarketStore


class NoNewCandles(ValueError):
    pass


PAPER_STRATEGY_NAMES = (*STRATEGIES.keys(), "llama_regime")


def start_paper_account(
    store: MarketStore, interval: int, pairs: list[str], strategy_name: str,
    starting_cash: float = 1000.0, costs: Costs = Costs(),
    max_position_fraction: float = 0.10, max_positions: int = 6,
    replay_history: bool = False, model: str = LocalOllamaAdvisor.DEFAULT_MODEL,
) -> dict[str, Any]:
    if strategy_name not in PAPER_STRATEGY_NAMES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    bars = store.get_candles(interval, pairs)
    if pairs:
        invalid = [pair for pair in pairs if not is_crypto_usd_pair(pair)]
        if invalid:
            raise ValueError("V1 paper accounts only support USD-quoted crypto pairs; excluded: " + ", ".join(invalid))
    else:
        bars = {pair: rows for pair, rows in bars.items() if is_crypto_usd_pair(pair)}
    if not bars:
        raise ValueError("No stored candles. Fetch market data first.")
    cursor = (
        min(rows[0].timestamp for rows in bars.values() if rows)
        if replay_history else max(rows[-1].timestamp for rows in bars.values() if rows)
    )
    # Use the same validated portfolio limits as the backtester.
    Simulator(starting_cash, costs, max_position_fraction, max_positions)
    state: dict[str, Any] = {
        "active": True,
        "interval": interval,
        "pairs": [pair.upper() for pair in bars],
        "strategy": strategy_name,
        "model": model if strategy_name == "llama_regime" else None,
        "starting_cash": starting_cash,
        "cash": starting_cash,
        "fee_bps": costs.fee_bps,
        "slippage_bps": costs.slippage_bps,
        "max_position_fraction": max_position_fraction,
        "max_positions": max_positions,
        "cursor": cursor,
        "positions": {},
        "open_trades": {},
        "closed_trades": [],
        "equity_curve": [],
        "last_marks": {},
    }
    store.save_paper_state(state, strategy_name)
    return state


def advance_paper_account(
    store: MarketStore, strategy_name: str,
    live_ticks: dict[str, tuple[int, float]] | None = None,
) -> dict[str, Any]:
    state = store.get_paper_state(strategy_name)
    if not state or not state.get("active"):
        raise ValueError("No active paper account. Start one first.")
    data = store.get_candles(int(state["interval"]), list(state["pairs"]))
    events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for pair, bars in data.items():
        for index, bar in enumerate(bars):
            if bar.timestamp > int(state["cursor"]):
                events[bar.timestamp].append((pair, index))
    # A newly forming candle is the next execution event. Its timestamp lets
    # the signal use completed history while the last-price snapshot gives a
    # current forward fill instead of the candle's stale historical open.
    for pair, (live_timestamp, _) in (live_ticks or {}).items():
        bars = data.get(pair, [])
        if bars and live_timestamp > int(state["cursor"]) and bars[-1].timestamp < live_timestamp:
            events.setdefault(live_timestamp, []).append((pair, len(bars)))
    if not events:
        raise NoNewCandles("No newer market-data event is available yet.")

    timestamp = min(events)
    rows = sorted(events[timestamp], key=lambda item: item[0])
    live_for_event = {
        pair: float(live_ticks[pair][1])
        for pair, index in rows
        if live_ticks and pair in live_ticks
        and live_ticks[pair][0] == timestamp
        and index == len(data.get(pair, []))
    }
    strategy_name = str(state["strategy"])
    strategy: Strategy = (
        LlamaRegimeSwitch(str(state.get("model") or LocalOllamaAdvisor.DEFAULT_MODEL))
        if strategy_name == "llama_regime" else STRATEGIES[strategy_name]
    )
    fee_rate = float(state["fee_bps"]) / 10_000
    slip_rate = float(state["slippage_bps"]) / 10_000
    positions: dict[str, dict[str, Any]] = state["positions"]
    open_trades: dict[str, dict[str, Any]] = state["open_trades"]
    marks: dict[str, float] = state["last_marks"]
    cash = float(state["cash"])

    for pair, index in rows:
        if pair in live_for_event:
            marks[pair] = live_for_event[pair]
        elif index < len(data[pair]):
            marks[pair] = data[pair][index].open
        else:
            continue

    # Close first so the same completed bar can release simulated cash.
    for pair, index in rows:
        position = positions.get(pair)
        if not position:
            continue
        if index >= len(data[pair]) and pair not in live_for_event:
            continue
        if not strategy.desired_long(data[pair][:index], currently_long=True):
            base_price = live_for_event[pair] if pair in live_for_event else data[pair][index].open
            fill = base_price * (1 - slip_rate)
            proceeds = float(position["quantity"]) * fill * (1 - fee_rate)
            cash += proceeds
            trade = open_trades.pop(pair)
            trade.update({
                "exit_timestamp": timestamp, "exit_price": fill,
                "exit_proceeds": proceeds, "pnl": proceeds - float(position["entry_cost"]),
            })
            state["closed_trades"].append(trade)
            del positions[pair]

    candidates: list[tuple[float, str, int]] = []
    for pair, index in rows:
        if pair in positions:
            continue
        if index >= len(data[pair]) and pair not in live_for_event:
            continue
        history = data[pair][:index]
        # Llama selects between these two deterministic baselines. If neither
        # baseline has an entry signal, its label cannot create an entry, so
        # skip an unnecessary local-model call for this pair.
        if isinstance(strategy, LlamaRegimeSwitch) and not (
            strategy.trend.desired_long(history, currently_long=False)
            or strategy.mean_reversion.desired_long(history, currently_long=False)
        ):
            continue
        if strategy.desired_long(history, currently_long=False):
            strength = strategy.entry_strength(history)
            candidates.append((strength, pair, index))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, pair, index in candidates:
        if len(positions) >= int(state["max_positions"]) or cash <= 0:
            break
        base_price = live_for_event[pair] if pair in live_for_event else data[pair][index].open
        fill = base_price * (1 + slip_rate)
        equity = cash + sum(float(p["quantity"]) * float(marks.get(name, p["entry_price"])) for name, p in positions.items())
        budget = min(cash, equity * float(state["max_position_fraction"]))
        quantity = budget / (fill * (1 + fee_rate))
        entry_cost = quantity * fill * (1 + fee_rate)
        if quantity <= 0 or entry_cost > cash:
            continue
        cash -= entry_cost
        positions[pair] = {
            "pair": pair, "quantity": quantity, "entry_timestamp": timestamp,
            "entry_price": fill, "entry_cost": entry_cost,
        }
        open_trades[pair] = {
            "pair": pair, "entry_timestamp": timestamp, "exit_timestamp": None,
            "quantity": quantity, "entry_price": fill, "exit_price": None,
            "entry_cost": entry_cost, "exit_proceeds": None, "pnl": None,
        }

    # After executing at this bar's open, mark to this bar's closed price.
    for pair, bars in data.items():
        eligible = [bar.close for bar in bars if bar.timestamp <= timestamp]
        if pair in live_for_event:
            marks[pair] = live_for_event[pair]
        elif eligible:
            marks[pair] = eligible[-1]
    equity = cash + sum(float(p["quantity"]) * float(marks.get(pair, p["entry_price"])) for pair, p in positions.items())
    state["cash"] = cash
    state["cursor"] = timestamp
    if isinstance(strategy, LlamaRegimeSwitch):
        state["model_notes"] = strategy.notes
    state["equity_curve"].append({"timestamp": timestamp, "equity": equity})
    store.save_paper_state(state, strategy_name)
    return paper_summary(state)


def paper_summary(state: dict[str, Any]) -> dict[str, Any]:
    positions = state["positions"]
    marks = state["last_marks"]
    cash = float(state["cash"])
    open_positions = []
    for pair, position in positions.items():
        mark = float(marks.get(pair, position["entry_price"]))
        open_positions.append({
            "pair": pair, "quantity": position["quantity"], "mark_price": mark,
            "unrealized_pnl": float(position["quantity"]) * mark - float(position["entry_cost"]),
        })
    equity = cash + sum(float(row["quantity"]) * float(marks.get(pair, row["entry_price"])) for pair, row in positions.items())
    start = float(state["starting_cash"])
    closed = state["closed_trades"]
    return {
        "active": state["active"], "strategy": state["strategy"], "model": state.get("model"), "cursor": state["cursor"],
        "starting_cash": start, "cash": cash, "equity": equity,
        "pnl": equity - start, "return_pct": (equity / start - 1) * 100,
        "realized_pnl": sum(float(t["pnl"]) for t in closed),
        "closed_trades": len(closed), "open_positions": open_positions,
        "trade_log": closed[-20:], "model_notes": state.get("model_notes", {}),
    }


def watch_paper_account(
    store: MarketStore, strategy_name: str, poll_seconds: int = 300, once: bool = False,
) -> None:
    """Refresh public candles and paper-trade from the latest completed signal bar."""
    if poll_seconds < 30:
        raise ValueError("Poll interval must be at least 30 seconds")
    state = store.get_paper_state(strategy_name)
    if not state or not state.get("active"):
        raise ValueError(f"No active {strategy_name} paper account. Start one first.")
    interval = int(state["interval"])
    pairs = list(state["pairs"])

    while True:
        feed_errors: list[dict[str, str]] = []
        live_ticks: dict[str, tuple[int, float]] = {}
        for pair in pairs:
            try:
                candles = fetch_ohlc(pair, interval, include_current=True)
                if candles:
                    store.save_candles(candles[:-1])
                    live_ticks[pair] = (candles[-1].timestamp, candles[-1].close)
            except Exception as exc:
                feed_errors.append({"pair": pair, "error": str(exc)})
                print(json.dumps({"pair": pair, "data_error": str(exc)}), flush=True)
        current = store.get_paper_state(strategy_name)
        if current and current.get("active"):
            current["last_check_timestamp"] = int(time.time())
            current["feed_errors"] = feed_errors
            store.save_paper_state(current, strategy_name)
        while True:
            try:
                snapshot = advance_paper_account(store, strategy_name, live_ticks)
                print(json.dumps(snapshot, allow_nan=False), flush=True)
            except NoNewCandles:
                break
        current = store.get_paper_state(strategy_name)
        if current and current.get("active"):
            current.setdefault("last_marks", {}).update({
                pair: price for pair, (_, price) in live_ticks.items()
            })
            store.save_paper_state(current, strategy_name)
        if once:
            return
        time.sleep(poll_seconds)