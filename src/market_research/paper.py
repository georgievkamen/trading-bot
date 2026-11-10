from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
import json
import time
from typing import Any

from .ai_strategy import LlamaRegimeSwitch
from .local_model import LocalOllamaAdvisor
from .models import Candle
from .kraken import fetch_ohlc, is_crypto_usd_pair
from .simulator import Costs, Simulator
from .strategies import STRATEGIES, Strategy, prepare_universe, should_remain_long
from .storage import MarketStore


class NoNewCandles(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    """A live price together with both its candle bucket and actual fetch time."""

    candle_timestamp: int
    price: float
    fetched_at: float


PAPER_STRATEGY_NAMES = (*STRATEGIES.keys(), "llama_regime")
LIVE_SNAPSHOT_MAX_SKEW_SECONDS = 30


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
    live_ticks: dict[str, LiveSnapshot] | None = None,
    *, live_only: bool = False,
) -> dict[str, Any]:
    state = store.get_paper_state(strategy_name)
    if not state or not state.get("active"):
        raise ValueError("No active paper account. Start one first.")
    Simulator(
        float(state.get("starting_cash", 1000.0)),
        Costs(float(state.get("fee_bps", 26.0)), float(state.get("slippage_bps", 5.0))),
        float(state.get("max_position_fraction", 0.10)),
        int(state.get("max_positions", 6)),
    )
    data = store.get_candles(int(state["interval"]), list(state["pairs"]))
    events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    if not live_only:
        for pair, bars in data.items():
            for index, bar in enumerate(bars):
                if bar.timestamp > int(state["cursor"]):
                    events[bar.timestamp].append((pair, index))
    # Watch mode uses only prices fetched in this poll. A stopped watcher must
    # not synthesize fills at the historical opens of candles it missed.
    for pair, snapshot in (live_ticks or {}).items():
        bars = data.get(pair, [])
        if (
            bars
            and snapshot.candle_timestamp > int(state["cursor"])
            and bars[-1].timestamp < snapshot.candle_timestamp
        ):
            events.setdefault(snapshot.candle_timestamp, []).append((pair, len(bars)))
    # A delayed pair response can land in a different candle bucket. Process
    # only the newest shared bucket in live mode; older snapshots are stale.
    if live_only and events:
        latest_live_event = max(events)
        events = defaultdict(list, {latest_live_event: events[latest_live_event]})
    if not events:
        raise NoNewCandles("No newer market-data event is available yet.")

    timestamp = min(events)
    rows = sorted(events[timestamp], key=lambda item: item[0])
    live_for_event = {
        pair: float(live_ticks[pair].price)
        for pair, index in rows
        if live_ticks and pair in live_ticks
        and live_ticks[pair].candle_timestamp == timestamp
        and index == len(data.get(pair, []))
    }
    strategy_name = str(state["strategy"])
    learning_feedback: dict[str, Any] | None = None
    if strategy_name == "llama_regime":
        store.resolve_ai_observations(
            strategy_name, data, timestamp, int(state["interval"]),
            float(state["fee_bps"]), float(state["slippage_bps"]),
        )
        learning_feedback = store.get_ai_feedback(strategy_name, timestamp)
    strategy: Strategy = (
        LlamaRegimeSwitch(
            str(state.get("model") or LocalOllamaAdvisor.DEFAULT_MODEL),
            learning_feedback=learning_feedback,
        )
        if strategy_name == "llama_regime" else STRATEGIES[strategy_name]
    )
    completed_histories = {
        pair: [bar for bar in bars if bar.timestamp < timestamp]
        for pair, bars in data.items()
    }
    prepare_universe(strategy, completed_histories, timestamp)
    fee_rate = float(state["fee_bps"]) / 10_000
    slip_rate = float(state["slippage_bps"]) / 10_000
    positions: dict[str, dict[str, Any]] = state["positions"]
    open_trades: dict[str, dict[str, Any]] = state["open_trades"]
    marks: dict[str, float] = state["last_marks"]
    cash = float(state["cash"])

    if isinstance(strategy, LlamaRegimeSwitch):
        model_histories: dict[str, list[Candle]] = {}
        for pair, index in rows:
            history = data[pair][:index]
            if strategy.eligible_for_classification(history, currently_long=pair in positions):
                model_histories[pair] = history
        strategy.prepare_batch(model_histories)

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
        if not should_remain_long(strategy, data[pair][:index], position):
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
        if isinstance(strategy, LlamaRegimeSwitch) and not strategy.eligible_for_classification(history, currently_long=False):
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
    previous_cursor = int(state["cursor"])
    state["cursor"] = timestamp
    if live_only:
        state["last_processing_mode"] = "live_forward"
        for pair, snapshot in (live_ticks or {}).items():
            if snapshot.candle_timestamp == timestamp:
                state.setdefault("market_snapshot_timestamps", {})[pair] = snapshot.fetched_at
        interval_seconds = int(state["interval"]) * 60
        if timestamp - previous_cursor > interval_seconds:
            state["last_downtime_gap"] = {
                "mode": "skipped_missed_events",
                "from_cursor": previous_cursor,
                "resumed_at": timestamp,
                "estimated_missed_intervals": max(0, (timestamp - previous_cursor) // interval_seconds - 1),
                "detected_at": int(time.time()),
            }
    else:
        state["last_processing_mode"] = "historical_replay"
    if isinstance(strategy, LlamaRegimeSwitch):
        notes = strategy.notes
        state["model_notes"] = notes
        observations = [
            {
                "pair": pair,
                "decision_timestamp": note["decision_timestamp"],
                "close": note["decision_close"],
                "regime": note.get("predicted_regime", note["regime"]),
                "confidence": note["confidence"],
                "news": note.get("news", []),
            }
            for pair, note in notes.items()
            if "decision_timestamp" in note and "decision_close" in note
        ]
        store.record_ai_observations(strategy_name, observations)
        state["learning_stats"] = store.get_ai_feedback(strategy_name, timestamp)
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
    trade_realized_pnl = sum(float(t["pnl"]) for t in closed)
    dividend_income = float(state.get("dividend_income", 0.0))
    return {
        "active": state["active"], "strategy": state["strategy"], "model": state.get("model"), "cursor": state["cursor"],
        "starting_cash": start, "cash": cash, "equity": equity,
        "pnl": equity - start, "return_pct": (equity / start - 1) * 100,
        "realized_pnl": trade_realized_pnl + dividend_income,
        "trade_realized_pnl": trade_realized_pnl,
        "dividend_income": dividend_income,
        "closed_trades": len(closed), "open_positions": open_positions,
        "trade_log": closed[-20:], "model_notes": state.get("model_notes", {}),
        "learning_stats": state.get("learning_stats", {
            "resolved_samples": 0, "ready": False,
            "status": "collecting_more_resolved_examples", "by_regime": {},
        }),
    }


def watch_paper_account(
    store: MarketStore, strategy_name: str, poll_seconds: int = 300, once: bool = False,
    replay_missed_history: bool = False,
) -> None:
    """Refresh public candles; default to forward-only fills after downtime."""
    if poll_seconds < 30:
        raise ValueError("Poll interval must be at least 30 seconds")
    state = store.get_paper_state(strategy_name)
    if not state or not state.get("active"):
        raise ValueError(f"No active {strategy_name} paper account. Start one first.")
    interval = int(state["interval"])
    pairs = list(state["pairs"])

    if replay_missed_history:
        print(json.dumps({"warning": "Historical catch-up replay is enabled; missed candles may receive simulated historical-open fills."}), flush=True)
    while True:
        feed_errors: list[dict[str, str]] = []
        live_ticks: dict[str, LiveSnapshot] = {}

        def fetch_pair(pair: str) -> tuple[str, list[Any], float]:
            candles = fetch_ohlc(pair, interval, include_current=True)
            return pair, candles, time.time()

        # Bounded parallel fetches keep a multi-market snapshot close in time
        # without sending an unbounded burst to the public data endpoint.
        with ThreadPoolExecutor(max_workers=min(5, max(1, len(pairs)))) as pool:
            futures = {pool.submit(fetch_pair, pair): pair for pair in pairs}
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    _, candles, fetched_at = future.result()
                    if candles:
                        store.save_candles(candles[:-1])
                        live_ticks[pair] = LiveSnapshot(
                            candles[-1].timestamp, candles[-1].close, fetched_at,
                        )
                except Exception as exc:
                    feed_errors.append({"pair": pair, "error": str(exc)})
                    print(json.dumps({"pair": pair, "data_error": str(exc)}), flush=True)

        successful_markets = len(live_ticks)
        snapshot_excluded_pairs: set[str] = set()
        observed_skew = 0.0
        if live_ticks:
            # First require all included prices to refer to the same candle
            # bucket, then bound their real fetch-time skew.
            newest_candle = max(snapshot.candle_timestamp for snapshot in live_ticks.values())
            for pair, snapshot in list(live_ticks.items()):
                if snapshot.candle_timestamp != newest_candle:
                    snapshot_excluded_pairs.add(pair)
                    del live_ticks[pair]
            if live_ticks:
                fetched_times = [snapshot.fetched_at for snapshot in live_ticks.values()]
                observed_skew = max(fetched_times) - min(fetched_times)
                newest_fetch = max(fetched_times)
                for pair, snapshot in list(live_ticks.items()):
                    if newest_fetch - snapshot.fetched_at > LIVE_SNAPSHOT_MAX_SKEW_SECONDS:
                        snapshot_excluded_pairs.add(pair)
                        del live_ticks[pair]

        current = store.get_paper_state(strategy_name)
        if current and current.get("active"):
            current["last_check_timestamp"] = int(time.time())
            current["feed_errors"] = feed_errors
            current["market_status"] = {
                "requested_markets": len(pairs),
                "successful_markets": successful_markets,
                "failed_markets": len(feed_errors),
                "snapshot_excluded_markets": len(snapshot_excluded_pairs),
                "synchronized_markets": len(live_ticks),
                "snapshot_skew_seconds": round(observed_skew, 1),
            }
            store.save_paper_state(current, strategy_name)
        while True:
            try:
                snapshot = advance_paper_account(
                    store, strategy_name, live_ticks,
                    live_only=not replay_missed_history,
                )
                state_after_step = store.get_paper_state(strategy_name) or {}
                gap = state_after_step.get("last_downtime_gap")
                if (
                    not replay_missed_history
                    and isinstance(gap, dict)
                    and int(gap.get("resumed_at", -1)) == int(state_after_step.get("cursor", -2))
                ):
                    print(json.dumps({"downtime": gap, "note": "Skipped missed candle fills; resumed at current public prices."}), flush=True)
                print(json.dumps(snapshot, allow_nan=False), flush=True)
            except NoNewCandles:
                break
        current = store.get_paper_state(strategy_name)
        if current and current.get("active"):
            current.setdefault("last_marks", {}).update({
                pair: snapshot.price for pair, snapshot in live_ticks.items()
            })
            current["market_snapshot_timestamps"] = {
                pair: snapshot.fetched_at for pair, snapshot in live_ticks.items()
            }
            store.save_paper_state(current, strategy_name)
        if once:
            return
        time.sleep(poll_seconds)
