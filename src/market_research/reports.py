from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import Candle
from .paper import paper_summary
from .simulator import Costs, Simulator
from .storage import MarketStore
from .strategies import STRATEGIES


def _utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _trade_costs(trade: dict[str, Any], fee_rate: float, slippage_rate: float,
                 *, open_mark: float | None = None) -> dict[str, float]:
    """Split P&L into market movement, fees, and estimated adverse slippage."""
    quantity = float(trade["quantity"])
    entry_fill = float(trade["entry_price"])
    entry_cost = float(trade["entry_cost"])
    entry_reference = entry_fill / (1 + slippage_rate)
    entry_fee = max(0.0, entry_cost - quantity * entry_fill)
    entry_slippage = quantity * (entry_fill - entry_reference)

    if open_mark is None:
        exit_fill = float(trade["exit_price"])
        exit_proceeds = float(trade["exit_proceeds"])
        exit_reference = exit_fill / (1 - slippage_rate)
        exit_fee = max(0.0, quantity * exit_fill - exit_proceeds)
        exit_slippage = quantity * (exit_reference - exit_fill)
        gross_pnl = quantity * (exit_reference - entry_reference)
        net_pnl = exit_proceeds - entry_cost
        estimated = False
    else:
        exit_reference = open_mark
        exit_fill = open_mark * (1 - slippage_rate)
        exit_proceeds = quantity * exit_fill * (1 - fee_rate)
        exit_fee = quantity * exit_fill * fee_rate
        exit_slippage = quantity * (exit_reference - exit_fill)
        gross_pnl = quantity * (exit_reference - entry_reference)
        net_pnl = exit_proceeds - entry_cost
        estimated = True

    return {
        "gross_price_pnl_before_costs": gross_pnl,
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "entry_slippage_cost_estimate": entry_slippage,
        "exit_slippage_cost_estimate": exit_slippage,
        "total_fees": entry_fee + exit_fee,
        "total_slippage_cost_estimate": entry_slippage + exit_slippage,
        "net_pnl": net_pnl,
        "estimated_exit": estimated,
    }


def paper_trade_attribution(state: dict[str, Any]) -> dict[str, Any]:
    """Read-only cost and P&L attribution for a persisted paper account."""
    fee_rate = float(state["fee_bps"]) / 10_000
    slippage_rate = float(state["slippage_bps"]) / 10_000
    closed_rows = list(state.get("closed_trades", []))
    open_rows = list(state.get("open_trades", {}).values())
    closed = [_trade_costs(row, fee_rate, slippage_rate) for row in closed_rows]
    marks = state.get("last_marks", {})
    opened = [
        _trade_costs(row, fee_rate, slippage_rate, open_mark=float(marks.get(row["pair"], row["entry_price"])))
        for row in open_rows
    ]
    by_pair: dict[str, dict[str, float]] = {}
    for trade, attribution in zip(closed_rows, closed):
        row = by_pair.setdefault(str(trade["pair"]), {
            "closed_trades": 0, "wins": 0, "gross_price_pnl_before_costs": 0.0,
            "fees": 0.0, "estimated_slippage_cost": 0.0, "net_pnl": 0.0,
        })
        row["closed_trades"] += 1
        row["wins"] += float(attribution["net_pnl"] > 0)
        row["gross_price_pnl_before_costs"] += float(attribution["gross_price_pnl_before_costs"])
        row["fees"] += float(attribution["total_fees"])
        row["estimated_slippage_cost"] += float(attribution["total_slippage_cost_estimate"])
        row["net_pnl"] += float(attribution["net_pnl"])

    pair_summary = {
        pair: {
            **values,
            "win_rate_pct": values["wins"] / values["closed_trades"] * 100,
        }
        for pair, values in sorted(by_pair.items())
    }

    def total(rows: list[dict[str, Any]], field: str) -> float:
        return sum(float(row[field]) for row in rows)

    closed_net = total(closed, "net_pnl")
    recorded_realized = sum(float(row["pnl"]) for row in closed_rows)
    open_net_liquidation = total(opened, "net_pnl")
    summary = paper_summary(state)
    return {
        "account": state["strategy"],
        "as_of_cursor": _utc(int(state["cursor"])),
        "latest_mark_as_of": _utc(int(max(state.get("market_snapshot_timestamps", {}).values(), default=state["cursor"]))),
        "basis_points": {"fee_per_side": state["fee_bps"], "slippage_per_side": state["slippage_bps"]},
        "closed_trades": len(closed_rows),
        "closed_trade_totals": {
            "gross_price_pnl_before_costs": total(closed, "gross_price_pnl_before_costs"),
            "fees": total(closed, "total_fees"),
            "estimated_slippage_cost": total(closed, "total_slippage_cost_estimate"),
            "net_pnl": closed_net,
            "ledger_realized_pnl": recorded_realized,
            "reconciliation_difference": closed_net - recorded_realized,
        },
        "closed_trades_by_pair": pair_summary,
        "open_positions": len(open_rows),
        "open_position_estimates_by_pair": {
            str(trade["pair"]): {
                "mark_price": float(marks.get(trade["pair"], trade["entry_price"])),
                "estimated_net_liquidation_pnl": float(attribution["net_pnl"]),
                "estimated_exit_fee": float(attribution["exit_fee"]),
                "estimated_exit_slippage_cost": float(attribution["exit_slippage_cost_estimate"]),
            }
            for trade, attribution in zip(open_rows, opened)
        },
        "open_position_estimates": {
            "gross_price_pnl_before_costs": total(opened, "gross_price_pnl_before_costs"),
            "fees_including_estimated_exit": total(opened, "total_fees"),
            "estimated_slippage_including_exit": total(opened, "total_slippage_cost_estimate"),
            "estimated_net_liquidation_pnl": open_net_liquidation,
            "note": "Open-position exit fee and slippage are estimates at the saved mark; no exit has occurred.",
        },
        "account_equity": summary["equity"],
        "account_pnl": summary["pnl"],
        "account_return_pct": summary["return_pct"],
    }


def _buy_and_hold(data: dict[str, list[Candle]], start: int, end: int,
                  initial_cash: float, costs: Costs) -> dict[str, Any]:
    """Equal-weight the available universe at first in-window opens, then hold."""
    entries: list[tuple[str, Candle, Candle]] = []
    for pair, bars in data.items():
        in_window = [bar for bar in bars if start <= bar.timestamp <= end]
        if in_window:
            entries.append((pair, in_window[0], in_window[-1]))
    if not entries:
        raise ValueError("No stored candles overlap the paper account period")

    allocation = initial_cash / len(entries)
    remaining_cash = 0.0
    holdings = []
    for pair, first, last in entries:
        entry_fill = first.open * (1 + costs.slippage_rate)
        quantity = allocation / (entry_fill * (1 + costs.fee_rate))
        entry_cost = quantity * entry_fill * (1 + costs.fee_rate)
        exit_fill = last.close * (1 - costs.slippage_rate)
        exit_proceeds = quantity * exit_fill * (1 - costs.fee_rate)
        remaining_cash += allocation - entry_cost
        holdings.append({
            "pair": pair,
            "entry_timestamp": first.timestamp,
            "mark_timestamp": last.timestamp,
            "quantity": quantity,
            "entry_cost": entry_cost,
            "exit_proceeds": exit_proceeds,
            "net_pnl": exit_proceeds - entry_cost,
        })
    ending_equity = remaining_cash + sum(float(row["exit_proceeds"]) for row in holdings)
    marks = [int(row["mark_timestamp"]) for row in holdings]
    return {
        "strategy": "equal_weight_buy_and_hold",
        "starting_cash": initial_cash,
        "ending_equity": ending_equity,
        "pnl": ending_equity - initial_cash,
        "return_pct": (ending_equity / initial_cash - 1) * 100,
        "symbols_held": len(holdings),
        "cash_remaining_after_entry_fees": remaining_cash,
        "mark_timestamp_min": min(marks),
        "mark_timestamp_max": max(marks),
        "mark_staleness_seconds_at_period_end_max": max(end - value for value in marks),
    }


def paper_period_comparison(store: MarketStore, strategy_name: str) -> dict[str, Any]:
    """Compare a saved paper account with same-window, same-universe baselines."""
    state = store.get_paper_state(strategy_name)
    if not state:
        raise ValueError(f"No {strategy_name} paper account exists")
    curve = state.get("equity_curve", [])
    if not curve:
        raise ValueError("The account has no recorded forward equity points yet")

    start = int(curve[0]["timestamp"])
    end = int(state["cursor"])
    pairs = list(state["pairs"])
    interval = int(state["interval"])
    data = store.get_candles(interval, pairs)
    data = {pair: bars for pair, bars in data.items() if bars}
    if not data:
        raise ValueError("No stored candles are available for the account's market universe")

    costs = Costs(float(state["fee_bps"]), float(state["slippage_bps"]))
    simulator = Simulator(
        initial_cash=float(state["starting_cash"]), costs=costs,
        max_position_fraction=float(state["max_position_fraction"]),
        max_positions=int(state["max_positions"]),
    )
    fixed_baselines = []
    for name, strategy in STRATEGIES.items():
        result = simulator.run(data, strategy, from_timestamp=start, through_timestamp=end)
        fixed_baselines.append({
            "strategy": name,
            "ending_equity": result.ending_equity,
            "pnl": result.ending_equity - result.initial_cash,
            "return_pct": result.return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "closed_trades": result.closed_trades,
            "open_positions": len(result.open_positions),
        })

    summary = paper_summary(state)
    buy_hold = _buy_and_hold(data, start, end, float(state["starting_cash"]), costs)
    peak = float(state["starting_cash"])
    paper_max_drawdown = 0.0
    for point in curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        if peak > 0:
            paper_max_drawdown = max(paper_max_drawdown, (peak - equity) / peak)
    available_pairs = sorted(data)
    if set(available_pairs) != set(pairs):
        missing = sorted(set(pairs) - set(available_pairs))
    else:
        missing = []
    return {
        "account": strategy_name,
        "model": state.get("model"),
        "period": {
            "start_timestamp": start, "start_utc": _utc(start),
            "last_processed_timestamp": end, "last_processed_utc": _utc(end),
            "latest_live_mark_utc": _utc(int(max(state.get("market_snapshot_timestamps", {}).values(), default=end))),
        },
        "interval_minutes": interval,
        "universe": {"requested_symbols": len(pairs), "symbols_with_stored_data": len(available_pairs), "missing_symbols": missing, "symbols": available_pairs},
        "costs_bps_per_side": {"fee": costs.fee_bps, "slippage": costs.slippage_bps},
        "risk_limits": {"max_position_fraction": state["max_position_fraction"], "max_positions": state["max_positions"]},
        "results": [
            {"strategy": strategy_name, "ending_equity": summary["equity"], "pnl": summary["pnl"], "return_pct": summary["return_pct"], "max_drawdown_pct": paper_max_drawdown * 100, "closed_trades": summary["closed_trades"], "open_positions": len(summary["open_positions"]), "source": "actual persisted paper ledger"},
            {"strategy": "cash", "ending_equity": float(state["starting_cash"]), "pnl": 0.0, "return_pct": 0.0, "source": "no trades"},
            buy_hold,
            *fixed_baselines,
        ],
        "execution_and_data_notes": [
            "Fixed baselines use the same stored symbols, interval, date bounds, fees, slippage, starting cash, position fraction, and position-count cap as the account.",
            "Llama uses forward public-price snapshots; fixed baselines use decisions from completed candles and next-bar-open fills. These are same-period comparisons, not identical-fill controlled experiments.",
            "Buy-and-hold is equal-weighted across symbols with data in the window, entered at each symbol's first in-window open; it is fully invested and does not use the strategy position cap.",
            "Historical ending marks use the latest stored closed candle at or before the account cursor; live paper marks may be newer.",
            "This short, selected-universe sample is descriptive and does not establish future profitability.",
        ],
    }
