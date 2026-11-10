from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from bisect import bisect_left
from math import isfinite

from .models import Candle, EquityPoint, Position, SimulationResult, Trade
from .strategies import Strategy, prepare_universe, should_remain_long


@dataclass(frozen=True, slots=True)
class Costs:
    fee_bps: float = 26.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if not isfinite(self.fee_bps) or not isfinite(self.slippage_bps):
            raise ValueError("Fee and slippage assumptions must be finite")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Costs cannot be negative")
        if self.slippage_bps >= 10_000:
            raise ValueError("Slippage must be below 10,000 bps so exit prices remain positive")

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000


class Simulator:
    """Long-only cash simulator: completed-history decisions, next-bar-open fills."""

    def __init__(
        self,
        initial_cash: float = 1000.0,
        costs: Costs = Costs(),
        max_position_fraction: float = 0.10,
        max_positions: int = 6,
    ) -> None:
        if not isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        if not 0 < max_position_fraction <= 0.10 or not 1 <= max_positions <= 6:
            raise ValueError("Position limits cannot exceed 10% of equity or six concurrent positions")
        self.initial_cash = initial_cash
        self.costs = costs
        self.max_position_fraction = max_position_fraction
        self.max_positions = max_positions

    def run(
        self,
        candles_by_pair: dict[str, list[Candle]],
        strategy: Strategy,
        from_timestamp: int | None = None,
        through_timestamp: int | None = None,
    ) -> SimulationResult:
        if from_timestamp is not None and through_timestamp is not None and from_timestamp > through_timestamp:
            raise ValueError("Simulation start must not be after its end")
        normalized = {pair: sorted(rows, key=lambda c: c.timestamp) for pair, rows in candles_by_pair.items() if rows}
        if not normalized:
            raise ValueError("No candles available")
        reset = getattr(strategy, "reset", None)
        if callable(reset):
            reset()
        timestamps_by_pair = {
            pair: [bar.timestamp for bar in bars] for pair, bars in normalized.items()
        }
        events: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for pair, bars in normalized.items():
            for index in range(1, len(bars)):
                timestamp = bars[index].timestamp
                if from_timestamp is not None and timestamp < from_timestamp:
                    continue
                if through_timestamp is not None and timestamp > through_timestamp:
                    continue
                events[timestamp].append((pair, index))
        if not events:
            raise ValueError("No candle events fall within the requested simulation period")

        cash = self.initial_cash
        positions: dict[str, Position] = {}
        open_trades: dict[str, Trade] = {}
        completed: list[Trade] = []
        last_price: dict[str, float] = {}
        curve: list[EquityPoint] = []

        for timestamp in sorted(events):
            rows = sorted(events[timestamp], key=lambda entry: entry[0])
            completed_histories = {
                pair: bars[:bisect_left(timestamps_by_pair[pair], timestamp)]
                for pair, bars in normalized.items()
            }
            prepare_universe(strategy, completed_histories, timestamp)
            for pair, index in rows:
                # At the open, only the previous close is known; current close is
                # reserved for post-fill marking below.
                last_price[pair] = normalized[pair][index].open

            # Process exits before entries so simulated proceeds can fund new positions.
            for pair, index in rows:
                position = positions.get(pair)
                if position and not should_remain_long(strategy, normalized[pair][:index], position):
                    fill = normalized[pair][index].open * (1 - self.costs.slippage_rate)
                    proceeds = position.quantity * fill * (1 - self.costs.fee_rate)
                    cash += proceeds
                    trade = open_trades.pop(pair)
                    trade.exit_timestamp = timestamp
                    trade.exit_price = fill
                    trade.exit_proceeds = proceeds
                    trade.pnl = proceeds - position.entry_cost
                    completed.append(trade)
                    del positions[pair]

            candidates: list[tuple[float, str, int]] = []
            for pair, index in rows:
                if pair in positions:
                    continue
                history = normalized[pair][:index]
                if strategy.desired_long(history, currently_long=False):
                    strength = strategy.entry_strength(history)
                    candidates.append((strength, pair, index))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _, pair, index in candidates:
                if len(positions) >= self.max_positions or cash <= 0:
                    break
                fill = normalized[pair][index].open * (1 + self.costs.slippage_rate)
                equity_before = cash + sum(
                    pos.quantity * last_price.get(pos.pair, pos.entry_price) for pos in positions.values()
                )
                budget = min(cash, equity_before * self.max_position_fraction)
                quantity = budget / (fill * (1 + self.costs.fee_rate))
                entry_cost = quantity * fill * (1 + self.costs.fee_rate)
                if quantity <= 0 or entry_cost > cash:
                    continue
                cash -= entry_cost
                positions[pair] = Position(pair, quantity, timestamp, fill, entry_cost)
                open_trades[pair] = Trade(pair, timestamp, None, quantity, fill, None, entry_cost, None, None)

            for pair, index in rows:
                last_price[pair] = normalized[pair][index].close
            equity = cash + sum(
                pos.quantity * last_price.get(pos.pair, pos.entry_price) for pos in positions.values()
            )
            curve.append(EquityPoint(timestamp, equity))

        # Mark open positions with each pair's latest available closed candle.
        for pair, bars in normalized.items():
            eligible = [bar for bar in bars if through_timestamp is None or bar.timestamp <= through_timestamp]
            if eligible:
                last_price[pair] = eligible[-1].close
        ending_equity = cash + sum(
            pos.quantity * last_price.get(pos.pair, pos.entry_price) for pos in positions.values()
        )
        if curve and abs(curve[-1].equity - ending_equity) > 1e-9:
            curve.append(EquityPoint(max(events), ending_equity))

        closed_pnls = [trade.pnl or 0.0 for trade in completed]
        wins = sum(pnl > 0 for pnl in closed_pnls)
        gross_profit = sum(pnl for pnl in closed_pnls if pnl > 0)
        gross_loss = -sum(pnl for pnl in closed_pnls if pnl < 0)
        peak = self.initial_cash
        max_drawdown = 0.0
        for point in curve:
            peak = max(peak, point.equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - point.equity) / peak)
        open_positions = [
            {"pair": p.pair, "quantity": p.quantity, "entry_timestamp": p.entry_timestamp,
             "entry_price": p.entry_price, "mark_price": last_price.get(p.pair, p.entry_price),
             "unrealized_pnl": p.quantity * last_price.get(p.pair, p.entry_price) - p.entry_cost}
            for p in positions.values()
        ]
        count = len(completed)
        return SimulationResult(
            strategy=strategy.name,
            initial_cash=self.initial_cash,
            ending_equity=ending_equity,
            return_pct=(ending_equity / self.initial_cash - 1) * 100,
            max_drawdown_pct=max_drawdown * 100,
            closed_trades=count,
            win_rate_pct=(wins / count * 100) if count else 0.0,
            # Null means no losing closed trades or no closed trades.
            profit_factor=(gross_profit / gross_loss) if gross_loss else None,
            cash=cash,
            open_positions=open_positions,
            trades=completed,
            equity_curve=curve,
        )


def equal_weight_buy_and_hold(
    candles_by_pair: dict[str, list[Candle]], initial_cash: float, costs: Costs,
    *, from_timestamp: int | None = None, through_timestamp: int | None = None,
) -> dict[str, object]:
    """Mark an equal-weight basket after next-open entries; no lookahead or shorts."""
    normalized = {
        pair: sorted(
            [bar for bar in bars if (from_timestamp is None or bar.timestamp >= from_timestamp)
             and (through_timestamp is None or bar.timestamp <= through_timestamp)],
            key=lambda bar: bar.timestamp,
        )
        for pair, bars in candles_by_pair.items()
    }
    normalized = {pair: bars for pair, bars in normalized.items() if bars}
    if not normalized:
        raise ValueError("No candles available for buy-and-hold benchmark")
    allocation = initial_cash / len(normalized)
    events: dict[int, list[tuple[str, Candle]]] = defaultdict(list)
    for pair, bars in normalized.items():
        for bar in bars:
            events[bar.timestamp].append((pair, bar))

    cash = initial_cash
    holdings: dict[str, float] = {}
    marks: dict[str, float] = {}
    curve: list[float] = [initial_cash]
    for timestamp in sorted(events):
        for pair, bar in sorted(events[timestamp], key=lambda row: row[0]):
            if pair not in holdings:
                fill = bar.open * (1 + costs.slippage_rate)
                quantity = allocation / (fill * (1 + costs.fee_rate))
                spent = quantity * fill * (1 + costs.fee_rate)
                cash -= spent
                holdings[pair] = quantity
            marks[pair] = bar.close
        curve.append(cash + sum(quantity * marks.get(pair, 0.0) for pair, quantity in holdings.items()))

    ending_equity = cash + sum(quantity * marks.get(pair, 0.0) for pair, quantity in holdings.items())
    peak = initial_cash
    max_drawdown = 0.0
    for equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return {
        "strategy": "equal_weight_buy_and_hold",
        "ending_equity": ending_equity,
        "pnl": ending_equity - initial_cash,
        "return_pct": (ending_equity / initial_cash - 1) * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "closed_trades": 0,
        "win_rate_pct": None,
        "profit_factor": None,
        "open_positions": len(holdings),
        "pairs": sorted(holdings),
    }
