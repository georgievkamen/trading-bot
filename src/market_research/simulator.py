from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from .models import Candle, EquityPoint, Position, SimulationResult, Trade
from .strategies import Strategy


@dataclass(frozen=True, slots=True)
class Costs:
    fee_bps: float = 26.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Costs cannot be negative")

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
        max_position_fraction: float = 0.25,
        max_positions: int = 4,
    ) -> None:
        if not isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        if not 0 < max_position_fraction <= 1 or max_positions < 1:
            raise ValueError("Invalid position limits")
        self.initial_cash = initial_cash
        self.costs = costs
        self.max_position_fraction = max_position_fraction
        self.max_positions = max_positions

    def run(self, candles_by_pair: dict[str, list[Candle]], strategy: Strategy) -> SimulationResult:
        normalized = {pair: sorted(rows, key=lambda c: c.timestamp) for pair, rows in candles_by_pair.items() if rows}
        if not normalized:
            raise ValueError("No candles available")
        events: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for pair, bars in normalized.items():
            for index in range(1, len(bars)):
                events[bars[index].timestamp].append((pair, index))
        if not events:
            raise ValueError("At least two candles per pair are required")

        cash = self.initial_cash
        positions: dict[str, Position] = {}
        open_trades: dict[str, Trade] = {}
        completed: list[Trade] = []
        last_price: dict[str, float] = {}
        curve: list[EquityPoint] = []

        for timestamp in sorted(events):
            rows = sorted(events[timestamp], key=lambda entry: entry[0])
            for pair, index in rows:
                # At the open, only the previous close is known; current close is
                # reserved for post-fill marking below.
                last_price[pair] = normalized[pair][index].open

            # Process exits before entries so simulated proceeds can fund new positions.
            for pair, index in rows:
                position = positions.get(pair)
                if position and not strategy.desired_long(normalized[pair][:index], currently_long=True):
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
            last_price[pair] = bars[-1].close
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
