from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite


@dataclass(frozen=True, slots=True)
class Candle:
    pair: str
    interval: int
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(isfinite(v) for v in values):
            raise ValueError("Candle values must be finite")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("Prices must be positive and volume cannot be negative")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("Candle high is inconsistent")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("Candle low is inconsistent")

    @property
    def datetime_utc(self) -> str:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()


@dataclass(slots=True)
class Position:
    pair: str
    quantity: float
    entry_timestamp: int
    entry_price: float
    entry_cost: float


@dataclass(slots=True)
class Trade:
    pair: str
    entry_timestamp: int
    exit_timestamp: int | None
    quantity: float
    entry_price: float
    exit_price: float | None
    entry_cost: float
    exit_proceeds: float | None
    pnl: float | None


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: int
    equity: float


@dataclass(slots=True)
class SimulationResult:
    strategy: str
    initial_cash: float
    ending_equity: float
    return_pct: float
    max_drawdown_pct: float
    closed_trades: int
    win_rate_pct: float
    profit_factor: float | None
    cash: float
    open_positions: list[dict[str, object]]
    trades: list[Trade]
    equity_curve: list[EquityPoint]

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "initial_cash": self.initial_cash,
            "ending_equity": self.ending_equity,
            "return_pct": self.return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "closed_trades": self.closed_trades,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "cash": self.cash,
            "open_positions": self.open_positions,
            "trades": [asdict(t) for t in self.trades],
            "equity_curve": [asdict(p) for p in self.equity_curve],
        }
