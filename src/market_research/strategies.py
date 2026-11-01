from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Protocol

from .models import Candle


class Strategy(Protocol):
    name: str

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool: ...

    def entry_strength(self, history: list[Candle]) -> float: ...


@dataclass(frozen=True, slots=True)
class TrendFollowing:
    fast_period: int = 10
    slow_period: int = 30
    name: str = "trend"

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        if len(history) < self.slow_period:
            return False
        closes = [bar.close for bar in history[-self.slow_period:]]
        return fmean(closes[-self.fast_period:]) > fmean(closes)

    def entry_strength(self, history: list[Candle]) -> float:
        if len(history) < self.slow_period:
            return 0.0
        closes = [bar.close for bar in history[-self.slow_period:]]
        return max(0.0, fmean(closes[-self.fast_period:]) / fmean(closes) - 1.0)


@dataclass(frozen=True, slots=True)
class MeanReversion:
    period: int = 20
    entry_z: float = -2.0
    name: str = "mean_reversion"

    def _z_score(self, history: list[Candle]) -> tuple[float, float]:
        closes = [bar.close for bar in history[-self.period:]]
        mean = fmean(closes)
        deviation = pstdev(closes)
        return mean, (closes[-1] - mean) / deviation if deviation else 0.0

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        if len(history) < self.period:
            return False
        mean, z_score = self._z_score(history)
        if currently_long:
            return history[-1].close < mean
        return z_score <= self.entry_z

    def entry_strength(self, history: list[Candle]) -> float:
        if len(history) < self.period:
            return 0.0
        _, z_score = self._z_score(history)
        return max(0.0, self.entry_z - z_score)


STRATEGIES: dict[str, Strategy] = {
    "trend": TrendFollowing(),
    "mean_reversion": MeanReversion(),
}
