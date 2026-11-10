from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Protocol

from .models import Candle


class Strategy(Protocol):
    name: str

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool: ...

    def entry_strength(self, history: list[Candle]) -> float: ...


def prepare_universe(
    strategy: Strategy, histories: dict[str, list[Candle]], timestamp: int,
) -> None:
    """Give cross-sectional strategies only completed bars before this event."""
    method = getattr(strategy, "prepare_universe", None)
    if callable(method):
        method(histories, timestamp)


def should_remain_long(strategy: Strategy, history: list[Candle], position: Any) -> bool:
    """Pass entry context to rules that need a per-position trailing stop."""
    method = getattr(strategy, "desired_long_for_position", None)
    if callable(method):
        return bool(method(history, position))
    return strategy.desired_long(history, currently_long=True)


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


@dataclass(frozen=True, slots=True)
class TrendFilteredMeanReversion:
    """Only take mean-reversion trades while the broader local trend is up."""

    period: int = 20
    entry_z: float = -2.0
    trend_period: int = 50
    name: str = "mean_reversion_filtered"

    def _trend_is_up(self, history: list[Candle]) -> bool:
        if len(history) < max(self.period, self.trend_period):
            return False
        trend_mean = fmean(bar.close for bar in history[-self.trend_period:])
        return history[-1].close > trend_mean

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        if not self._trend_is_up(history):
            return False
        return MeanReversion(period=self.period, entry_z=self.entry_z).desired_long(history, currently_long)

    def entry_strength(self, history: list[Candle]) -> float:
        if not self._trend_is_up(history):
            return 0.0
        return MeanReversion(period=self.period, entry_z=self.entry_z).entry_strength(history)


def _true_ranges(history: list[Candle]) -> list[float]:
    result: list[float] = []
    for previous, current in zip(history, history[1:]):
        result.append(max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        ))
    return result


def _volatility_caps(history: list[Candle]) -> tuple[float, float]:
    """Return (ATR/price, per-bar return-volatility) ceilings by market cadence."""
    if not history:
        return 0.08, 0.08
    latest = history[-1]
    is_equity = "/" not in latest.pair
    if latest.interval <= 15:
        return (0.01, 0.01) if is_equity else (0.015, 0.015)
    if latest.interval <= 60:
        return 0.04, 0.04
    if is_equity:
        # A 3% daily ATR and 2.5% daily close-return volatility already imply
        # unusually large single-name swings; keep these names in cash.
        return 0.03, 0.025
    return 0.08, 0.08


@dataclass(frozen=True, slots=True)
class VolatilityAdjustedBreakout:
    """Donchian breakout with expanding ATR, trend confirmation and ATR exit."""

    breakout_period: int = 20
    fast_atr_period: int = 5
    atr_period: int = 20
    trend_period: int = 60
    atr_expansion_ratio: float = 1.05
    trailing_atr_multiple: float = 3.0
    max_atr_fraction: float = 0.08
    name: str = "volatility_breakout"

    @staticmethod
    def _atr(history: list[Candle], period: int) -> float:
        ranges = _true_ranges(history)
        if len(ranges) < period:
            return 0.0
        return fmean(ranges[-period:])

    def _entry_signal(self, history: list[Candle]) -> bool:
        minimum = max(self.breakout_period + 1, self.atr_period + 1, self.trend_period)
        if len(history) < minimum:
            return False
        close = history[-1].close
        prior_high = max(bar.high for bar in history[-self.breakout_period - 1:-1])
        fast_atr = self._atr(history, self.fast_atr_period)
        slow_atr = self._atr(history, self.atr_period)
        trend_values = [bar.close for bar in history[-self.trend_period:]]
        confirmation = fmean(trend_values[-20:]) > fmean(trend_values)
        atr_cap, _ = _volatility_caps(history)
        return (
            close > prior_high
            and slow_atr > 0
            and slow_atr / close <= min(self.max_atr_fraction, atr_cap)
            and fast_atr >= slow_atr * self.atr_expansion_ratio
            and close > fmean(trend_values)
            and confirmation
        )

    def _exit_signal(self, history: list[Candle], entry_timestamp: int | None) -> bool:
        minimum = max(self.atr_period + 1, self.trend_period)
        if len(history) < minimum:
            return False
        atr = self._atr(history, self.atr_period)
        if atr <= 0:
            return False
        atr_cap, _ = _volatility_caps(history)
        held = (
            [bar for bar in history if bar.timestamp >= entry_timestamp]
            if entry_timestamp is not None else history[-self.breakout_period:]
        )
        if not held:
            held = history[-self.breakout_period:]
        trailing_high = max(bar.high for bar in held)
        trailing_stop = trailing_high - self.trailing_atr_multiple * atr
        momentum_faded = history[-1].close < fmean(bar.close for bar in history[-20:])
        volatility_spike = atr / history[-1].close > min(self.max_atr_fraction, atr_cap)
        return history[-1].close <= trailing_stop or momentum_faded or volatility_spike

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        if currently_long:
            return not self._exit_signal(history, None)
        return self._entry_signal(history)

    def desired_long_for_position(self, history: list[Candle], position: Any) -> bool:
        if isinstance(position, dict):
            entry_timestamp = int(position.get("entry_timestamp", 0)) or None
        else:
            entry_timestamp = int(getattr(position, "entry_timestamp", 0)) or None
        return not self._exit_signal(history, entry_timestamp)

    def entry_strength(self, history: list[Candle]) -> float:
        if not self._entry_signal(history):
            return 0.0
        prior_high = max(bar.high for bar in history[-self.breakout_period - 1:-1])
        return max(0.0, history[-1].close / prior_high - 1.0)


@dataclass(slots=True)
class RelativeMomentum:
    """Weekly dual momentum: rank cross-sectionally, require positive absolute return."""

    lookback: int = 60
    top_k: int = 6
    minimum_return: float = 0.01
    max_bar_volatility: float = 0.08
    name: str = "relative_momentum"
    _scores: dict[str, float] | None = None
    _selected: set[str] | None = None
    _prepared_window: int | None = None

    REBALANCE_SECONDS = 7 * 24 * 60 * 60
    # Unix epoch begins Thursday; this anchor makes weekly windows start Monday 00:00 UTC.
    WEEK_ANCHOR = -3 * 24 * 60 * 60

    def reset(self) -> None:
        self._scores = {}
        self._selected = set()
        self._prepared_window = None

    def prepare_universe(self, histories: dict[str, list[Candle]], timestamp: int) -> None:
        boundary = self.WEEK_ANCHOR + ((timestamp - self.WEEK_ANCHOR) // self.REBALANCE_SECONDS) * self.REBALANCE_SECONDS
        if self._prepared_window == boundary:
            return
        scores: dict[str, float] = {}
        for symbol, bars in histories.items():
            completed = [bar for bar in bars if bar.timestamp < boundary]
            if len(completed) <= self.lookback:
                continue
            earlier = completed[-self.lookback - 1].close
            latest = completed[-1].close
            if earlier > 0:
                raw_return = latest / earlier - 1.0
                closes = [bar.close for bar in completed[-self.lookback - 1:]]
                returns = [new / old - 1 for old, new in zip(closes, closes[1:]) if old > 0]
                volatility = pstdev(returns) if len(returns) > 1 else float("inf")
                _, volatility_cap = _volatility_caps(completed)
                if raw_return >= self.minimum_return and volatility <= min(self.max_bar_volatility, volatility_cap):
                    # Rank leaders by return per unit of realized volatility.
                    scores[symbol] = raw_return / max(volatility, 1e-6)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        self._scores = dict(ranked)
        self._selected = {symbol for symbol, _ in ranked[:self.top_k]}
        self._prepared_window = boundary

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        if not history or not self._selected or history[-1].pair not in self._selected:
            return False
        if len(history) <= self.lookback:
            return False
        closes = [bar.close for bar in history[-self.lookback - 1:]]
        returns = [new / old - 1 for old, new in zip(closes, closes[1:]) if old > 0]
        _, volatility_cap = _volatility_caps(history)
        return bool(returns and pstdev(returns) <= min(self.max_bar_volatility, volatility_cap))

    def entry_strength(self, history: list[Candle]) -> float:
        if not history or not self._scores:
            return 0.0
        return max(0.0, self._scores.get(history[-1].pair, 0.0))


STRATEGIES: dict[str, Strategy] = {
    "trend": TrendFollowing(),
    "mean_reversion": MeanReversion(),
    "mean_reversion_filtered": TrendFilteredMeanReversion(),
    "volatility_breakout": VolatilityAdjustedBreakout(),
    "relative_momentum": RelativeMomentum(),
}
