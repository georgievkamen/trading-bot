from __future__ import annotations

from .local_model import LocalOllamaAdvisor
from .models import Candle
from .news import CryptoNewsFeed
from .strategies import (
    MeanReversion, RelativeMomentum, Strategy, TrendFollowing,
    VolatilityAdjustedBreakout, prepare_universe, should_remain_long,
)


class LlamaRegimeSwitch:
    """Use bounded regime labels to choose one approved deterministic strategy or cash."""

    name = "llama_regime"
    minimum_confidence = 0.60
    classification_batch_size = 3

    def __init__(
        self,
        model: str = LocalOllamaAdvisor.DEFAULT_MODEL,
        learning_feedback: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.learning_feedback = learning_feedback
        self.advisor = LocalOllamaAdvisor(model)
        self.news_feed = CryptoNewsFeed()
        self.strategies: dict[str, Strategy] = {
            "trend": TrendFollowing(),
            "range": MeanReversion(),
            "breakout": VolatilityAdjustedBreakout(),
            "relative_momentum": RelativeMomentum(),
        }
        self._cache: dict[tuple[str, int], str] = {}
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
        """Classify eligible markets in bounded requests before portfolio decisions."""
        eligible = {
            pair: rows for pair, rows in histories.items()
            if len(rows) >= 30
        }
        pairs = list(eligible)
        for offset in range(0, len(pairs), self.classification_batch_size):
            batch_pairs = pairs[offset:offset + self.classification_batch_size]
            batch = {pair: eligible[pair] for pair in batch_pairs}
            news_by_pair = {pair: self.news_feed.for_pair(pair) for pair in batch_pairs}
            try:
                notes = self.advisor.classify_batch(
                    {pair: [bar.close for bar in rows[-61:]] for pair, rows in batch.items()},
                    timeout=90,
                    news_by_pair=news_by_pair,
                    learning_feedback=self.learning_feedback,
                )
            except RuntimeError as exc:
                for pair, rows in batch.items():
                    last = rows[-1]
                    self._cache[(pair, last.timestamp)] = "unavailable"
                    self._notes[pair] = {
                        "regime": "unavailable",
                        "confidence": 0.0,
                        "rationale": f"Local model unavailable; fail-closed for this bar. {str(exc)[:160]}",
                        "news": [headline.as_dict() for headline in news_by_pair[pair]],
                        "news_status": self.news_feed.status,
                        "decision_timestamp": last.timestamp,
                        "decision_close": last.close,
                    }
                continue
            for pair, note in notes.items():
                last = batch[pair][-1]
                regime = note.regime if note.confidence >= self.minimum_confidence else "unknown"
                self._cache[(pair, last.timestamp)] = regime
                self._notes[pair] = {
                    "regime": regime,
                    "predicted_regime": note.regime,
                    "confidence": note.confidence,
                    "rationale": note.rationale,
                    "news": [headline.as_dict() for headline in note.news],
                    "news_status": self.news_feed.status,
                    "decision_timestamp": last.timestamp,
                    "decision_close": last.close,
                    "learning_feedback": note.learning_feedback,
                }

    def _regime(self, history: list[Candle]) -> str:
        if len(history) < 30:
            return "unknown"
        last = history[-1]
        key = (last.pair, last.timestamp)
        if key in self._cache:
            return self._cache[key]
        headlines = self.news_feed.for_pair(last.pair)
        news = [headline.as_dict() for headline in headlines]
        try:
            note = self.advisor.classify(
                last.pair,
                [bar.close for bar in history[-61:]],
                timeout=60,
                news=headlines,
                learning_feedback=self.learning_feedback,
            )
        except RuntimeError:
            # If the local model is unavailable, flatten existing simulated exposure
            # and prevent new entries until a later successful classification.
            regime = "unavailable"
            self._notes[last.pair] = {
                "regime": regime,
                "confidence": 0.0,
                "rationale": "Local model unavailable; fail-closed for this bar.",
                "news": news,
                "news_status": self.news_feed.status,
                "decision_timestamp": last.timestamp,
                "decision_close": last.close,
            }
        else:
            regime = note.regime if note.confidence >= self.minimum_confidence else "unknown"
            self._notes[last.pair] = {
                "regime": regime,
                "predicted_regime": note.regime,
                "confidence": note.confidence,
                "rationale": note.rationale,
                "news": [headline.as_dict() for headline in note.news],
                "news_status": self.news_feed.status,
                "decision_timestamp": last.timestamp,
                "decision_close": last.close,
                "learning_feedback": note.learning_feedback,
            }
        self._cache[key] = regime
        # Keep the per-process cache bounded during long-running paper sessions.
        if len(self._cache) > 500:
            self._cache = {key: regime}
        return regime

    def desired_long(self, history: list[Candle], currently_long: bool) -> bool:
        regime = self._regime(history)
        strategy = self.strategies.get(regime)
        if strategy is not None:
            return strategy.desired_long(history, currently_long)
        if regime in {"cash", "high_volatility", "unavailable"}:
            return False
        # Uncertain model labels cause no new entries and do not churn an open position.
        return currently_long

    def desired_long_for_position(self, history: list[Candle], position: object) -> bool:
        regime = self._regime(history)
        strategy = self.strategies.get(regime)
        if strategy is not None:
            return should_remain_long(strategy, history, position)
        if regime in {"cash", "high_volatility", "unavailable"}:
            return False
        return True

    def entry_strength(self, history: list[Candle]) -> float:
        regime = self._regime(history)
        strategy = self.strategies.get(regime)
        if strategy is not None:
            return strategy.entry_strength(history)
        return 0.0
