from __future__ import annotations

from .local_model import LocalOllamaAdvisor
from .models import Candle
from .news import CryptoNewsFeed
from .strategies import MeanReversion, TrendFollowing


class LlamaRegimeSwitch:
    """Use local Llama regime labels to choose a deterministic paper baseline."""

    name = "llama_regime"
    minimum_confidence = 0.60

    def __init__(
        self,
        model: str = LocalOllamaAdvisor.DEFAULT_MODEL,
        learning_feedback: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.learning_feedback = learning_feedback
        self.advisor = LocalOllamaAdvisor(model)
        self.news_feed = CryptoNewsFeed()
        self.trend = TrendFollowing()
        self.mean_reversion = MeanReversion()
        self._cache: dict[tuple[str, int], str] = {}
        self._notes: dict[str, dict[str, object]] = {}

    @property
    def notes(self) -> dict[str, dict[str, object]]:
        return {pair: dict(note) for pair, note in self._notes.items()}

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
                timeout=25,
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
        if regime == "trend":
            return self.trend.desired_long(history, currently_long)
        if regime == "range":
            return self.mean_reversion.desired_long(history, currently_long)
        if regime in {"high_volatility", "unavailable"}:
            return False
        # Uncertain model labels cause no new entries and do not churn an open position.
        return currently_long

    def entry_strength(self, history: list[Candle]) -> float:
        regime = self._regime(history)
        if regime == "trend":
            return self.trend.entry_strength(history)
        if regime == "range":
            return self.mean_reversion.entry_strength(history)
        return 0.0
