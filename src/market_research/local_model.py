from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import pstdev


@dataclass(frozen=True, slots=True)
class RegimeNote:
    regime: str
    confidence: float
    rationale: str
    model: str


class LocalOllamaAdvisor:
    """Optional read-only regime annotator. It cannot create signals or orders."""

    ALLOWED = {"trend", "range", "high_volatility", "unknown"}
    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def classify(self, pair: str, closes: list[float], timeout: int = 45) -> RegimeNote:
        recent = closes[-61:]
        returns = [((new / old) - 1) * 100 for old, new in zip(recent, recent[1:]) if old > 0]
        change_10 = (recent[-1] / recent[-11] - 1) * 100 if len(recent) >= 11 else 0.0
        change_30 = (recent[-1] / recent[-31] - 1) * 100 if len(recent) >= 31 else 0.0
        volatility_20 = pstdev(returns[-20:]) * (20 ** 0.5) if len(returns) >= 20 else 0.0
        up_bars = sum(value > 0 for value in returns[-20:])
        features = {
            "bars": len(recent),
            "change_10_bars_pct": round(change_10, 3),
            "change_30_bars_pct": round(change_30, 3),
            "realized_volatility_20_bars_pct": round(volatility_20, 3),
            "up_bars_last_20": up_bars,
            "down_bars_last_20": min(20, len(returns[-20:])) - up_bars,
        }
        prompt = (
            "Classify this market snapshot as trend, range, high_volatility, or unknown. "
            "Use only the supplied percentage and volatility features. Choose unknown if the evidence "
            "is mixed or insufficient. This is a paper-trading research label. Return only JSON with "
            "keys regime, confidence (0 to 1), rationale.\n"
            f"pair={pair}; interval_bars={json.dumps(features)}"
        )
        payload = json.dumps({
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": "You are a cautious quantitative market-regime classifier. Never invent data or provide free-form trade instructions. Return only the requested JSON."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("Local Ollama service is unavailable") from exc
        raw = result.get("message", {}).get("content", "")
        try:
            decoded = json.loads(raw)
            regime = decoded.get("regime", "unknown")
            confidence = float(decoded.get("confidence", 0))
            rationale = str(decoded.get("rationale", ""))[:500]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("Local model returned invalid regime JSON") from exc
        if regime not in self.ALLOWED or not 0 <= confidence <= 1:
            raise RuntimeError("Local model returned an invalid regime label or confidence")
        return RegimeNote(regime, confidence, rationale, self.model)
