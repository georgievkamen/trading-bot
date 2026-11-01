from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Candle


BASE_URL = "https://api.kraken.com/0/public/OHLC"
PUBLIC_BASE = "https://api.kraken.com/0/public"

# A USD quote filter alone also includes fiat FX pairs and USD-pegged tokens.
# They do not belong in the first crypto strategy comparison.
NON_CRYPTO_BASES = {
    "USD", "EUR", "GBP", "CAD", "AUD", "CHF", "JPY",
    "USDT", "USDC", "DAI", "TUSD", "PYUSD", "FDUSD", "BUSD", "USDP",
    "UST", "USTC", "USDE", "USDS", "USDG", "USD1", "GUSD", "RLUSD",
    "EURC", "EURT",
}


def is_crypto_usd_pair(pair: str) -> bool:
    """Return whether a normalized pair is in the first USD crypto universe."""
    parts = pair.upper().split("/")
    return len(parts) == 2 and parts[1] == "USD" and parts[0] not in NON_CRYPTO_BASES


def _public_json(endpoint: str, params: dict[str, object] | None = None) -> dict[str, object]:
    query = urlencode(params or {})
    url = f"{PUBLIC_BASE}/{endpoint}" + (f"?{query}" if query else "")
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "AutonomousMarketResearch/0.1"})
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    errors = payload.get("error", [])
    if errors:
        raise RuntimeError("Kraken returned: " + "; ".join(errors))
    return payload.get("result", {})


def discover_usd_spot_pairs(limit: int = 10) -> list[dict[str, object]]:
    """Discover online Kraken USD-quoted spot pairs, ranked by 24h dollar turnover."""
    if limit < 1:
        raise ValueError("Universe limit must be positive")
    pairs = _public_json("AssetPairs", {"assetVersion": 1})
    tickers = _public_json("Ticker", {"assetVersion": 1})
    ranked: list[dict[str, object]] = []
    for pair, meta in pairs.items():
        if not isinstance(meta, dict) or meta.get("status") != "online":
            continue
        if meta.get("quote") != "USD" or meta.get("aclass_base") != "currency":
            continue
        if str(meta.get("base", "")).upper() in NON_CRYPTO_BASES:
            continue
        ticker = tickers.get(pair)
        if not isinstance(ticker, dict):
            continue
        try:
            volume_24h = float(ticker["v"][1])
            vwap_24h = float(ticker["p"][1])
            bid = float(ticker["b"][0])
            ask = float(ticker["a"][0])
            turnover = volume_24h * vwap_24h
            spread_bps = (ask - bid) / ((ask + bid) / 2) * 10_000 if ask + bid else 0.0
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            continue
        if turnover <= 0 or ask < bid:
            continue
        ranked.append({"pair": pair, "turnover_24h_usd": turnover, "spread_bps": spread_bps})
    ranked.sort(key=lambda row: (-float(row["turnover_24h_usd"]), str(row["pair"])))
    return ranked[:limit]


def fetch_ohlc(
    pair: str, interval: int = 60, timeout: int = 20, include_current: bool = False,
) -> list[Candle]:
    """Fetch public Kraken candles; optionally include the still-forming latest bar."""
    query = urlencode({"pair": pair, "interval": interval, "assetVersion": 1})
    request = Request(
        f"{BASE_URL}?{query}",
        headers={"Accept": "application/json", "User-Agent": "AutonomousMarketResearch/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    errors = payload.get("error", [])
    if errors:
        raise RuntimeError("Kraken returned: " + "; ".join(errors))
    result = payload.get("result", {})
    pair_keys = [key for key in result if key != "last"]
    if not pair_keys:
        raise RuntimeError(f"No OHLC data returned for {pair}")
    raw_rows = result[pair_keys[0]]
    if len(raw_rows) < 2:
        raise RuntimeError(f"Not enough OHLC data returned for {pair}")

    # Historical scans and backtests discard the uncommitted bar. The live
    # paper watcher may read its current close as a public last-price snapshot.
    rows = raw_rows if include_current else raw_rows[:-1]
    return [
        Candle(
            pair=pair.upper(), interval=interval, timestamp=int(row[0]),
            open=float(row[1]), high=float(row[2]), low=float(row[3]),
            close=float(row[4]), volume=float(row[6]),
        )
        for row in rows
    ]
