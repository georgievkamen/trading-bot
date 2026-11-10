from __future__ import annotations

import email.utils
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus, urlsplit


@dataclass(frozen=True, slots=True)
class NewsHeadline:
    source: str
    published_at: str
    title: str
    url: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class CryptoNewsFeed:
    """Small, cached RSS context source. Headlines are not independently verified."""

    FEED_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    CACHE_SECONDS = 900
    MAX_AGE = timedelta(hours=24)
    MAX_HEADLINES = 12
    MAX_TITLE_LENGTH = 240
    _ALIASES = {
        "BTC": ("bitcoin", "btc", "xbt"),
        "XBT": ("bitcoin", "btc", "xbt"),
        "ETH": ("ethereum", "ether", "eth"),
        "SOL": ("solana", "sol"),
        "XRP": ("xrp", "ripple"),
        "DOGE": ("dogecoin", "doge"),
        "ADA": ("cardano", "ada"),
        "AVAX": ("avalanche", "avax"),
        "LINK": ("chainlink", "link"),
        "DOT": ("polkadot", "dot"),
    }
    _BROAD_TERMS = (
        "crypto", "digital asset", "bitcoin", "ethereum", "sec", "etf",
        "exchange", "hack", "exploit", "liquidation", "fed", "interest rate",
        "inflation", "stablecoin", "regulation",
    )

    def __init__(self, timeout: float = 4.0) -> None:
        self.timeout = timeout
        self._cached_at = 0.0
        self._cached: tuple[NewsHeadline, ...] = ()
        self._last_status = "not_checked"
        self._last_error: str | None = None

    @property
    def status(self) -> str:
        return self._last_status

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def for_pair(self, pair: str) -> tuple[NewsHeadline, ...]:
        headlines = self._load()
        base = pair.upper().split("/", 1)[0].strip()
        aliases = self._ALIASES.get(base, (base.lower(),))
        selected = []
        for headline in headlines:
            text = headline.title.casefold()
            pair_match = any(
                re.search(rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])", text)
                for term in aliases
            )
            broad_match = any(term in text for term in self._BROAD_TERMS)
            if pair_match or broad_match:
                selected.append(headline)
            if len(selected) >= 5:
                break
        return tuple(selected)

    def _load(self) -> tuple[NewsHeadline, ...]:
        now = time.monotonic()
        if now - self._cached_at < self.CACHE_SECONDS:
            return self._cached
        request = urllib.request.Request(
            self.FEED_URL,
            headers={"User-Agent": "MarketResearchPaperBot/1.0 (local paper research)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                root = ET.fromstring(response.read(1_000_000))
        except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError, ValueError) as exc:
            self._cached_at = now
            self._last_status = "unavailable"
            self._last_error = str(exc)[:200]
            return self._cached

        current = datetime.now(timezone.utc)
        cutoff = current - self.MAX_AGE
        items: list[NewsHeadline] = []
        seen: set[str] = set()
        for item in root.findall(".//item"):
            title = " ".join((item.findtext("title") or "").split())[: self.MAX_TITLE_LENGTH]
            url = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if not title or not url or not pub_date or url in seen:
                continue
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or not (parsed.hostname == "coindesk.com" or parsed.hostname.endswith(".coindesk.com"))):
                continue
            try:
                published = email.utils.parsedate_to_datetime(pub_date)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                published = published.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
            if published < cutoff or published > current + timedelta(minutes=10):
                continue
            seen.add(url)
            items.append(NewsHeadline(
                source="CoinDesk",
                published_at=published.isoformat().replace("+00:00", "Z"),
                title=title,
                url=url,
            ))
            if len(items) >= self.MAX_HEADLINES:
                break
        self._cached = tuple(items)
        self._cached_at = now
        self._last_status = "available" if self._cached else "no_fresh_items"
        self._last_error = None
        return self._cached


class EquityNewsFeed:
    """Cached public ticker-news headlines for cautious, supplementary LLM context."""

    CACHE_SECONDS = 900
    MAX_AGE = timedelta(hours=24)
    MAX_HEADLINES = 3
    _NAMES = {
        "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF",
        "DIA": "Dow Jones ETF", "AAPL": "Apple", "MSFT": "Microsoft",
        "NVDA": "Nvidia", "AMZN": "Amazon", "INTC": "Intel",
        "TSLA": "Tesla", "SPCX": "SpaceX",
    }

    def __init__(self, timeout: float = 4.0) -> None:
        self.timeout = timeout
        self._cache: dict[str, tuple[float, tuple[NewsHeadline, ...], str, str | None]] = {}

    def for_symbol(self, symbol: str) -> tuple[NewsHeadline, ...]:
        return self.fetch_many([symbol]).get(symbol.upper(), ())

    def fetch_many(self, symbols: Sequence[str]) -> dict[str, tuple[NewsHeadline, ...]]:
        normalized = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip()))
        result: dict[str, tuple[NewsHeadline, ...]] = {}
        pending: list[str] = []
        now = time.monotonic()
        for symbol in normalized:
            cached = self._cache.get(symbol)
            if cached and now - cached[0] < self.CACHE_SECONDS:
                result[symbol] = cached[1]
            else:
                pending.append(symbol)
        if pending:
            with ThreadPoolExecutor(max_workers=min(8, len(pending))) as pool:
                futures = {pool.submit(self._load_symbol, symbol): symbol for symbol in pending}
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        items, status, error = future.result()
                    except Exception as exc:  # a feed failure must not stop the price-only path
                        items, status, error = (), "unavailable", str(exc)[:200]
                    self._cache[symbol] = (time.monotonic(), items, status, error)
                    result[symbol] = items
        return result

    def status_for(self, symbol: str) -> str:
        cached = self._cache.get(symbol.upper())
        return cached[2] if cached else "not_checked"

    def _load_symbol(self, symbol: str) -> tuple[tuple[NewsHeadline, ...], str, str | None]:
        subject = self._NAMES.get(symbol, symbol)
        query = quote_plus(f'"{subject}" stock when:1d')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        request = urllib.request.Request(
            url, headers={"User-Agent": "MarketResearchPaperBot/1.0 (local paper research)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                root = ET.fromstring(response.read(1_000_000))
        except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError, ValueError) as exc:
            return (), "unavailable", str(exc)[:200]

        current = datetime.now(timezone.utc)
        cutoff = current - self.MAX_AGE
        items: list[NewsHeadline] = []
        seen: set[str] = set()
        for item in root.findall(".//item"):
            title = " ".join((item.findtext("title") or "").split())[:240]
            item_url = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source = " ".join((item.findtext("source") or "Google News").split())[:80]
            if not title or not item_url or not pub_date or item_url in seen:
                continue
            parsed = urlsplit(item_url)
            if parsed.scheme != "https" or not parsed.hostname:
                continue
            try:
                published = email.utils.parsedate_to_datetime(pub_date)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                published = published.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
            if published < cutoff or published > current + timedelta(minutes=10):
                continue
            seen.add(item_url)
            items.append(NewsHeadline(
                source=source,
                published_at=published.isoformat().replace("+00:00", "Z"),
                title=title,
                url=item_url,
            ))
            if len(items) >= self.MAX_HEADLINES:
                break
        return tuple(items), "available" if items else "no_fresh_items", None
