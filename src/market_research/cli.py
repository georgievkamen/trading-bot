from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from statistics import median

from .kraken import discover_usd_spot_pairs, fetch_ohlc, is_crypto_usd_pair
from .dashboard import serve_dashboard
from .local_model import LocalOllamaAdvisor
from .paper import PAPER_STRATEGY_NAMES, advance_paper_account, paper_summary, start_paper_account, watch_paper_account
from .simulator import Costs, Simulator
from .storage import MarketStore
from .strategies import STRATEGIES


def rank_universe(data: dict, min_bars: int = 30) -> list[dict[str, object]]:
    ranked = []
    for pair, bars in data.items():
        if not is_crypto_usd_pair(pair) or len(bars) < min_bars:
            continue
        dollar_volume = median([bar.close * bar.volume for bar in bars[-min(24, len(bars)):]])
        ranked.append({"pair": pair, "median_dollar_volume": dollar_volume, "bars": len(bars)})
    return sorted(ranked, key=lambda row: (-float(row["median_dollar_volume"]), str(row["pair"])))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market_research",
        description="Local crypto research and paper simulation; it cannot place live orders.",
    )
    parser.add_argument("--db", default="data/market_research.sqlite3", help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch public Kraken Spot candles (no credentials)")
    fetch.add_argument("--pairs", nargs="+", default=["BTC/USD", "ETH/USD", "SOL/USD"])
    fetch.add_argument("--interval", type=int, default=1440, choices=[1, 5, 15, 30, 60, 240, 1440, 10080, 21600])

    scan = sub.add_parser("scan", help="Discover liquid USD spot pairs and fetch recent candles")
    scan.add_argument("--interval", type=int, default=1440, choices=[1, 5, 15, 30, 60, 240, 1440, 10080, 21600])
    scan.add_argument("--limit", type=int, default=30, help="Maximum number of highest-turnover crypto pairs to fetch")

    universe = sub.add_parser("universe", help="Rank supplied pairs by recent dollar volume")
    universe.add_argument("--interval", type=int, default=1440)

    backtest = sub.add_parser("backtest", help="Compare trend and mean-reversion baselines")
    backtest.add_argument("--pairs", nargs="+")
    backtest.add_argument("--interval", type=int, default=1440)
    backtest.add_argument("--cash", type=float, default=1000)
    backtest.add_argument("--fee-bps", type=float, default=26)
    backtest.add_argument("--slippage-bps", type=float, default=5)
    backtest.add_argument("--max-position-fraction", type=float, default=0.25)
    backtest.add_argument("--max-positions", type=int, default=4)

    paper = sub.add_parser("paper", help="Manage the persistent simulated account")
    paper_sub = paper.add_subparsers(dest="paper_action", required=True)
    start = paper_sub.add_parser("start", help="Create/reset a simulated account and replay cursor")
    start.add_argument("--pairs", nargs="+", help="Optional subset; otherwise use all stored/scanned pairs")
    start.add_argument("--interval", type=int, default=15)
    start.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="trend")
    start.add_argument("--model", default="llama3.2:3b", help="Local Ollama model for llama_regime paper account")
    start.add_argument("--cash", type=float, default=1000)
    start.add_argument("--fee-bps", type=float, default=26)
    start.add_argument("--slippage-bps", type=float, default=5)
    start.add_argument("--max-position-fraction", type=float, default=0.10)
    start.add_argument("--max-positions", type=int, default=6)
    start.add_argument("--replay-history", action="store_true", help="Start from oldest stored candle instead of now")
    step = paper_sub.add_parser("step", help="Advance one synchronized closed candle in the replay")
    step.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="trend")
    watch = paper_sub.add_parser("watch", help="Automatically paper-trade new closed candles")
    watch.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="trend")
    watch.add_argument("--poll-seconds", type=int, default=300)
    watch.add_argument("--once", action="store_true", help="Fetch/process once, then exit")
    status = paper_sub.add_parser("status", help="Show paper balance, P&L, positions, and recent trades")
    status.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="trend")
    reset = paper_sub.add_parser("reset", help="Clear one or both paper accounts")
    reset.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES)

    regime = sub.add_parser("regime", help="Optional local Ollama market-regime annotation")
    regime.add_argument("pair")
    regime.add_argument("--interval", type=int, default=1440)
    regime.add_argument("--model", default="llama3.2:3b", help="Name of an installed local Ollama model (default: llama3.2:3b)")

    dashboard = sub.add_parser("dashboard", help="Open the local paper-trading dashboard")
    dashboard.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    dashboard.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = MarketStore(args.db)
    try:
        if args.command == "fetch":
            for pair in args.pairs:
                candles = fetch_ohlc(pair, args.interval)
                saved = store.save_candles(candles)
                print(f"{pair}: fetched {len(candles)} closed candles; saved/updated {saved}")
            return 0

        if args.command == "scan":
            universe = discover_usd_spot_pairs(args.limit)
            fetched = []
            for item in universe:
                pair = str(item["pair"])
                candles = fetch_ohlc(pair, args.interval)
                store.save_candles(candles)
                fetched.append({**item, "closed_candles": len(candles)})
            print(json.dumps(fetched, indent=2, allow_nan=False))
            return 0

        if args.command == "universe":
            print(json.dumps(rank_universe(store.get_candles(args.interval)), indent=2))
            return 0

        if args.command == "backtest":
            data = store.get_candles(args.interval, args.pairs)
            if args.pairs:
                invalid = [pair for pair in args.pairs if not is_crypto_usd_pair(pair)]
                if invalid:
                    raise ValueError("V1 backtests only support USD-quoted crypto pairs; excluded: " + ", ".join(invalid))
            else:
                data = {pair: bars for pair, bars in data.items() if is_crypto_usd_pair(pair)}
            if not data:
                raise RuntimeError("No stored candles. Run fetch first.")
            simulator = Simulator(
                initial_cash=args.cash,
                costs=Costs(args.fee_bps, args.slippage_bps),
                max_position_fraction=args.max_position_fraction,
                max_positions=args.max_positions,
            )
            results = []
            for strategy in STRATEGIES.values():
                result = simulator.run(data, strategy).to_dict()
                result.pop("trades", None)
                result.pop("equity_curve", None)
                results.append(result)
            print(json.dumps(results, indent=2, allow_nan=False))
            return 0

        if args.command == "paper":
            if args.paper_action == "start":
                state = start_paper_account(
                    store, args.interval, args.pairs or [], args.strategy, args.cash,
                    Costs(args.fee_bps, args.slippage_bps),
                    max_position_fraction=args.max_position_fraction,
                    max_positions=args.max_positions,
                    replay_history=args.replay_history,
                    model=args.model,
                )
                print(json.dumps(paper_summary(state), indent=2, allow_nan=False))
            elif args.paper_action == "step":
                print(json.dumps(advance_paper_account(store, args.strategy), indent=2, allow_nan=False))
            elif args.paper_action == "watch":
                try:
                    watch_paper_account(store, args.strategy, args.poll_seconds, args.once)
                except KeyboardInterrupt:
                    print("Paper watch stopped.")
            elif args.paper_action == "status":
                state = store.get_paper_state(args.strategy)
                if not state:
                    raise RuntimeError(f"No {args.strategy} paper account exists. Run paper start first.")
                print(json.dumps(paper_summary(state), indent=2, allow_nan=False))
            else:
                store.clear_paper_state(args.strategy)
                print("Paper account(s) cleared.")
            return 0

        if args.command == "regime":
            bars = store.get_candles(args.interval, [args.pair]).get(args.pair.upper(), [])
            if not bars:
                raise RuntimeError("No stored candles for pair. Run fetch first.")
            note = LocalOllamaAdvisor(args.model).classify(args.pair, [bar.close for bar in bars])
            print(json.dumps(asdict(note), indent=2, allow_nan=False))
            return 0

        if args.command == "dashboard":
            serve_dashboard(args.host, args.port, args.db)
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
