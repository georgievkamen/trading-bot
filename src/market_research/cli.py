from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from statistics import median

from .kraken import discover_usd_spot_pairs, fetch_ohlc, is_crypto_usd_pair
from .dashboard import serve_dashboard
from .equities import (
    DEFAULT_STOCK_SYMBOLS,
    PAPER_ACCOUNT_ID as STOCK_PAPER_ACCOUNT_ID,
    INTRADAY_INTERVAL_MINUTES,
    DAILY_INTERVAL_MINUTES,
    STOCK_PAPER_STRATEGIES,
    activate_intraday_equity_paper,
    compare_equity_paper_account,
    extend_equity_paper_universe,
    refresh_equity_data,
    run_equity_backtests,
    start_equity_paper_account,
    stock_paper_account_id,
    watch_equity_paper_account,
)
from .local_model import LocalOllamaAdvisor
from .paper import PAPER_STRATEGY_NAMES, advance_paper_account, paper_summary, start_paper_account, watch_paper_account
from .reports import paper_period_comparison, paper_trade_attribution
from .simulator import Costs, Simulator, equal_weight_buy_and_hold
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
        description="Local multi-market research and paper simulation; it cannot place live orders.",
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

    backtest = sub.add_parser("backtest", help="Compare trend, mean-reversion, and filtered experiments")
    backtest.add_argument("--pairs", nargs="+")
    backtest.add_argument("--interval", type=int, default=1440)
    backtest.add_argument("--cash", type=float, default=1000)
    backtest.add_argument("--fee-bps", type=float, default=26)
    backtest.add_argument("--slippage-bps", type=float, default=5)
    backtest.add_argument("--max-position-fraction", type=float, default=0.10)
    backtest.add_argument("--max-positions", type=int, default=6)
    backtest.add_argument("--test-fraction", type=float, default=0.30,
                          help="Chronological out-of-sample fraction; strategy history before the split is warm-up only")

    paper = sub.add_parser("paper", help="Manage the persistent simulated account")
    paper_sub = paper.add_subparsers(dest="paper_action", required=True)
    start = paper_sub.add_parser("start", help="Create a simulated account only when no saved ledger exists")
    start.add_argument("--pairs", nargs="+", help="Optional subset; otherwise use all stored/scanned pairs")
    start.add_argument("--interval", type=int, default=15)
    start.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="trend")
    start.add_argument("--model", default=LocalOllamaAdvisor.DEFAULT_MODEL, help="Local Ollama model for llama_regime paper account")
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
    watch.add_argument(
        "--replay-missed-history", action="store_true",
        help="Explicitly replay candles missed while stopped, using historical opens",
    )
    status = paper_sub.add_parser("status", help="Show paper balance, P&L, positions, and recent trades")
    status.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="trend")
    analyze = paper_sub.add_parser("analyze", help="Attribute paper P&L to market movement, fees, and estimated slippage")
    analyze.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="llama_regime")
    compare = paper_sub.add_parser("compare", help="Compare a paper account with cash, buy-and-hold, and fixed strategies over its recorded period")
    compare.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES, default="llama_regime")
    reset = paper_sub.add_parser("reset", help="Clear one or both paper accounts")
    reset.add_argument("--strategy", choices=PAPER_STRATEGY_NAMES)

    stocks = sub.add_parser("stocks", help="Research and paper-trade US equities on daily or 15-minute bars")
    stocks.add_argument("--database", default="data/stocks_market_research.sqlite3", help="Separate stock research database")
    stock_sub = stocks.add_subparsers(dest="stocks_action", required=True)
    stock_fetch = stock_sub.add_parser("fetch", help="Fetch public Yahoo Finance bars and corporate actions")
    stock_fetch.add_argument("--symbols", nargs="+", default=list(DEFAULT_STOCK_SYMBOLS))
    stock_fetch.add_argument("--interval", default="1d", choices=["1d", "15m"])
    stock_fetch.add_argument("--period", choices=["1d", "5d", "10d", "1mo", "3mo", "1y", "2y", "5y", "10y", "max"])
    stock_backtest = stock_sub.add_parser("backtest", help="Compare daily stock-market baseline strategies")
    stock_backtest.add_argument("--symbols", nargs="+", default=list(DEFAULT_STOCK_SYMBOLS))
    stock_backtest.add_argument("--cash", type=float, default=1000)
    stock_backtest.add_argument("--fee-bps", type=float, default=0)
    stock_backtest.add_argument("--slippage-bps", type=float, default=5)
    stock_backtest.add_argument("--max-position-fraction", type=float, default=0.10)
    stock_backtest.add_argument("--max-positions", type=int, default=6)
    stock_backtest.add_argument("--test-fraction", type=float, default=0.30)
    stocks_paper = stock_sub.add_parser("paper", help="Manage the separate stock paper account")
    stocks_paper_sub = stocks_paper.add_subparsers(dest="stocks_paper_action", required=True)
    stocks_start = stocks_paper_sub.add_parser("start", help="Create the forward-only stock paper account")
    stocks_start.add_argument("--symbols", nargs="+", default=list(DEFAULT_STOCK_SYMBOLS))
    stocks_start.add_argument("--cash", type=float, default=1000)
    stocks_start.add_argument("--fee-bps", type=float, default=0)
    stocks_start.add_argument("--slippage-bps", type=float, default=5)
    stocks_start.add_argument("--max-position-fraction", type=float, default=0.10)
    stocks_start.add_argument("--max-positions", type=int, default=6)
    stocks_start.add_argument("--model", default=LocalOllamaAdvisor.DEFAULT_MODEL)
    stocks_start.add_argument("--strategy", choices=STOCK_PAPER_STRATEGIES, default="llama_regime")
    stocks_start.add_argument("--interval", type=int, default=DAILY_INTERVAL_MINUTES,
                              choices=[DAILY_INTERVAL_MINUTES, INTRADAY_INTERVAL_MINUTES])
    stocks_start.add_argument("--reset", action="store_true", help="Replace an existing stock paper account")
    stocks_paper_sub.add_parser(
        "intraday", help="Switch the existing stock paper ledger to 15-minute bars without resetting it",
    )
    stocks_watch = stocks_paper_sub.add_parser("watch", help="Watch completed 15-minute bars during US hours (or daily bars after close)")
    stocks_watch.add_argument("--poll-seconds", type=int, default=60)
    stocks_watch.add_argument("--strategy", choices=STOCK_PAPER_STRATEGIES, default="llama_regime")
    stocks_watch.add_argument("--once", action="store_true")
    stocks_status = stocks_paper_sub.add_parser("status", help="Show stock paper account P&L and positions")
    stocks_status.add_argument("--strategy", choices=STOCK_PAPER_STRATEGIES, default="llama_regime")
    stocks_reset = stocks_paper_sub.add_parser("reset", help="Clear only the stock paper account")
    stocks_reset.add_argument("--yes", action="store_true", help="Confirm clearing the separate stock account")
    stocks_reset.add_argument("--strategy", choices=STOCK_PAPER_STRATEGIES, default="llama_regime")
    stock_compare = stocks_paper_sub.add_parser("compare", help="Compare stock paper results with SPY and fixed baselines")
    stock_compare.add_argument("--strategy", choices=STOCK_PAPER_STRATEGIES, default="llama_regime")
    stocks_add = stocks_paper_sub.add_parser(
        "add-symbols", help="Add fetched tickers to the active account without resetting its history",
    )
    stocks_add.add_argument("--symbols", nargs="+", required=True)
    stocks_add.add_argument("--strategy", choices=STOCK_PAPER_STRATEGIES, default="llama_regime")

    regime = sub.add_parser("regime", help="Optional local Ollama market-regime annotation")
    regime.add_argument("pair")
    regime.add_argument("--interval", type=int, default=1440)
    regime.add_argument("--model", default=LocalOllamaAdvisor.DEFAULT_MODEL, help="Name of an installed local Ollama model (default: llama3.1:8b-instruct-q4_K_M)")

    learn = sub.add_parser("learn", help="Train and walk-forward evaluate the offline four-hour return model")
    learn.add_argument("--full-archive", default="data/kraken_ohlcvt_archive/Kraken_OHLCVT_Full.zip")
    learn.add_argument("--quarterly-archive", default="data/kraken_ohlcvt_archive/Kraken_OHLCVT_2026Q3.zip")
    learn.add_argument("--live-db", default="data/market_research.sqlite3", help="Include newly accumulated closed candles from the local paper database")
    learn.add_argument("--research-db", default="data/historical_learning.sqlite3")
    learn.add_argument("--report", default="data/historical_learning_report.json")
    learn.add_argument("--model-output", default="data/models/return_4h_hist_gradient.joblib")

    dashboard = sub.add_parser("dashboard", help="Open the local paper-trading dashboard")
    dashboard.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--stocks-db", default="data/stocks_market_research.sqlite3")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = MarketStore(args.database if args.command == "stocks" else args.db)
    try:
        if args.command == "learn":
            from .historical_learning import run_historical_learning

            report = run_historical_learning(
                full_archive=args.full_archive,
                quarterly_archive=args.quarterly_archive,
                live_db=args.live_db,
                research_db=args.research_db,
                report_path=args.report,
                model_path=args.model_output,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0

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
            if not 0.1 <= args.test_fraction <= 0.5:
                raise ValueError("Out-of-sample fraction must be between 0.10 and 0.50")
            simulator = Simulator(
                initial_cash=args.cash,
                costs=Costs(args.fee_bps, args.slippage_bps),
                max_position_fraction=args.max_position_fraction,
                max_positions=args.max_positions,
            )
            timestamps = sorted({bar.timestamp for rows in data.values() for bar in rows})
            split_index = int(len(timestamps) * (1 - args.test_fraction))
            if split_index < 1 or split_index >= len(timestamps):
                raise RuntimeError("Not enough distinct candle timestamps for a chronological split")
            oos_start = timestamps[split_index]
            training_end = timestamps[split_index - 1]

            def compact(result):
                row = result.to_dict()
                row.pop("trades", None)
                row.pop("equity_curve", None)
                return row

            results = []
            for strategy in STRATEGIES.values():
                training = simulator.run(data, strategy, through_timestamp=training_end)
                out_of_sample = simulator.run(data, strategy, from_timestamp=oos_start)
                results.append({
                    "strategy": strategy.name,
                    "training_period": compact(training),
                    "out_of_sample": compact(out_of_sample),
                })
            benchmark = equal_weight_buy_and_hold(
                data, args.cash, Costs(args.fee_bps, args.slippage_bps),
                from_timestamp=oos_start,
            )
            print(json.dumps({
                "market": "USD crypto",
                "interval_minutes": args.interval,
                "split": {
                    "training_end_utc": datetime.fromtimestamp(training_end, timezone.utc).isoformat(),
                    "out_of_sample_start_utc": datetime.fromtimestamp(oos_start, timezone.utc).isoformat(),
                    "out_of_sample_fraction": args.test_fraction,
                    "distinct_bars": len(timestamps),
                },
                "symbols": sorted(data),
                "initial_cash": args.cash,
                "costs_bps_per_side": {"fee": args.fee_bps, "slippage": args.slippage_bps},
                "limits": {"max_position_fraction": args.max_position_fraction, "max_positions": args.max_positions},
                "out_of_sample_benchmarks": [
                    {"strategy": "cash", "ending_equity": args.cash, "pnl": 0.0,
                     "return_pct": 0.0, "max_drawdown_pct": 0.0, "closed_trades": 0,
                     "win_rate_pct": None, "profit_factor": None, "open_positions": 0},
                    benchmark,
                ],
                "results": results,
                "notes": [
                    "The chronological test interval is untouched by the simulation; earlier candles are warm-up only.",
                    "Strategy parameters are fixed in code and were not selected using these out-of-sample results.",
                    "Equal-weight buy-and-hold invests all eligible symbols at their first test-period open and pays the same entry fee and slippage.",
                    "The universe is the currently stored set, so the crypto results have selection and survivorship bias.",
                ],
            }, indent=2, allow_nan=False))
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
                    watch_paper_account(
                        store, args.strategy, args.poll_seconds, args.once,
                        replay_missed_history=args.replay_missed_history,
                    )
                except KeyboardInterrupt:
                    print("Paper watch stopped.")
            elif args.paper_action == "status":
                state = store.get_paper_state(args.strategy)
                if not state:
                    raise RuntimeError(f"No {args.strategy} paper account exists. Run paper start first.")
                print(json.dumps(paper_summary(state), indent=2, allow_nan=False))
            elif args.paper_action == "analyze":
                state = store.get_paper_state(args.strategy)
                if not state:
                    raise RuntimeError(f"No {args.strategy} paper account exists. Run paper start first.")
                print(json.dumps(paper_trade_attribution(state), indent=2, allow_nan=False))
            elif args.paper_action == "compare":
                print(json.dumps(paper_period_comparison(store, args.strategy), indent=2, allow_nan=False))
            else:
                store.clear_paper_state(args.strategy)
                print("Paper account(s) cleared.")
            return 0

        if args.command == "stocks":
            if args.stocks_action == "fetch":
                interval_minutes = INTRADAY_INTERVAL_MINUTES if args.interval == "15m" else DAILY_INTERVAL_MINUTES
                period = args.period or ("10d" if args.interval == "15m" else "5y")
                print(json.dumps(refresh_equity_data(
                    store, args.symbols, period=period, interval_minutes=interval_minutes,
                ), indent=2, allow_nan=False))
            elif args.stocks_action == "backtest":
                report = run_equity_backtests(
                    store, args.symbols, initial_cash=args.cash,
                    costs=Costs(args.fee_bps, args.slippage_bps),
                    max_position_fraction=args.max_position_fraction,
                    max_positions=args.max_positions, test_fraction=args.test_fraction,
                )
                print(json.dumps(report, indent=2, allow_nan=False))
            elif args.stocks_action == "paper":
                if args.stocks_paper_action == "start":
                    state = start_equity_paper_account(
                        store, symbols=args.symbols, starting_cash=args.cash,
                        costs=Costs(args.fee_bps, args.slippage_bps),
                        max_position_fraction=args.max_position_fraction,
                        max_positions=args.max_positions, model=args.model, reset=args.reset,
                        interval_minutes=args.interval,
                        strategy_name=args.strategy,
                    )
                    print(json.dumps(paper_summary(state), indent=2, allow_nan=False))
                elif args.stocks_paper_action == "intraday":
                    print(json.dumps(activate_intraday_equity_paper(store), indent=2, allow_nan=False))
                elif args.stocks_paper_action == "watch":
                    try:
                        watch_equity_paper_account(store, args.poll_seconds, args.once, args.strategy)
                    except KeyboardInterrupt:
                        print("Stock paper watch stopped.")
                elif args.stocks_paper_action == "status":
                    state = store.get_paper_state(stock_paper_account_id(args.strategy))
                    if not state:
                        raise RuntimeError("No stock paper account exists. Run `stocks paper start` first.")
                    print(json.dumps(paper_summary(state), indent=2, allow_nan=False))
                elif args.stocks_paper_action == "compare":
                    print(json.dumps(compare_equity_paper_account(store, args.strategy), indent=2, allow_nan=False))
                elif args.stocks_paper_action == "add-symbols":
                    print(json.dumps(extend_equity_paper_universe(store, args.symbols, args.strategy), indent=2, allow_nan=False))
                else:
                    if not args.yes:
                        raise ValueError("Pass `--yes` to confirm clearing the stock paper account")
                    store.clear_paper_state(stock_paper_account_id(args.strategy))
                    print("Stock paper account cleared.")
            return 0

        if args.command == "regime":
            bars = store.get_candles(args.interval, [args.pair]).get(args.pair.upper(), [])
            if not bars:
                raise RuntimeError("No stored candles for pair. Run fetch first.")
            note = LocalOllamaAdvisor(args.model).classify(args.pair, [bar.close for bar in bars])
            print(json.dumps(asdict(note), indent=2, allow_nan=False))
            return 0

        if args.command == "dashboard":
            serve_dashboard(args.host, args.port, args.db, args.stocks_db)
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

