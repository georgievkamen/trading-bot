# Autonomous Market Research — Core Engine

This local research system compares deterministic crypto and stock strategies using public market candles, adds a constrained local-AI regime experiment, and includes a dashboard for paper-account monitoring. All account balances, fills, and results are simulated.

## What is implemented

- **Automatic crypto universe scan:** finds Kraken's online USD quoted spot crypto markets, excludes common fiat and stablecoin bases, ranks by 24 hour dollar turnover, and fetches recent candles for the top set.
- **Local market data:** saves OHLCV candles to a SQLite database using only Python's standard library.
- **Strategy comparison:** runs trend following and mean reversion alongside volatility-adjusted breakout and weekly relative-momentum strategies. Every historical comparison uses a chronological 70/30 split by default and reports cash and buy-and-hold references.
- **Backtest mechanics:** decisions use completed candles only; fills occur at the next candle open; fees and adverse slippage are applied on both sides.
- **Risk limits:** cash only, no leverage or shorting, maximum simulated position fraction, and maximum concurrent positions.
- **Persistent paper ledger:** tracks virtual cash, marked equity, realized and unrealized P&L, holdings, and closed trades. It can replay candles one step at a time.
- **AI-assisted paper experiment:** the `llama_regime` account asks local Llama to classify market conditions, then selects among approved deterministic rules or stays in cash. High-volatility and uncertain conditions can map to cash. Headlines are optional, unverified context; they cannot create a trade signal by themselves. The model cannot invent rules, alter sizing, or fine-tune itself.
- **Offline historical ML experiment:** an optional local scikit-learn model learns to score the next four-hour return after estimated costs. It is evaluated on later dates and is kept separate from paper decisions until evidence justifies a paper-only trial.

There is no account connection, API key handling, live order endpoint, or real money execution code.

## Your machine and installation

The intended machine is a Windows desktop with 16 GB RAM and integrated Intel graphics. The core engine is CPU based and has no third party Python dependencies. Python 3.11 or newer is required.

Python 3.14.10, Ollama, and `llama3.2:3b` are installed on this machine. Open PowerShell in this project folder and run:

```powershell
$env:PYTHONPATH = "$PWD\src"
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'
& $python -m market_research --help
```

The separate US equities experiment uses Yahoo Finance daily bars through the community `yfinance` package. Install its optional dependencies with:

```powershell
& $python -m pip install -e ".[stocks]"
```

To install the optional historical-model dependencies in the project environment:

```powershell
& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[learning]"
```

## First run

Scan the automatically discovered USD crypto market, then fetch recent candles for the 30 highest-turnover pairs:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python -m market_research scan --interval 1440 --limit 30
```

Rank stored instruments and compare the daily strategy rules on a separate 30% out-of-sample period:

```powershell
& $python -m market_research universe --interval 1440
& $python -m market_research backtest --interval 1440 --cash 1000 --test-fraction 0.30 --max-position-fraction 0.10 --max-positions 6
```

The crypto backtest accepts 15-minute, hourly, and daily bars. Crypto costs default to 26 bps fee plus 5 bps adverse slippage per side. For the fixed stock universe, refresh daily bars and run the matching comparison:

```powershell
& $python -m market_research backtest --interval 15 --test-fraction 0.30
& $python -m market_research backtest --interval 60 --test-fraction 0.30
& $python -m market_research stocks fetch --interval 1d --period 10d
& $python -m market_research stocks backtest --test-fraction 0.30
```

The deterministic rules are long only. Breakouts require a prior-high break, expanding ATR, and a rising trend; positions exit on an ATR trail, fading momentum, or a volatility spike. Relative momentum selects up to six positive-return leaders weekly, ranks them by return per unit of realized volatility, and drops holdings that breach the volatility ceiling. Thresholds scale by bar interval; daily stocks have a 3% ATR/price cap for breakouts and a 2.5% realized-volatility cap for momentum. The caps can keep the strategy in cash when individual stocks swing sharply. See [the algorithm](docs/ALGORITHM.md) and [the dated out-of-sample results](docs/NEW_STRATEGIES_RESULTS_2026-11-10.md).

To paper-trade a deterministic strategy, create its separate $1,000 virtual account once, then run its watcher and status command. Starting a named account does not change the Llama account:

```powershell
& $python -m market_research paper start --strategy volatility_breakout --interval 1440 --cash 1000 --max-position-fraction 0.10 --max-positions 6
& $python -m market_research paper watch --strategy volatility_breakout --poll-seconds 300
& $python -m market_research paper status --strategy volatility_breakout
```

The same breakout rules can run on daily and hourly crypto candles because the hourly trial has its own ledger ID. Crypto relative momentum has a separate daily ledger:

```powershell
& $python -m market_research paper start --strategy volatility_breakout_1h --interval 60 --cash 1000 --max-position-fraction 0.10 --max-positions 6
& $python -m market_research paper watch --strategy volatility_breakout_1h --poll-seconds 300
& $python -m market_research paper start --strategy relative_momentum --interval 1440 --cash 1000 --max-position-fraction 0.10 --max-positions 6
& $python -m market_research paper watch --strategy relative_momentum --poll-seconds 300
```

`paper start` refuses to replace any saved ledger. If the account already exists, resume it with `paper watch` and inspect it with `paper status`.

The Llama regime experiment uses 15-minute crypto bars and can run alongside the deterministic accounts. It checks Kraken every five minutes, starts from the latest stored candle, and can allocate up to 10% of marked account equity per entry with at most six open positions. Create its $1,000 virtual account only if it has no saved ledger, then run its watcher:

```powershell
& $python -m market_research paper start --interval 15 --cash 1000 --strategy llama_regime --model llama3.1:8b-instruct-q4_K_M --max-position-fraction 0.10 --max-positions 6
& $python -m market_research paper watch --strategy llama_regime --poll-seconds 300
& $python -m market_research paper status --strategy llama_regime
```

Review the running experiment without resetting or advancing its account:

```powershell
& $python -m market_research paper analyze --strategy llama_regime
& $python -m market_research paper compare --strategy llama_regime
```

`paper analyze` reconciles closed-trade P&L into estimated gross price movement, fees, and slippage, then estimates the cost of liquidating currently open positions at their saved marks. `paper compare` uses the account's recorded forward period, stored market universe, and configured costs/risk limits to compare its ledger with cash and the fixed strategy baselines. It also reports an equal-weight buy-and-hold reference. Llama's live snapshot fills and the baselines' historical next-bar-open fills differ, so this is a same-period reference rather than a controlled identical-fill test. The report is descriptive and cannot establish future profitability.

Each experiment has a separate paper ledger. `paper start` will not overwrite a saved crypto ledger; the stocks command also requires an explicit `--reset` to replace an active stock ledger.

`paper watch` refreshes public candles and automatically simulates entries/exits when a new live candle snapshot is available. It fetches markets concurrently, records each fetch time, and only combines prices from the same candle bucket fetched within 30 seconds of one another; older snapshots are excluded and reported. Forward watcher fills use the current public last price with adverse slippage, not a completed bar's historical open. If the watcher was stopped, the default mode skips missed candle events and resumes at current public prices, recording the gap in the account state. `--replay-missed-history` explicitly opts into historical catch-up fills and is intended only for deliberate replay experiments. The dashboard reports successful, failed, and snapshot-excluded markets separately; a partial feed error no longer marks the whole watcher offline. The Llama account batches eligible markets in groups of three to reduce sequential CPU inference time; it uses Llama to select a deterministic baseline, while risk limits and simulated fills remain in the engine. This improves cadence, not expected returns. No live order route exists. `paper step` manually advances one stored bar using next-bar-open replay. `paper reset` clears all accounts; add `--strategy` to clear one. The local database and paper balances are saved under `data/` in this project folder.

## Local dashboard

With the paper watchers running, start the dashboard server in another PowerShell window:

```powershell
& $python -m market_research dashboard
```

Open `http://127.0.0.1:8765/` in your browser. It shows each experiment's equity, P&L, open positions, closed trades, feed heartbeat, and Llama regime notes. The page is served only on this computer and refreshes every 15 seconds. Stop the server with Ctrl+C.

The historical research default is one day (1,440 minutes), allowing Kraken's recent-candle window to cover more history. CoinDesk RSS headlines are cached for 15 minutes, limited to recent items, and attached to Llama's audit notes. RSS can be unavailable or incomplete; a headline is not independent confirmation of a claim. When no fresh news is available, Llama continues with candle features. The active paper experiment uses 15-minute bars and ranks the online USD crypto market before fetching the top 30 by 24-hour turnover. Kraken returns at most 720 bars, so 15-minute history covers about 7.5 days. For custom pairs, run `& $python -m market_research fetch --pairs BTC/USD ETH/USD --interval 1440` to fetch daily candles. `& $python -m market_research backtest --pairs BTC/USD ETH/USD --interval 1440` limits the comparison to those pairs.

## Separate US equities experiment

The US equities research is isolated in `data/stocks_market_research.sqlite3`. Its historical comparison uses daily bars and a fixed universe of SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, AMZN, INTC, TSLA, and SPCX. Yahoo Finance through `yfinance` is a community-maintained public-data interface; bars can be delayed, unavailable, or changed, and the fixed universe has survivorship bias. A symbol with a shorter listing history joins only once its stored bars begin; it no longer truncates the entire evaluation period.

Fetch daily history and compare the deterministic strategies with cash, equal-weight buy-and-hold, and SPY buy-and-hold:

```powershell
& $python -m market_research stocks fetch
& $python -m market_research stocks backtest --test-fraction 0.30 --max-position-fraction 0.10 --max-positions 6
```

To paper-trade a new strategy on the fixed stock universe, use daily candles. Each direct strategy account is independent from `llama_regime`:

```powershell
& $python -m market_research stocks paper start --strategy relative_momentum --interval 1440 --cash 1000 --max-position-fraction 0.10 --max-positions 6
& $python -m market_research stocks paper watch --strategy relative_momentum
& $python -m market_research stocks paper status --strategy relative_momentum
```

The active forward paper account uses **completed 15-minute bars** during the U.S. regular session. Yahoo Finance intraday history is limited to recent data. The fetcher drops incomplete bars and bars outside the regular session. The watcher polls every 60 seconds while the U.S. session is open, but only evaluates when a new 15-minute bar is complete. A signal from one completed bar schedules a simulated fill at the following bar's open with adverse slippage. This is a paper price model, not an exchange execution simulation.

Llama classifies each symbol's recent price regime and selects one of the fixed long-only trend or mean-reversion rules. For additional context, the classifier receives recent ticker headlines from public Google News RSS. Headlines are unverified and may be irrelevant or maliciously worded; they are supplementary context, never proof of a claim or a standalone order instruction. If news is unavailable, price-only classification continues. Llama does not tune its weights. Risk controls remain in the simulator: cash only, no leverage or shorts, at most 10% of marked equity per entry, and no more than six concurrent positions.

Migrate the existing $1,000 daily paper ledger to intraday without resetting its balance or trade history, then run the watcher:

```powershell
& $python -m market_research stocks paper intraday
& $python -m market_research stocks paper watch --poll-seconds 60
& $python -m market_research stocks paper status
& $python -m market_research stocks paper compare
```

The migration anchors the cursor at the newest stored 15-minute bar and clears only unfilled daily targets, so old bars are not replayed as new trades. Classification needs at least 30 completed bars per symbol; the fetched recent history supplies this warm-up so the first live bar can be classified. A target is eligible only after its signal bar. If the watcher misses a bar or restarts after downtime, stale targets are skipped rather than backfilled. Outside regular hours, weekends, and holidays, the watcher stays alive and reports that it is waiting for the next session. Use `stocks fetch --interval 15m --period 1mo` to refresh recent intraday data manually. A paper result, whether positive or negative, is not evidence of a future profit.

The dashboard reads both the crypto database and `data/stocks_market_research.sqlite3`. It shows the stock account's 15-minute regime notes and headline links when the watcher has processed a completed bar. All equity results remain simulated; no broker credentials or real-money order path are used.

## Historical learning experiment

The optional learning package is installed in the local `.venv`. With Kraken's full historical archive and quarterly update under `data/kraken_ohlcvt_archive/`, run:

```powershell
& .\.venv\Scripts\python.exe -m market_research learn
```

This streams the 15-minute USD crypto files into a separate local research database, adds new closed candles accumulated in `data/market_research.sqlite3`, ranks the top 30 by trailing turnover at each historical decision time, and builds features from completed candles. The target enters at the next 15-minute open and exits four hours later, after the configured fees and adverse slippage. It compares the trained model with cash, BTC buy-and-hold, trend following, and mean reversion on chronological 2025–2026 test periods, then repeats the paper results with doubled illustrative costs. The report, research database, and fitted model stay under ignored `data/`.

Weight training remains an explicit offline command; the running Llama model does not fine-tune itself. Its paper account now stores each eligible regime label and resolves a next-four-hour long return after configured fees and slippage, using only outcomes available by the current simulated timestamp. It keeps no more than one sample per pair per four-hour window to reduce overlapping examples. Once a regime has at least 25 resolved samples, the average after-cost return and win rate are supplied as cautious context to later Llama classifications. This is outcome memory, not a weight update or a proof of edge. The separate gradient-boosting candidate remains an offline research artifact and is not connected to any account or order route. A positive backtest only qualifies a candidate for a later paper-only test; it cannot promise future profit.

## Initial strategy theories

- **Trend following:** hold long when the simple moving average of the previous 10 closed bars is above the average of the previous 30 closed bars; otherwise exit.
- **Mean reversion:** enter long when the previous close is at least 2 standard deviations below its 20 bar mean; exit when the price recovers to the rolling mean.
- **Trend-filtered mean reversion (experimental):** apply the same mean-reversion entry and rolling-mean exit, but only open or keep positions while the latest completed close is above its 50-bar simple moving average. Its short intraday results vary by timeframe and are too small to establish profitability. See `docs/STRATEGY_FILTER_EXPERIMENT_2026-11-01.md`.
- **Volatility-adjusted breakout (experimental):** require a 20-bar high break, expanding ATR, and an uptrend; exit on a three-ATR trail, momentum fade, or a volatility ceiling breach.
- **Weekly relative momentum (experimental):** hold up to six positive 60-bar leaders ranked by return per unit of realized volatility; rebalance weekly and move to cash when leaders breach the volatility ceiling.

These are deliberately simple baseline theories, not claims of profitability. All are long only. For simultaneous entry signals, the engine ranks signal strength and uses cash subject to the configured limits. It may leave cash idle and hold no assets. Measure performance after fees and slippage against cash and buy-and-hold baselines across chronological periods before considering any strategy change.

Defaults are a $1,000 virtual balance, 26 basis points of fee per side, and 5 basis points of slippage per side for crypto. Historical backtests and paper accounts default to 10% maximum equity per position and at most six concurrent positions (up to 60% of marked equity deployed if every slot is filled). Stock backtests default to 0 bps commission and 5 bps slippage per side. Fee and slippage defaults are assumptions to adjust to the chosen venue and order style; they are not a guaranteed fee quote. Position allocation is not a guaranteed loss limit.

## Evaluation limits

Kraken's public OHLC API returns at most 720 recent entries and includes a current uncommitted candle; historical scans and backtests drop that final candle, while the live paper watcher uses its last price only as an execution snapshot. The auto scanner ranks today's online USD spot markets. A historical backtest on today's universe can have survivorship and selection bias; it cannot establish historical universe membership. It also does not yet model partial fills, queue position, market impact, downtime, taxes, token transfer restrictions, or all venue fee tiers.

Review `docs/ALGORITHM.md`, `docs/ARCHITECTURE.md`, `docs/LOCAL_AI.md`, `docs/INITIAL_CHECK.md`, `docs/PAPER_RUN_CHECK_2026-11-01.md`, `docs/LEARNING_PLAN.md`, `docs/HISTORICAL_MODEL_RESULTS_2026-11-01.md`, and `docs/NEW_STRATEGIES_RESULTS_2026-11-10.md` before changing the research rules. The paper dashboard is served locally at `http://127.0.0.1:8765/` while its server is running.

