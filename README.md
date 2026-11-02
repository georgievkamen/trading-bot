# Autonomous Market Research — Core Engine

This local research system compares two transparent crypto strategy baselines using public Kraken candles, adds a constrained local-AI regime experiment, and includes a dashboard for paper-account monitoring. All account balances, fills, and results are simulated.

## What is implemented

- **Automatic crypto universe scan:** finds Kraken's online USD quoted spot crypto markets, excludes common fiat and stablecoin bases, ranks by 24 hour dollar turnover, and fetches recent candles for the top set.
- **Local market data:** saves OHLCV candles to a SQLite database using only Python's standard library.
- **Strategy comparison:** runs trend following, mean reversion, and an isolated 50-bar trend-filtered mean-reversion experiment against the same data and cost assumptions.
- **Backtest mechanics:** decisions use completed candles only; fills occur at the next candle open; fees and adverse slippage are applied on both sides.
- **Risk limits:** cash only, no leverage or shorting, maximum simulated position fraction, and maximum concurrent positions.
- **Persistent paper ledger:** tracks virtual cash, marked equity, realized and unrealized P&L, holdings, and closed trades. It can replay candles one step at a time.
- **AI-assisted paper experiment:** the `llama_regime` account asks local Llama to classify candidate coins' 15-minute market regime, then selects the trend baseline, mean-reversion baseline, or stays flat. It can attach fresh CoinDesk RSS headlines as unverified context; headlines are not fact-checked and cannot create a trade signal by themselves. Matured four-hour outcomes are stored and summarized as feedback only after at least 25 distinct four-hour windows per regime; same-window coin outcomes are aggregated and do not count as independent samples. This memory does not fine-tune Llama weights or change risk limits.
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

Rank stored instruments and compare the daily strategy rules, including the experimental trend filter:

```powershell
& $python -m market_research universe --interval 1440
& $python -m market_research backtest --interval 1440 --cash 1000
```

The selected active paper experiment uses **Llama only** on 15-minute bars. It checks Kraken every five minutes, starts from the latest stored candle, and can allocate up to 10% of marked account equity per entry with at most six open positions. Start the $1,000 virtual account once, then run its watcher:

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

The separate trend and mean-reversion accounts are optional controls; they are not part of the selected active run. Their saved history can be retained without running their watchers. `paper start` resets only the named strategy's account, so do not repeat it when you mean to resume an existing account.

`paper watch` refreshes public candles and automatically simulates entries/exits when a new live candle snapshot is available. It fetches markets concurrently, records each fetch time, and only combines prices from the same candle bucket fetched within 30 seconds of one another; older snapshots are excluded and reported. Forward watcher fills use the current public last price with adverse slippage, not a completed bar's historical open. If the watcher was stopped, the default mode skips missed candle events and resumes at current public prices, recording the gap in the account state. `--replay-missed-history` explicitly opts into historical catch-up fills and is intended only for deliberate replay experiments. The dashboard reports successful, failed, and snapshot-excluded markets separately; a partial feed error no longer marks the whole watcher offline. The Llama account batches eligible markets in groups of three to reduce sequential CPU inference time; it uses Llama to select a deterministic baseline, while risk limits and simulated fills remain in the engine. This improves cadence, not expected returns. No live order route exists. `paper step` manually advances one stored bar using next-bar-open replay. `paper reset` clears all accounts; add `--strategy` to clear one. The local database and paper balances are saved under `data/` in this project folder.

## Local dashboard

With the paper watchers running, start the dashboard server in another PowerShell window:

```powershell
& $python -m market_research dashboard
```

Open `http://127.0.0.1:8765/` in your browser. It shows each experiment's equity, P&L, open positions, closed trades, feed heartbeat, and Llama regime notes. The page is served only on this computer and refreshes every 15 seconds. Stop the server with Ctrl+C.

The historical research default is one day (1,440 minutes), allowing Kraken's recent-candle window to cover more history. CoinDesk RSS headlines are cached for 15 minutes, limited to recent items, and attached to Llama's audit notes. RSS can be unavailable or incomplete; a headline is not independent confirmation of a claim. When no fresh news is available, Llama continues with candle features. The active paper experiment uses 15-minute bars and ranks the online USD crypto market before fetching the top 30 by 24-hour turnover. Kraken returns at most 720 bars, so 15-minute history covers about 7.5 days. For custom pairs, run `& $python -m market_research fetch --pairs BTC/USD ETH/USD --interval 1440` to fetch daily candles. `& $python -m market_research backtest --pairs BTC/USD ETH/USD --interval 1440` limits the comparison to those pairs.

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
- **Trend-filtered mean reversion (experimental):** apply the same mean-reversion entry and rolling-mean exit, but only open or keep positions while the latest completed close is above its 50-bar simple moving average. The filter is a separate paper candidate; its current historical sample remains negative on intraday bars and does not establish profitability. See `docs/STRATEGY_FILTER_EXPERIMENT_2026-11-01.md`.

The systems are deliberately simple baseline theories, not claims of profitability. Both are long only. For simultaneous entry signals, the engine ranks signal strength and uses cash subject to the configured limits. It may leave cash idle and hold no assets. Measure performance after fees and slippage against cash and buy-and-hold baselines across untouched chronological periods before considering any strategy change.

Defaults are a $1,000 virtual balance, 26 basis points of fee per side, and 5 basis points of slippage per side. Historical backtests default to a 25% maximum position; paper accounts default to 10% per entry and at most six concurrent positions (up to 60% of marked equity deployed if every slot is filled). Fee and slippage defaults are assumptions to adjust to the chosen venue and order style; they are not a guaranteed Kraken fee quote. Position allocation is not a guaranteed loss limit.

## Evaluation limits

Kraken's public OHLC API returns at most 720 recent entries and includes a current uncommitted candle; historical scans and backtests drop that final candle, while the live paper watcher uses its last price only as an execution snapshot. The auto scanner ranks today's online USD spot markets. A historical backtest on today's universe can have survivorship and selection bias; it cannot establish historical universe membership. It also does not yet model partial fills, queue position, market impact, downtime, taxes, token transfer restrictions, or all venue fee tiers.

Review `docs/ALGORITHM.md`, `docs/ARCHITECTURE.md`, `docs/LOCAL_AI.md`, `docs/INITIAL_CHECK.md`, `docs/PAPER_RUN_CHECK_2026-11-01.md`, `docs/LEARNING_PLAN.md`, and `docs/HISTORICAL_MODEL_RESULTS_2026-11-01.md` before changing the research rules. The paper dashboard is served locally at `http://127.0.0.1:8765/` while its server is running.
## Separate US equities paper experiment

Stocks use a separate database and paper account, independent of the crypto ledger. The fixed initial universe is SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, AMZN, INTC, TSLA, and SPCX (SpaceX), using daily bars. SpaceX has a shorter history than the other symbols, so it limits the common backtest period. Yahoo Finance data comes through the community-maintained yfinance package and can be delayed, rate limited, revised, or unavailable. A universe selected today creates survivorship bias in historical comparisons.

Install the data adapter and fetch five years of daily prices and corporate actions:

~~~powershell
& $python -m pip install -e ".[stocks]"
& $python -m market_research stocks fetch
& $python -m market_research stocks backtest
~~~

Historical comparisons include cash, equal-weight buy-and-hold, SPY buy-and-hold, trend following, mean reversion, and trend-filtered mean reversion. Illustrative costs default to 0 bps commission and 5 bps adverse slippage per side. Prices are split-adjusted by the provider; ex-date dividends are credited to held positions.

The separate $1,000 Llama paper account classifies completed daily price features only and selects between the fixed trend and mean-reversion rules. It does not fine-tune Llama or consume stock news.

~~~powershell
& $python -m market_research stocks paper start --cash 1000 --model llama3.1:8b-instruct-q4_K_M
& $python -m market_research stocks paper watch
& $python -m market_research stocks paper status
& $python -m market_research stocks paper compare
~~~

For an existing account, fetch new tickers, stop its watcher, run `stocks paper add-symbols --symbols INTC TSLA SPCX`, and restart the watcher. This extends the account without resetting cash or trade history.

The watcher polls after 4:15 p.m. New York time on weekdays, allowing the feed to finish after the regular 4 p.m. session close. Each after-close target applies to the following session and is simulated at that session's close. The watcher never replays historical fills; after downtime or missed sessions it discards stale targets and forms a fresh classification. The experiment is cash-only and long-only, caps entries at 10% of marked equity and six open positions, and has no broker credentials or live order route. Weekends and market holidays can pass without decisions.

The dashboard reads both crypto and stock databases. Stock paper results are descriptive research outcomes and do not establish a profitable strategy.

