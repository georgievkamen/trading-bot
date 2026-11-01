# Autonomous Market Research — Core Engine

This is the first implementation stage: research logic and simulation first, dashboard later. It compares two transparent crypto strategy baselines using public Kraken candles. All account balances, fills, and results are simulated.

## What is implemented

- **Automatic crypto universe scan:** finds Kraken's online USD quoted spot crypto markets, excludes common fiat and stablecoin bases, ranks by 24 hour dollar turnover, and fetches recent candles for the top set.
- **Local market data:** saves OHLCV candles to a SQLite database using only Python's standard library.
- **Strategy comparison:** runs trend following and long only mean reversion against the same data and cost assumptions.
- **Backtest mechanics:** decisions use completed candles only; fills occur at the next candle open; fees and adverse slippage are applied on both sides.
- **Risk limits:** cash only, no leverage or shorting, maximum simulated position fraction, and maximum concurrent positions.
- **Persistent paper ledger:** tracks virtual cash, marked equity, realized and unrealized P&L, holdings, and closed trades. It can replay candles one step at a time.
- **AI-assisted paper experiment:** a separate `llama_regime` account asks local Llama to classify each coin's hourly market regime, then selects the trend baseline, mean-reversion baseline, or stays flat. Confidence, model errors, trade size, and risk caps are handled conservatively.

There is no account connection, API key handling, live order endpoint, or real money execution code.

## Your machine and installation

The intended machine is a Windows desktop with 16 GB RAM and integrated Intel graphics. The core engine is CPU based and has no third party Python dependencies. Python 3.11 or newer is required.

Python 3.14.10, Ollama, and `llama3.2:3b` are installed on this machine. Open PowerShell in this project folder and run:

```powershell
$env:PYTHONPATH = "$PWD\src"
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'
& $python -m market_research --help
```

## First run

Scan the automatically discovered USD crypto market, then fetch recent candles for the 30 highest-turnover pairs:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python -m market_research scan --interval 1440 --limit 30
```

Rank stored instruments and compare the two daily strategy baselines:

```powershell
& $python -m market_research universe --interval 1440
& $python -m market_research backtest --interval 1440 --cash 1000
```

For forward paper trading, the active experiment uses hourly bars and a 5% maximum allocation per entry. It checks Kraken every five minutes, then makes a decision only after a new hourly candle closes. It starts from the latest stored candle; `--replay-history` opts into replaying the stored bars from the beginning. Start one separate $1,000 account per theory:

```powershell
& $python -m market_research paper start --interval 60 --cash 1000 --strategy trend --max-position-fraction 0.05
& $python -m market_research paper start --interval 60 --cash 1000 --strategy mean_reversion --max-position-fraction 0.05
& $python -m market_research paper watch --strategy trend --poll-seconds 300
& $python -m market_research paper watch --strategy mean_reversion --poll-seconds 300
& $python -m market_research paper status --strategy mean_reversion
```

After Ollama is running and `llama3.2:3b` is installed, start the AI-assisted account alongside the two controls:

```powershell
& $python -m market_research paper start --interval 60 --cash 1000 --strategy llama_regime --model llama3.2:3b --max-position-fraction 0.05
& $python -m market_research paper watch --strategy llama_regime --poll-seconds 300
& $python -m market_research paper status --strategy llama_regime
```

`paper watch` refreshes public candles and automatically simulates entries/exits whenever a new closed bar is available. It runs until stopped with Ctrl+C. The AI account uses Llama only to label market regime; the deterministic risk limits and simulated fills remain in control. No live order route exists. `paper step` manually advances one stored bar. `paper start` resets only that strategy's account. `paper reset` clears all accounts; add `--strategy` to clear one. The local database and paper balances are saved under `data/` in this project folder.

## Local dashboard

With the paper watchers running, start the dashboard server in another PowerShell window:

```powershell
& $python -m market_research dashboard
```

Open `http://127.0.0.1:8765/` in your browser. It shows each experiment's equity, P&L, open positions, closed trades, feed heartbeat, and Llama regime notes. The page is served only on this computer and refreshes every 15 seconds. Stop the server with Ctrl+C.

The historical research default is one day (1,440 minutes), allowing Kraken's recent-candle window to cover more history. The active paper experiment uses one-hour bars for more frequent decisions and ranks the full online USD crypto market before fetching the top 30 by 24-hour turnover. Kraken returns at most 720 bars, so hourly history covers about 30 days. For custom pairs, run `& $python -m market_research fetch --pairs BTC/USD ETH/USD --interval 1440` to fetch daily candles. `& $python -m market_research backtest --pairs BTC/USD ETH/USD --interval 1440` limits the comparison to those pairs.

## Initial strategy theories

- **Trend following:** hold long when the simple moving average of the previous 10 closed bars is above the average of the previous 30 closed bars; otherwise exit.
- **Mean reversion:** enter long when the previous close is at least 2 standard deviations below its 20 bar mean; exit when the price recovers to the rolling mean.

The systems are deliberately simple baseline theories, not claims of profitability. Both are long only. For simultaneous entry signals, the engine ranks signal strength and uses cash subject to the configured limits. It may leave cash idle and hold no assets.

Defaults are a $1,000 virtual balance, 26 basis points of fee per side, and 5 basis points of slippage per side. Historical backtests default to a 25% maximum position; paper accounts default to 5% per entry and at most four concurrent positions. Fee and slippage defaults are assumptions to adjust to the chosen venue and order style; they are not a guaranteed Kraken fee quote.

## Evaluation limits

Kraken's public OHLC API returns at most 720 recent entries and includes a current uncommitted candle; the fetcher drops that final candle. The auto scanner ranks today's online USD spot markets. A historical backtest on today's universe can have survivorship and selection bias; it cannot establish historical universe membership. It also does not yet model partial fills, queue position, market impact, downtime, taxes, token transfer restrictions, or all venue fee tiers.

Review `docs/ALGORITHM.md`, `docs/ARCHITECTURE.md`, `docs/LOCAL_AI.md`, and `docs/INITIAL_CHECK.md` before changing the research rules. The dashboard remains a later step after the engine's accounting and evaluation workflow are established.

