# First experiment: algorithm and model boundary

## System objective

At each simulated candle, choose among a bounded, automatically discovered crypto universe; allow the portfolio to hold cash when no signal qualifies; and measure results after transaction costs. The output is a research result or paper ledger entry, never a live order.

## Candidate universe

1. Read Kraken's public market-pair catalog.
2. Keep online spot pairs quoted in USD.
3. Rank these pairs by 24-hour dollar turnover (`24h base volume × 24h VWAP`).
4. Fetch the configured number of top pairs, storing their closed OHLCV candles locally.
5. Rank stored candidates by recent median dollar volume before strategy evaluation.

The first cut uses current turnover to select today's candidate set. Historical backtests therefore have survivorship/selection bias; future work needs point-in-time universe membership before results can be interpreted as market-wide evidence.

## Baseline A — trend following

For an instrument with at least 30 completed bars, compute:

- `fast = mean(close[-10:])`
- `slow = mean(close[-30:])`

Target long when `fast > slow`; otherwise target flat. Entry strength for cross-asset tie-breaking is `max(0, fast / slow - 1)`.

## Baseline B — mean reversion

For an instrument with at least 20 completed bars, compute the rolling mean `m` and population standard deviation `s` for the previous 20 closes. If `s > 0`, compute `z = (last_close - m) / s`.

- When flat, target long if `z <= -2`.
- When long, remain long while the latest completed close is below the rolling mean; exit at/above the mean.
- Entry strength is `max(0, -2 - z)`, the amount beyond the entry threshold.

## Experiment C — trend-filtered mean reversion

This separate paper candidate applies the Baseline B entry and exit rules only while the latest completed close is above its 50-bar simple moving average. It exits when the close falls to or below that filter. The recent stored-data check reduced short-sample intraday losses but remained negative; see `STRATEGY_FILTER_EXPERIMENT_2026-11-01.md`. It is not a validated profit signal and does not replace the baseline or local AI account.

The strategy is long only; no margin, leverage, or shorting.

## Strategy D — volatility-adjusted breakout

Use only completed bars. Enter when the latest close breaks above the prior 20-bar high, the 5-bar ATR is at least 1.05 times the 20-bar ATR, the 20-bar average is above the 60-bar average, and the close is above the 60-bar average. The engine fills at the next bar's open. The 20-bar ATR/price ceiling adapts to market and bar interval: crypto 15-minute bars 1.5%, hourly bars 4%, and daily bars 8%; stock 15-minute bars 1% and daily bars 3%. The strategy also exits at the next open if ATR rises above its ceiling while a position is open.

While long, set a trailing stop at the highest high observed since the simulated entry minus 3 × 20-bar ATR. Also exit if the completed close falls below its 20-bar average. The entry timestamp is part of the position record, so the trailing peak begins at the actual simulated entry rather than using pre-entry highs. The ATR ceiling and expansion condition are fixed parameters, not Llama-generated rules.

## Strategy E — weekly relative momentum

At the start of each UTC week, use only bars completed before that boundary. For each market with 60 prior bars, calculate its 60-bar return and per-bar realized volatility. Keep markets with at least +1% absolute momentum and below the volatility ceiling: 15-minute crypto 1.5%, hourly 4%, daily crypto 8%, 15-minute stocks 1%, and daily stocks 2.5%. Rank eligible names by return divided by realized volatility, then target the top six. Recheck realized volatility while holding and exit at the next open if the ceiling is breached. If none pass, target cash. The paper engine still applies the account's 10%-per-entry and six-position limits; this strategy does not change allocation or risk controls.

These fixed values are an initial, declared experiment. They have not been optimized against the reported test windows. The stock daily caps are intentionally tighter because single-name equity volatility varies widely; they filter names such as INTC, TSLA, and SPCX when current ATR or realized volatility exceeds the ceiling. Relative ranking uses the currently stored universe, so the result inherits that universe's selection and survivorship bias.

## Time and fill model

For historical backtests and replay, a signal at candle timestamp `t` is calculated using bars strictly before `t`. Any change is filled at the open of bar `t`, with adverse slippage. The candle close is used only to mark equity after the simulated fill. This prevents same-close lookahead.

Entry fill = `open × (1 + slippage_bps / 10,000)`.

Exit fill = `open × (1 - slippage_bps / 10,000)`.

Fees are charged on both entry and exit. Entry spend includes the entry fee; exit proceeds are net of the exit fee. Realized P&L is net exit proceeds minus total entry spend.

The live paper watcher uses the current forming candle's public last price as its execution-price snapshot and applies adverse slippage and fees. Each snapshot records its actual fetch time. A poll processes only prices from the same candle bucket whose fetch times are within 30 seconds; out-of-bucket or older snapshots are excluded and counted. When the watcher resumes after downtime, it skips missed candle events and trades only from the newest live snapshot, recording the gap. `paper watch --replay-missed-history` explicitly enables historical catch-up fills for deliberate replay experiments. A missing current price skips that pair's action; marks refresh on every poll.

## Portfolio controls

- Starting virtual cash: $1,000.
- Historical and paper runs default to a maximum position size of 10% of marked equity per instrument and at most six concurrent positions, up to 60% of marked equity deployed.
- Position allocation is an exposure limit, not a maximum-loss guarantee.
- No borrowing: buys cannot exceed available cash.
- For simultaneous eligible entries, rank signal strength descending, then pair name for deterministic tie-breaking.
- Use an illustrative fee assumption of 26 basis points and slippage assumption of 5 basis points per side; both are CLI-configurable and must be calibrated to the venue and order style.
- A portfolio can remain partly or fully in cash.
- The active paper setup uses 15-minute bars; `paper watch` polls the read-only market-data endpoint every five minutes and uses forward-only snapshots by default. A paper account starts at the newest stored candle by default; `--replay-history` opts into replaying stored history at account creation, and `paper watch --replay-missed-history` explicitly opts into catch-up replay after the watcher has been stopped.

## Model role

The core decision model is deterministic strategy code plus a deterministic simulator. A local language model may classify a market as `trend`, `range`, `breakout`, `relative_momentum`, `cash`, `high_volatility`, or `unknown`. These labels map only to the approved fixed trend, mean-reversion, volatility-breakout, and relative-momentum rules, or to cash. The engine independently checks every strategy's entry and exit conditions. Low-confidence and unknown labels block new entries; `cash` and `high_volatility` flatten simulated exposure. The model cannot invent rules, size positions, change limits, or access an order endpoint. Measure its contribution with the same chronological test split as the deterministic rules.

## Interpretation

The `backtest` commands use a chronological 70/30 training and out-of-sample split by default. The earlier period warms up rolling features; the test portfolio starts in cash at the split. The stock split uses the full available market timeline, so shorter-history tickers join only after their first stored bar instead of shortening the test for every stock. Report total return, marked ending equity, maximum drawdown, closed trades, win rate, and profit factor after fees and slippage. Crypto defaults to 26 bps fee plus 5 bps slippage per side; the stock experiment uses 0 bps commission plus 5 bps slippage per side unless overridden. Compare against cash and equal-weight buy-and-hold; stock results also include SPY buy-and-hold. A positive short-period result does not establish a durable edge. The current crypto 15-minute and hourly test windows remain short compared with the roughly five-year stock sample; see the date-stamped report for their exact spans and results.
