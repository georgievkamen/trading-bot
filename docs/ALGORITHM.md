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

The strategy is long only; no margin, leverage, or shorting.

## Time and fill model

For historical backtests and replay, a signal at candle timestamp `t` is calculated using bars strictly before `t`. Any change is filled at the open of bar `t`, with adverse slippage. The candle close is used only to mark equity after the simulated fill. This prevents same-close lookahead.

Entry fill = `open × (1 + slippage_bps / 10,000)`.

Exit fill = `open × (1 - slippage_bps / 10,000)`.

Fees are charged on both entry and exit. Entry spend includes the entry fee; exit proceeds are net of the exit fee. Realized P&L is net exit proceeds minus total entry spend.

The live paper watcher waits for a newly completed signal bar, then uses the current forming candle's public last price as its execution-price snapshot and applies adverse slippage and fees. It does not use the completed signal bar's historical open as a fill. If it cannot fetch a current price for a pair, it skips that pair's action for the latest event; marks refresh on every poll.

## Portfolio controls

- Starting virtual cash: $1,000.
- Historical backtests default to a maximum position size of 25% of marked equity per instrument; forward paper accounts default to 10% per entry and six concurrent positions, up to 60% of marked equity deployed.
- Position allocation is an exposure limit, not a maximum-loss guarantee.
- No borrowing: buys cannot exceed available cash.
- For simultaneous eligible entries, rank signal strength descending, then pair name for deterministic tie-breaking.
- Use an illustrative fee assumption of 26 basis points and slippage assumption of 5 basis points per side; both are CLI-configurable and must be calibrated to the venue and order style.
- A portfolio can remain partly or fully in cash.
- The active paper setup uses 15-minute bars; `paper watch` polls the read-only market-data endpoint every five minutes and simulates each newly closed candle. A paper account starts at the newest stored candle by default; `--replay-history` opts into replaying stored history.

## Model role

The core decision model is deterministic strategy code plus a deterministic simulator. A local language model classifies a series as `trend`, `range`, `high_volatility`, or `unknown`; the label selects between the two deterministic baselines or abstains. Low-confidence or unknown labels block new entries; high-volatility labels flatten simulated exposure. The model has no authority over sizing, portfolio caps, or fills, and has no exchange credentials or order interface. Measure its contribution through an ablation: compare the same data and costs with the regime switch and each fixed baseline.

## Interpretation

Compare both strategies on the same symbols, time interval, cash, fees, and slippage. Report total return, marked ending equity, maximum drawdown, closed trade count, win rate, and profit factor. A short sample, current top-turnover universe, and simplified fills are not enough to infer a durable edge. Add out-of-sample periods and cost sensitivity before using the result to make further project decisions.
