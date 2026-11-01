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

At candle timestamp `t`, a signal is calculated using bars strictly before `t`. Any change is filled at the open of bar `t`, with adverse slippage. The candle close is used only to mark equity after the simulated fill. This prevents same-close lookahead.

Entry fill = `open × (1 + slippage_bps / 10,000)`.

Exit fill = `open × (1 - slippage_bps / 10,000)`.

Fees are charged on both entry and exit. Entry spend includes the entry fee; exit proceeds are net of the exit fee. Realized P&L is net exit proceeds minus total entry spend.

## Portfolio controls

- Starting virtual cash: $1,000.
- Historical backtests default to a maximum position size of 25% of marked equity per instrument; forward paper accounts default to 5% per entry to test smaller speculative allocations.
- Maximum concurrent positions: four.
- No borrowing: buys cannot exceed available cash.
- For simultaneous eligible entries, rank signal strength descending, then pair name for deterministic tie-breaking.
- Use an illustrative fee assumption of 26 basis points and slippage assumption of 5 basis points per side; both are CLI-configurable and must be calibrated to the venue and order style.
- A portfolio can remain partly or fully in cash.
- A paper account starts at the newest stored candle by default. `paper watch` polls the read-only market-data endpoint and simulates each new candle; `--replay-history` opts into replaying stored history.

## Model role

The core decision model is deterministic strategy code plus a deterministic simulator. An optional local language model can annotate a series as `trend`, `range`, `high_volatility`, or `unknown`. The LLM result is not currently consumed by strategy or allocation logic. It has no exchange credentials or order interface. The appropriate integration test later is an ablation: compare the exact same strategy, data, and cost assumptions with AI annotations disabled and enabled, then decide whether annotations add measurable research value.

## Interpretation

Compare both strategies on the same symbols, time interval, cash, fees, and slippage. Report total return, marked ending equity, maximum drawdown, closed trade count, win rate, and profit factor. A short sample, current top-turnover universe, and simplified fills are not enough to infer a durable edge. Add out-of-sample periods and cost sensitivity before using the result to make further project decisions.
