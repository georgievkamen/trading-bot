# New strategy evaluation — 2026-11-10

## Scope and method

This report compares two new deterministic, long-only candidates with the existing trend and mean-reversion strategies. Every result is simulated from $1,000, uses completed bars for signals, fills at the next bar open with adverse slippage, and marks open positions to the final available close. There are no live orders, leverage, shorts, or API trading credentials.

- **Crypto universe:** the 10 stored USD pairs for daily bars and 30 pairs for 15-minute and hourly bars. Out-of-sample bars run from 2026-11-06 01:15 UTC through 2026-11-10 19:00 UTC (15m), 2026-10-30 00:00 through 2026-11-10 18:00 UTC (1h), and 2026-04-05 through 2026-11-09 UTC (daily).
- **Stocks:** fixed 11-symbol universe in `DEFAULT_STOCK_SYMBOLS`, daily bars from 2021-11-01 through 2026-11-09 UTC. The chronological test contains 379 sessions, from 2025-05-08 to 2026-11-09. A ticker joins when its stored history begins; the newest or shortest-history ticker no longer shortens every strategy's sample.
- **Split:** chronological 70% earlier history / 30% later test by default. The account starts in cash at the test boundary; prior bars are feature warm-up only.
- **Costs:** crypto 26 bps fee plus 5 bps adverse slippage per side. Stocks 0 bps commission plus 5 bps adverse slippage per side. These are assumptions, not a broker or exchange quote.
- **Risk:** maximum 10% of marked equity per position and six concurrent positions (up to 60% gross exposure). No strategy changes those limits.
- **Benchmarks:** cash and equal-weight buy-and-hold for each universe; stocks also include SPY buy-and-hold. Benchmarks pay the same entry costs but remain invested, so they do not share the strategies' six-position limit.

Profit factor, win rate, and trade counts use **closed trades only**. Total return and drawdown include open positions marked to the final close. A dash means a statistic is undefined because the benchmark does not close trades or the strategy made none.

## New strategy rules

1. **Volatility-adjusted breakout:** enter after a close breaks the prior 20-bar high, short ATR expands relative to 20-bar ATR, and the 20/60-bar trend confirms upward movement. Exit on a three-ATR trailing stop, fading momentum, or a volatility-ceiling breach.
2. **Weekly relative momentum:** rank positive 60-bar returns by return divided by realized volatility; target up to six leaders, rebalance weekly, and exit a holding if it breaches the volatility ceiling. If no assets qualify, hold cash.

The rules and thresholds were fixed before the final results run. Volatility ceilings vary with market and bar interval: 15-minute crypto 1.5%, hourly 4%, daily crypto 8%; 15-minute stocks 1% and daily stocks 2.5% realized volatility. Breakout ATR/price ceilings are 1.5%, 4%, 8%, 1%, and 3%, respectively. These are coarse risk screens, not optimal values.

## Crypto out-of-sample results

| Interval | Strategy | Return | Max drawdown | Profit factor | Closed trades | Win rate |
|---|---|---:|---:|---:|---:|---:|
| 15m | Cash | 0.00% | 0.00% | — | 0 | — |
| 15m | Equal-weight buy-and-hold | -8.15% | 13.25% | — | 0 | — |
| 15m | Trend | -18.95% | 21.42% | 0.17 | 129 | 14.0% |
| 15m | Mean reversion | -9.93% | 12.44% | 0.28 | 128 | 26.6% |
| 15m | Filtered mean reversion | +0.65% | 0.78% | 1.87 | 10 | 10.0% |
| 15m | Volatility breakout | -5.75% | 6.08% | 0.19 | 59 | 16.9% |
| 15m | Relative momentum | 0.00% | 0.00% | — | 0 | — |
| 1h | Cash | 0.00% | 0.00% | — | 0 | — |
| 1h | Equal-weight buy-and-hold | -8.74% | 14.55% | — | 0 | — |
| 1h | Trend | -11.94% | 15.72% | 0.41 | 80 | 21.3% |
| 1h | Mean reversion | -10.99% | 12.92% | 0.27 | 55 | 34.5% |
| 1h | Filtered mean reversion | -2.43% | 2.50% | 0.00 | 14 | 7.1% |
| 1h | Volatility breakout | -3.89% | 5.02% | 0.32 | 31 | 22.6% |
| 1h | Relative momentum | -3.19% | 6.91% | 0.35 | 8 | 25.0% |
| Daily | Cash | 0.00% | 0.00% | — | 0 | — |
| Daily | Equal-weight buy-and-hold | +84.48% | 25.67% | — | 0 | — |
| Daily | Trend | +46.62% | 14.35% | 3.77 | 38 | 52.6% |
| Daily | Mean reversion | -0.85% | 9.73% | 1.43 | 27 | 59.3% |
| Daily | Filtered mean reversion | +1.28% | 1.48% | 2.15 | 5 | 40.0% |
| Daily | Volatility breakout | +39.28% | 5.47% | 3.91 | 31 | 51.6% |
| Daily | Relative momentum | +23.10% | 18.30% | 2.42 | 28 | 53.6% |

The crypto 15-minute and hourly results cover only about five and twelve out-of-sample days, respectively. They are too short to support a strategy conclusion. Daily breakout was positive, but substantially behind equal-weight buy-and-hold in this test. None of these results supports a claim of durable profitability.

## Stocks daily out-of-sample results

| Strategy | Return | Max drawdown | Profit factor | Closed trades | Win rate |
|---|---:|---:|---:|---:|---:|
| Cash | 0.00% | 0.00% | — | 0 | — |
| Equal-weight buy-and-hold | +50.39% | 22.24% | — | 0 | — |
| SPY buy-and-hold | +35.07% | 8.80% | — | 0 | — |
| Trend | +33.32% | 10.87% | 2.28 | 56 | 48.2% |
| Mean reversion | +10.91% | 5.61% | 3.69 | 54 | 77.8% |
| Filtered mean reversion | +0.12% | 1.14% | 1.09 | 9 | 66.7% |
| Volatility breakout | -1.23% | 4.45% | 0.77 | 41 | 36.6% |
| Relative momentum | +11.58% | 4.95% | 2.42 | 39 | 53.8% |

Relative momentum beat cash but lagged both equal-weight buy-and-hold and SPY. Breakout lost money. Neither new strategy demonstrated stock-market outperformance. The fixed universe also has survivorship bias; the full 11-stock equal-weight benchmark is not subject to the strategies' six-position cap.

## Volatility observed in the latest stock bars

The latest completed local stock bar is 2026-11-09. Current indicators show why the stock risk screen is needed:

| Symbol | 20-bar ATR / price | 60-bar realized daily volatility | New breakout ceiling | New momentum ceiling |
|---|---:|---:|---:|---:|
| INTC | 10.52% | 5.63% | 3.0% | 2.5% |
| SPCX | 5.95% | 3.69% | 3.0% | 2.5% |
| TSLA | 4.52% | 3.00% | 3.0% | 2.5% |
| NVDA | 4.55% | 2.56% | 3.0% | 2.5% |
| SPY | 1.45% | 0.92% | 3.0% | 2.5% |

The screens disqualify the first four names from new breakout/momentum selections at those readings; if an existing direct-strategy position crosses its ceiling, the simulator schedules an exit at the next bar open. This is a per-symbol filter, not a market-wide prediction. Llama remains a bounded regime selector and can independently choose cash for high-volatility conditions.

## Limitations and next validation

- The test split is chronological, but this final report should be treated as exploratory rather than a pristine untouched holdout: development reviewed preliminary results and the latest volatility readings before setting the final risk ceilings. Do not tune these ceilings against this report.
- The fixed stock and current crypto universes introduce survivorship/selection bias. Historical point-in-time membership is not available.
- Short crypto intraday samples are especially weak evidence. More history is required before any meaningful comparison.
- Historical fills model next-open prices with fixed slippage; they do not model queue position, partial fills, market impact, exchange outages, taxes, or all venue costs. Stock bars and corporate actions are sourced from Yahoo Finance through `yfinance` and may be delayed or revised.
- Win rate and profit factor do not include unrealized outcomes on positions still open at the test end.
- Llama only classifies regimes and selects from the approved rules or cash. It does not write rules, change position sizes, alter risk limits, or train its weights. No live execution path exists.

Keep the new rules in paper mode. Start a genuinely forward-only observation period with frozen parameters, preserve the saved ledger, and collect several market regimes before considering whether a strategy merits further testing. These results do not predict future profits.
