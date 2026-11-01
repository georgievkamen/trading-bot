# 15-minute paper-run check (2026-11-01)

The scanner fetched 720 closed 15-minute candles (about 7.5 days) for the current top 30 USD crypto pairs. A same-window diagnostic used $1,000, 26 bps fees and 5 bps adverse slippage per side, 10% maximum marked-equity allocation per entry, and six concurrent positions:

| Baseline | Ending marked equity | Return | Max drawdown | Closed trades | Win rate | Profit factor |
|---|---:|---:|---:|---:|---:|---:|
| Trend (10/30 SMA) | $916.84 | -8.32% | 10.64% | 148 | 28.38% | 0.54 |
| Mean reversion (20 bar, -2 z entry) | $908.41 | -9.16% | 9.64% | 167 | 37.13% | 0.40 |

At scan time, the latest completed bars had nine trend entry candidates and one mean-reversion candidate. This indicates the rules are producing opportunities; it does not guarantee any future fill. The sample is only about a week on today's selected universe, so it is a weak and biased estimate. Both results are negative after the current cost assumptions; increasing paper size magnifies those losses. The Llama strategy is not represented in this historical table and needs separate forward evaluation.

## Cross-timeframe sanity check

The same simulator and 10%/six-position limits were also applied to the currently stored hourly and daily data. All figures use the same illustrative fees/slippage. The hourly file contains 720 bars (about 30 days) for the top 30; daily data contains only 13 pairs and spans different listing dates.

| Interval and sample | Baseline | Return | Max drawdown | Closed trades | Profit factor |
|---|---|---:|---:|---:|---:|
| 15-minute, about 7.5 days, 30 pairs | Trend | -8.32% | 10.64% | 148 | 0.54 |
| 15-minute, about 7.5 days, 30 pairs | Mean reversion | -9.16% | 9.64% | 167 | 0.40 |
| Hourly, about 30 days, 30 pairs | Trend | -25.16% | 26.34% | 162 | 0.25 |
| Hourly, about 30 days, 30 pairs | Mean reversion | -12.36% | 14.47% | 158 | 0.48 |
| Daily, roughly two years, 13 pairs | Trend | +95.23% | 37.18% | 117 | 2.04 |
| Daily, roughly two years, 13 pairs | Mean reversion | +30.78% | 20.38% | 103 | 1.70 |

The daily figures are not directly comparable to the intraday experiment and use a selected present-day universe. This large difference by timeframe is a warning that the current simple rules are regime-sensitive. No Llama/ML model, buy-and-hold comparator, point-in-time universe, or spread/market-impact model is included in these checks. Do not infer a tradable edge from them.
