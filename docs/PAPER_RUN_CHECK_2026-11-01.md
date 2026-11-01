# 15-minute paper-run check (2026-11-01)

The scanner fetched 720 closed 15-minute candles (about 7.5 days) for the current top 30 USD crypto pairs. A same-window diagnostic used $1,000, 26 bps fees and 5 bps adverse slippage per side, 10% maximum marked-equity allocation per entry, and six concurrent positions:

| Baseline | Ending marked equity | Return | Max drawdown | Closed trades | Win rate | Profit factor |
|---|---:|---:|---:|---:|---:|---:|
| Trend (10/30 SMA) | $916.84 | -8.32% | 10.64% | 148 | 28.38% | 0.54 |
| Mean reversion (20 bar, -2 z entry) | $908.41 | -9.16% | 9.64% | 167 | 37.13% | 0.40 |

At scan time, the latest completed bars had nine trend entry candidates and one mean-reversion candidate. This indicates the rules are producing opportunities; it does not guarantee any future fill. The sample is only about a week on today's selected universe, so it is a weak and biased estimate. Both results are negative after the current cost assumptions; increasing paper size magnifies those losses. The Llama strategy is not represented in this historical table and needs separate forward evaluation.
