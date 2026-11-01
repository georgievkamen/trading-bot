# Trend-filtered mean-reversion experiment (2026-11-01)

## Hypothesis

Mean reversion can buy a falling asset while its broader direction is still down. This candidate requires the latest completed close to be above its 50-bar simple moving average before it can open or keep a mean-reversion position. It otherwise uses the existing 20-bar, two-standard-deviation entry and rolling-mean exit.

The 50-bar filter was evaluated as a separate paper strategy. It does not replace the baseline or the Llama strategy.

## Exploratory result

The local database had 720 candles per recently scanned market: about 7.5 days at 15 minutes and 30 days at one hour. Daily data covered roughly two years across 13 pairs selected from today's universe. All runs used $1,000 virtual cash, 10% maximum equity allocation per position, six maximum positions, 26 bps fee and 5 bps adverse slippage per side.

| Interval | Strategy | Return | Maximum drawdown | Closed trades | Profit factor |
|---|---|---:|---:|---:|---:|
| 15-minute | Mean reversion | -11.98% | 12.19% | 171 | 0.392 |
| 15-minute | 50-bar trend-filtered mean reversion | -3.13% | 3.13% | 47 | 0.104 |
| 1-hour | Mean reversion | -12.36% | 14.47% | 158 | 0.476 |
| 1-hour | 50-bar trend-filtered mean reversion | -3.88% | 4.24% | 41 | 0.172 |
| Daily | Mean reversion | +30.78% | 20.38% | 103 | 1.698 |
| Daily | 50-bar trend-filtered mean reversion | +2.20% | 1.48% | 15 | 1.989 |

The filter reduced losses and drawdown on the short intraday samples, but those filtered samples still lost money and had profit factors below 1. The daily run had few trades and a present-day universe, so it is subject to substantial selection bias. This is a risk-reduction hypothesis for forward paper evaluation, not evidence of a profitable edge.

A separate downside cutoff at a rolling z-score of -3 was also explored and rejected: it made the 15-minute and one-hour results worse than the unfiltered baseline. That cutoff was not retained.

## Decision

Keep the candidate isolated as `mean_reversion_filtered` for paper comparison. Do not connect it to live execution, increase its size, or describe it as profitable. Forward paper results are a new evaluation period; the historical-learning model remains offline because it failed its earlier walk-forward gate.

