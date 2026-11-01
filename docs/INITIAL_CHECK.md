# Initial BTC baseline check

This is a small exploratory calculation against Kraken's public BTC/USD daily OHLC data, covering 2026-08-01 through 2026-10-31 (92 completed daily bars). It uses $1,000 starting cash, 26 bps fee and 5 bps adverse slippage per side, 25% maximum equity allocation, long-only positions, decisions from completed bars, and fills at the next bar open. The current, uncommitted candle was excluded.

| Baseline | Ending marked equity | Net P&L | Return | Max drawdown | Closed trades |
|---|---:|---:|---:|---:|---:|
| Trend (10/30 SMA) | $979.95 | -$20.05 | -2.01% | 3.13% | 2 |
| Mean reversion (20 bar, -2 z entry) | $998.06 | -$1.94 | -0.19% | 0.27% | 0 |

Mean reversion had no completed trade in this window; the reported ending equity includes its marked open position. Its small loss is mainly estimated entry costs and mark-to-market movement. This check is not a robust backtest or evidence of future profitability: it covers one asset, one short period, and simplified execution. The local shell could not reach Kraken due to the current network policy, so these figures were computed from the public API response displayed in the in-app browser. Use the project fetcher and simulator when network access is available, then expand across assets and out-of-sample periods before drawing conclusions.

## Multi-asset screen (2026-11-01)

After market-data access became available, the project scanner downloaded the latest 720 daily candles for the ten highest-turnover USD crypto pairs, excluding common fiat and stablecoin bases. The period is approximately 2024-11-11 through 2026-10-31; HYPE has only 277 available bars. With the same $1,000 starting balance and cost assumptions, the portfolio simulator reported:

| Baseline | Ending marked equity | Return | Max drawdown | Closed trades | Win rate | Profit factor |
|---|---:|---:|---:|---:|---:|---:|
| Trend (10/30 SMA) | $2,758.92 | 175.89% | 52.54% | 89 | 33.71% | 1.91 |
| Mean reversion (20 bar, -2 z entry) | $1,402.52 | 40.25% | 35.58% | 78 | 70.51% | 1.51 |

Treat this only as a screening result. The assets were selected by their turnover on the run date, then tested over their recent history, so the result has substantial current-universe survivorship and selection bias. It also covers only the API's recent-history window, uses simplified next-open fills, and carries drawdowns above 35%. The attractive returns are not evidence of a durable edge. The next research work is point-in-time universe selection, longer independent/out-of-sample data, cost sensitivity, and a comparison against buy-and-hold. The closed-trade win rate alone is not a performance measure.
