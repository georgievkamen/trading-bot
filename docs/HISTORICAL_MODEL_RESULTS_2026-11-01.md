# Historical learning result (2026-11-01)

## Decision

The first learned four-hour return model did **not** pass the paper-trial gate. Keep it offline and do not connect it to the running paper accounts. The results show that this model and these features do not currently provide a dependable profit signal after costs.

## Experiment

- Source: Kraken's full OHLCVT archive through June 2026 plus the Q3 2026 update. The final offline fit also included 21,690 closed 15-minute candles saved by the local paper watcher through 2026-11-01; those forward candles did not enter the fixed 2025–2026 walk-forward test windows.
- Universe: 588 USD-quoted crypto markets with usable history; at each decision time the simulator ranks the point-in-time top 30 by trailing 24-hour dollar turnover.
- Target: enter at the next 15-minute open and exit 16 bars later, exactly four hours after entry. Fees and adverse slippage apply at both ends.
- Model: local scikit-learn histogram gradient-boosting regressor, 15 price/volume/BTC-relative features. This is separate from Ollama/Llama and does not change Llama weights.
- Evaluation: four expanding chronological test windows. For each window, the trade threshold is selected only on the preceding six-month period; the training fit ends at least four hours before that validation or test window. If validation does not show a positive edge under the pre-set trade-count, profit-factor, and drawdown checks, the model stays in cash.
- Portfolio: $1,000 virtual equity, at most six positions, at most 10% of equity per position. The BTC buy-and-hold comparison uses 60% exposure. The model and both baselines use the same decision timestamps and four-hour holding horizon.
- Base costs: 26 bps fee plus 5 bps adverse slippage per side. Sensitivity: 52 bps fee plus 10 bps slippage per side.

## Out-of-sample results

| Test window | Model threshold selected beforehand | Model return, base costs | Model trades | Model return, higher costs | Trend | Mean reversion | BTC buy-and-hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 H1 | 3.0% predicted net return | -9.24% | 58 | -12.39% | -97.67% | -44.14% | +8.41% |
| 2025 H2 | Abstain | 0.00% | 0 | 0.00% | -97.89% | -50.45% | -11.29% |
| 2026 H1 | 2.0% predicted net return | -30.05% | 64 | -32.63% | -98.33% | -60.40% | -20.11% |
| 2026 Q3 | Abstain | 0.00% | 0 | 0.00% | -87.85% | -22.32% | +25.12% |

The learned model lost money in both periods when validation chose to trade. In the other two periods it correctly abstained under its validation rule, but earned nothing. The model gate failed: it had zero positive test periods at base costs and zero at higher costs. BTC holding also varied sharply by period, which reinforces that crypto direction changes and no single comparison proves an edge.

## Interpretation and next step

This is a useful negative result, not a profitable bot. Do not increase position sizes or relax the trade threshold to force activity. The next experiment should diagnose which predictions fail (by volatility, turnover, asset age, and market regime), verify them against a stricter spread/fill model, and require a new untouched test period before any paper integration. A later paper loss does not automatically train the model: retraining must be explicit, use only completed observations, and preserve a fresh future holdout.

The results use archived markets available in Kraken's downloadable files and simplified next-open fills; archive coverage cannot establish that every historical/delisted market is represented, and real fills can be worse. The running paper watcher is a separate 15-minute system and remains simulated. The trained artifact and detailed machine report stay in ignored `data/` and are not part of this repository change.

Reproduce locally with `python -m market_research learn` after placing both archives in `data/kraken_ohlcvt_archive/` and installing the optional `learning` dependencies. The command writes `data/historical_learning_report.json`, a local research database, and an offline model artifact.
