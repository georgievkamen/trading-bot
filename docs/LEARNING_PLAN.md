# Historical model experiment

## Objective

The first learned model should estimate the next four-hour return after the configured entry and exit fees and slippage. It is a separate supervised research model; Llama 3.2 remains a local regime-classification experiment and is not fine-tuned by paper trades.

## Data

The Kraken public OHLC REST endpoint only returns its 720 most recent rows, so it is not enough for multi-year 15-minute training. Kraken also publishes downloadable OHLCVT CSV history for each market from its first trade, including 15-minute candles. Its full archive is split into large files, and the quarterly Q3 2026 update extends coverage through 30 September 2026. Keep this raw data and the resulting local model out of GitHub; commit only the importer, features, training code, and evaluation report.

The archive omits intervals with no trades. The importer must preserve timestamp gaps explicitly and build a regular 15-minute grid for feature and four-hour target calculation. It must not treat a multi-hour gap as a four-hour return.

## Label and features

- At completed bar `t`, use only information available through that close.
- Simulate entry at the next 15-minute bar's open and exit at the open four hours later (16 intervals after entry).
- Compute the target return from the actual simulated entry/exit fills after fees and adverse slippage on both sides.
- Initial features: recent 15-minute returns over multiple windows, realized volatility, high-low range, volume/turnover changes, and BTC-relative movement. Add no feature that uses future bars.
- Reconstruct the liquid universe at each timestamp from trailing turnover; do not train only on the top assets selected today.

## Evaluation gate

Use chronological walk-forward splits with a 16-bar purge between train and test windows. Compare the learned model against cash, buy-and-hold, trend, and mean-reversion under identical symbols, timestamps, capital limits, fees, and slippage. Report net return, drawdown, trades, win rate, and profit factor, plus a higher-cost sensitivity run. Keep it paper-only unless it shows repeatable positive out-of-sample performance across more than one market period and cost setting.

The `market_research learn` command implements this first pass. It processes the 15-minute USD crypto files as streams, selects the top 30 by trailing 24-hour turnover at each four-hour decision point, and keeps gaps explicit. The four chronological test windows are 2025 H1, 2025 H2, 2026 H1, and 2026 Q3. Before each test window, a separate preceding six-month period can select one of a small set of prediction thresholds; if none meets a minimum return, trade-count, profit-factor, and drawdown check, the model abstains. Test dates are not used to choose the threshold. The model is refit using all eligible data before each test window, with a four-hour purge.

At each four-hour decision time, the simulation closes the prior test position and can allocate up to 10% of equity to each of at most six selected pairs. Signal selection uses only the top-turnover candidates available at that decision time. If a selected pair lacks a verifiable next open or four-hour exit because the archive has a gap, that slot remains unfilled; the backtest does not replace it after seeing the future. The buy-and-hold benchmark uses a 60% BTC position over that same test window, with the same entry and exit cost assumptions.

The current short intraday checks are negative and the positive daily check has high drawdown and present-universe selection bias. They are reasons to investigate, not evidence that a model will make money. The model should abstain when its predicted net return does not clear costs; increasing trade size cannot repair a missing edge.

Training is an explicit research command. The paper watcher already stores closed 15-minute candles in the local SQLite market database; `market_research learn` can reconstruct features and four-hour labels from those candles once the outcomes have matured. It does not train from paper balance changes or update the running model automatically. The historical test windows remain fixed, and any paper model trial must start after the tested dates so later forward observations remain a fresh evaluation set. Do not infer learning from the fact that a paper balance changed.

## Local compute

The first candidate is scikit-learn's histogram-based gradient-boosting regressor, which is designed to train faster on larger feature tables. Fine-tuning the 3-billion-parameter Llama model is not the first experiment: it would be heavier and would not provide a direct objective tied to post-cost returns. Llama can remain a separate regime labeler for an ablation after the supervised model is validated. To install the optional learning dependency in an isolated Windows environment, create `.venv` and run `\.venv\Scripts\python.exe -m pip install -e ".[learning]"`.
