# Architecture and research rules

## Data flow

`Public venue data -> validation -> SQLite candle store -> universe ranking -> strategy signals -> deterministic simulator -> paper ledger and reports -> local dashboard`

Market data, strategy definitions, persistence, and simulation are kept in separate modules. A later equity adapter can normalize data into the same candle contract, with exchange calendar and corporate action support added before it is considered comparable.

## Automatic selection

The Kraken scanner discovers online spot instruments quoted in USD and ranks them by 24 hour dollar turnover, using public ticker information. It fetches recent candles for the top configured number. This is a bounded first universe, not a claim to cover every exchange or every asset. It can abstain if no eligible data exists or if no strategy signal qualifies.

The scanner ranks current turnover; historical testing over this present day universe is vulnerable to survivorship and selection bias. Long term research needs point in time universe history.

The current endpoint reference is the [Kraken Spot public OHLC endpoint](https://docs.kraken.com/api-reference/market-data/get-ohlc-data), [tradable asset pairs](https://docs.kraken.com/api-reference/market-data/get-tradable-asset-pairs), and [ticker information](https://docs.kraken.com/api-reference/market-data/get-ticker-information).

## Strategy baselines

The trend baseline compares 10 and 30 bar simple moving averages. The mean reversion baseline enters below a 20 bar mean minus two population standard deviations and exits after recovery to the mean. Parameter values are fixed in this first comparison; do not tune them on the same sample used to report performance.

## Timing and accounting

- At each bar, a strategy decision uses only bars that completed before that bar.
- The simulated fill is at the next bar's open, with slippage applied against the account.
- Entry and exit fees are charged separately.
- The account is long only, cash funded, and has no leverage.
- Position notional is capped as a fraction of portfolio equity; concurrent positions are capped.
- Equity is marked to closed candle prices, with cash and open positions shown separately.
- Trade P&L includes entry and exit fees and simulated slippage through fill prices.
- The final current candle returned by the public source is discarded because it may still be forming.

## Missing realism to address in future research

The simulator does not model queue position, partial fills, market impact, exchange outages, taxes, funding or borrow costs, token transfer constraints, or all fee tiers. The short public candle window is not proof of strategy edge. Add out of sample and walk forward evaluation, data quality reports, and sensitivity to costs before relying on results.

## Project boundaries

The app uses public read only market endpoints for data. Do not add account credentials or live order routes. Historical backtesting and simulated paper trading are the boundary. The deterministic engine owns portfolio and risk controls. An LLM may classify or summarize market context, but must never submit trades, produce an executable order, or override a risk limit.

Paper accounts start at the newest stored candle unless `--replay-history` is requested. The watcher periodically refreshes public OHLC data, attaches the actual fetch timestamp to each live price, and limits a simulated event to snapshots from the same candle bucket fetched within a 30-second window. By default, watcher restarts skip missed events and resume from current prices; intentional catch-up replay requires `paper watch --replay-missed-history`. Per-poll successful and failed market fetch counts and snapshot exclusions are persisted for the dashboard. The watcher never calls an order endpoint.
