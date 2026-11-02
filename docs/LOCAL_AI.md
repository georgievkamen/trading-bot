# Local model experiment

The selected model is Llama 3.1 8B Instruct Q4_K_M through Ollama on the Windows desktop. The local advisor receives derived 15-minute price-return and volatility features and returns one strict label: `trend`, `range`, `high_volatility`, or `unknown`, with confidence and a short rationale. Eligible markets are classified in groups of three to balance throughput with response reliability on the local CPU. It selects between deterministic trend and mean-reversion rules; it cannot select position size or bypass the simulator's risk limits.

Check the local model's regime label with:

```powershell
$env:PYTHONPATH = "$PWD\src"
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'
& $python -m market_research regime BTC/USD --interval 15
```

## Optional headline context

Before an eligible Llama classification, the paper strategy fetches the public CoinDesk RSS feed and attaches up to five recent, potentially relevant headlines. The feed is cached for 15 minutes, limited to the last 24 hours, and uses only title, publisher, timestamp, and URL. It does not download full articles or require an API key.

A headline is not verified just because Llama can read it. The feed is one publisher and may be incomplete, stale, inaccurate, or irrelevant. The prompt tells the model to treat headlines as untrusted quoted context and ignore instructions within them. Headlines may influence the model's regime label, so the audit notes store the supplied source metadata and feed status for later review. If the feed is unavailable, classification continues using the candle features alone.

## Outcome feedback memory

Every eligible Llama label is saved with its pair, candle timestamp, close, confidence, regime, and any supplied headline metadata. Once the full next-four-hour window is available, the watcher resolves a long return using the next candle open and the candle close four hours later, after the configured fee and slippage assumptions. It only resolves an example once that outcome timestamp is at or before the current simulated event time, so replayed history cannot see future outcomes. It keeps at most one sample per pair in each four-hour window to limit overlapping labels.

Llama receives the prior resolved return and win rate for a regime only after that regime has at least 25 distinct four-hour decision windows. Multiple coins labeled with the same regime in one window are collapsed to one median return, so a single market move cannot satisfy the sample gate by itself. Until then it gets an explicit “not enough independent windows” status. This evidence memory may help calibrate labels; it does not fine-tune Llama's weights, change the deterministic entry rules, or change risk limits. The dashboard distinguishes raw resolved coin outcomes from independent four-hour windows and shows per-regime feedback once the gate is met.

News should be evaluated with an ablation: compare the same Llama design with news context on and off over untouched chronological paper periods, after fees and slippage. Do not attribute performance changes to the headlines without enough observations. Add authoritative source feeds and cross-source corroboration only if that experiment shows measurable value.

## Paper behavior and boundaries

The `llama_regime` account lets Llama choose which deterministic baseline to apply: `trend` selects the 10/30 moving-average rule, `range` selects the 20-bar mean-reversion rule, and `high_volatility` exits and blocks entries. Low-confidence or unknown labels block new entries and keep existing positions unchanged; an unavailable local model fails closed by exiting simulated positions.

The 10% paper allocation cap, six-position limit, fees, slippage, and simulated fills remain in the engine. The selected active setup runs the Llama account only. Other saved paper accounts may be kept as inactive controls. No live order route or automatic model-weight training exists. Public candles and headline metadata can be retained for later research, but they are not training labels by themselves.

Compare net results against cash and buy-and-hold baselines using chronological out-of-sample periods, with realistic costs and drawdown reporting. Llama is a regime labeler, not a reliable standalone price predictor. Require longer, reproducible paper evidence before claiming an improvement or changing the strategy.
