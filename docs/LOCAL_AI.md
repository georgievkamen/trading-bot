# Local model experiment

The user wants the system to use local AI as part of its research. The selected model is Llama 3.2 3B via Ollama, installed on the Windows machine. The advisor calls the local Ollama HTTP service on `127.0.0.1:11434` and requests a strict label: `trend`, `range`, `high_volatility`, or `unknown`, with confidence and a short rationale. It receives derived percentage-return and volatility features rather than raw coin prices.

Check the local model's regime label with:

```powershell
$env:PYTHONPATH = "$PWD\src"
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'
& $python -m market_research regime BTC/USD --interval 60
```

The `llama_regime` paper strategy now lets the model choose which deterministic baseline to apply: `trend` selects the 10/30 moving-average rule, `range` selects the 20-bar mean-reversion rule, and `high_volatility` exits and blocks entries. Low-confidence/unknown labels block new entries and keep existing positions unchanged; an unavailable local model fails closed by exiting simulated positions. The 5% paper allocation cap, 4-position limit, fees, and slippage remain deterministic. The LLM cannot choose size or bypass risk controls.

Compare this forward AI-assisted account with the independent `trend` and `mean_reversion` paper accounts. The current historical backtest covers only deterministic baselines; do not treat LLM classifications generated today as historical evidence. Llama may misclassify regimes and is not a reliable standalone price predictor. Require longer out-of-sample paper results before claiming improvement.
