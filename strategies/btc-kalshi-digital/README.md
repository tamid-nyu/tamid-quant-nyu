# BTC Kalshi digital-option backtest

Kalshi "BTC above $X at 5pm ET" contracts are digital options. This backtest compares the contract price to a model probability
(normal or filtered-historical distribution; rv / EWMA / GARCH(1,1) vol) and trades the gap, with an optional delta hedge.

## Run
```
pip install -r ../../requirements.txt
python btc_pm_data_pull.py              # pulls Kalshi + spot data into output/btc_pm/ (not committed; includes a large cache)
python btc_digital_model_backtest.py    # writes results to output/btc_pm_results/
python btc_digital_model_backtest.py --h 6 --edge 0.04 --model garch --dist fhs --hedge-bps 5
```

## Contents
- `btc_pm_data_pull.py` - data pull (Kalshi hourly prices, markets, BTC spot)
- `btc_digital_model_backtest.py` - model, calibration test (Brier), trading rule, hedge, bootstrap CIs
- `output/btc_pm_results/` - chart, trades, observations, reliability table from the last run
- `output/btc_bt_stdout.txt` - console output from the last run

## Tail strategy (worst-case sized)
Trades only tail contracts (mid < 10% or > 90%) 6h and 3h before expiry, where the market overprices longshots.
Edge is a walk-forward estimate per (horizon, price bin) from prior expiry days, shrunk toward the market price.
Each expiry day's book is sized with a linear program so that the worst BTC settlement loses at most 3% of bankroll
(`tail_sizer.py`), and it is compared against one-contract and per-trade-cap baselines.
Spec: `docs/superpowers/specs/2026-11-11-btc-tail-strategy-design.md`.

```
python btc_digital_model_backtest.py     # writes output/btc_pm_results/btc_observations.csv (input)
python tail_strategy.py                  # writes output/btc_tail_results/
python tail_strategy.py --cap 0.02 --slip 0.03 --q-max 25
pip install -r ../../requirements-dev.txt && python -m pytest tests -q
```

- `tail_sizer.py` - settlement payoff matrix, worst-case check, LP sizer, per-trade baseline
- `tail_strategy.py` - tail filter, walk-forward edge, simulator, report and CLI
- `output/btc_tail_results/` - `tail_trades.csv`, `tail_daily.csv`, `tail_summary.txt`, `tail_backtest.png`
  (120 expiries, Jun - Oct 2026)

### Long history (base version)
815 expiries, Jul 2024 - Nov 2026. This is the base version that later variants are compared against.
```
python btc_pm_data_pull.py --days 860                  # raw data to output/btc_pm/ (git-ignored, ~1-2h)
python btc_digital_model_backtest.py --data-dir output/btc_pm --out-dir output/btc_pm_results_long --no-plot
python tail_strategy.py --data output/btc_pm_results_long/btc_observations.csv --out-dir output/btc_tail_results_long
```
Portfolio sizing over 481 live days (237 traded):
- **Return:** +5.3%, with a mean daily return of +0.011% [95% CI -0.023%, +0.041%]. That is not significantly above zero.
- **Losses:** 7 tail days lost about 3% each. Max drawdown -5.5%.
- **Cap:** 24 days ended past the 3% cap (worst 3.14%), because fills come an hour after sizing.

Before 2026 the 4-5c spreads ate the edge, and the strategy traded only twice.
