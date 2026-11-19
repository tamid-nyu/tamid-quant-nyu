# Kalshi longshot tails (Marcos Garay)

Everything for the Kalshi "above $K" tail project: BTC digital-option pricing, the BTC tail strategy, its
multi-market extension and predictors v2 and v3.

| Folder | What it holds |
|---|---|
| `btc-kalshi-digital/` | BTC digital pricing backtest, the BTC tail strategy and sizer, model variants #0 to #7 (PR #1, #2) |
| `kalshi-multi-tail/` | Multi-market tail strategy v1: S&P 500, gold, WTI and gas, with k-worst-market sizing (PR #3) |
| `kalshi-multi-tail/v2/` | Predictor v2: pooled logistic model, frozen set F1 |
| `kalshi-multi-tail/v3/` | Predictor v3: fixed 50/50 blend of v1 P and v2 F1, frozen |
| `docs/specs/`, `docs/plans/` | Design specs and implementation plans, by date |
| `one-pagers/` | Strategy one-pagers (PDF); start with `Kalshi_Longshot_Tails_Project_Overview_Marcos_Garay.pdf` |
| `data/kalshi/` | Pulled Kalshi hourly quotes and contract metadata for all five markets, through 2026-11-11 |

## Status
All v1, v2 and v3 settings are frozen. The single forward test covers expiry days 2026-11-12 to 2027-05-12 and is
scored once on or after 2027-05-12 (`python -m v2.evaluate --period forward`, `python -m v3.evaluate --period forward`
from `kalshi-multi-tail/`; the code refuses earlier).

## Tests
Run each suite from its own folder (shared `conftest.py` names clash if suites are collected together):
```
cd btc-kalshi-digital && python -m pytest tests -q
cd kalshi-multi-tail && python -m pytest tests -q && python -m pytest v2/tests -q && python -m pytest v3/tests -q
```

A copy of the pulled Kalshi data is in `data/kalshi/`; the per-contract caches stay local.
