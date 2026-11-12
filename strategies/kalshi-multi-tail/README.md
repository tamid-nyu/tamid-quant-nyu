# Multi-market Kalshi tail strategy

The BTC tail strategy (`../btc-kalshi-digital/tail_strategy.py`) traded on several Kalshi "above K" ladders at once:
BTC, S&P 500, gold, WTI and natural gas (Nasdaq-100 and silver can be switched on in `markets.yaml`).
Each decision round is sized so that no market's worst settlement loses more than `c = 3%` of bankroll and the
`k = 2` worst markets together lose no more than `C = 4%`. Edge comes from estimator **S** (each market learns
alone) or **P** (each market is shrunk toward the cross-market pooled rate).

Spec: `docs/superpowers/specs/2026-11-12-multi-market-tail-design.md`
Plan: `docs/superpowers/plans/2026-11-12-multi-market-tail.md`

## Run
```
python multi_pull.py --market SPX --days 860     # one market at a time; raw data in output/raw/ (git-ignored)
python multi_pull.py --market GOLD --days 860    # ... WTI, NATGAS, NDX, SILVER
python multi_obs.py                              # output/obs/<KEY>.csv + history gate (>= 150 expiry days)
python multi_tail.py --all                       # S, P, BTC-only, option-1 baseline, k sensitivity -> output/
python -m pytest tests -q
```
BTC is not pulled again. Its observations come from `../btc-kalshi-digital/output/btc_pm_results_long/btc_observations.csv`.
The pull is rate-limited (about 6 requests/s) and resumable: candles are cached per ticker, and any day that fails
after all retries is listed in `output/raw/<KEY>/failed_days.txt`. Re-run the same command to retry.

## Files
- `markets.yaml`: series, event suffix, strike rule, pull band, enabled flag per market
- `multi_pull.py`: event-by-event pull of settled contracts and hourly bid/ask candles
- `multi_obs.py`: observations at 6h/3h before each market's own expiry, plus the history gate
- `estimators.py`: S and P edge estimators
- `multi_sizer.py`: LP with per-market and k-worst caps
- `multi_tail.py`: walk-forward simulator, report, CLI
- `output/summary.txt`, `summary_check.csv`, `summary_holdout.csv`, `trades_{S,P}.csv`, `daily_{S,P}.csv`,
  `market_daily_{S,P}.csv`, `equity.png`, `contribution.png`

## Checks
- BTC alone with `k = 1, C = c` reproduces the committed Model 0 trades exactly (402 trades, +$52.61).

## First results (2026-11-12)
See `output/RESULTS.md`. Holdout (Jul–Nov 2026): no setting beat BTC-only by a statistically significant margin.
Partial pooling (P) on five markets made 4.6 winning trades a day against 1.8 for BTC alone, with near-zero
cross-market P&L correlation. SPX helped and WTI lost money.
- `output/gate150/`: by the spec (BTC + SPX; gold, WTI and gas have < 150 expiry days)
- `output/gate100_posthoc/`: all five markets with `--min-history-days 100` (rule broken on purpose, exploratory)

## Frozen configuration (2026-11-12)
These settings are fixed for the forward test. Results on days after 2026-11-11 are the clean out-of-sample check,
so nothing below changes without a new, pre-registered spec.
- Markets: BTC + SPX (spec run); GOLD, WTI, NATGAS logged as exploratory (`--min-history-days 100`)
- Estimators: S (`N0 = 50`, min cell n 100) and P (`N1 = 100`, pooled min cell n 100); warmup 30 expiry days per market
- Sizing: c = 3%, k = 2, C = 4% of a $1,000 bankroll; `Q_MAX = 50`; `SLIP = 0.02`; `MIN_EDGE = 0.005`
- Rounds 6h and 3h before each market's expiry; taker fill at the next hour's quote; fee 0.07 × P × (1 − P), rounded up per order
- Code state: this commit; tests `python -m pytest tests -q` (32 passing)

Results page (charts of both runs): https://claude.ai/artifact/VLG94kNrxNmt4V4JW2JWAi
