# Predictor v2: pooled logistic bias model

A pooled logistic model of the bias in Kalshi tail prices (mid < 10% or > 90%), across markets, used as the edge
estimator of the multi-market tail strategy in place of v1's S/P cell rates.

Spec: `strategies/kalshi-tails/docs/specs/2026-11-13-predictor-v2-design.md`
Plan: `strategies/kalshi-tails/docs/plans/2026-11-13-predictor-v2.md`

## Frozen v2 configuration (2026-11-13)
Fixed for the forward test. Nothing below changes without a new, pre-registered spec.

- **Feature set: F1** = F-base (`logit(mid)` slope, `h3` = 1[H = 3], `side` = 1[mid > 0.5], `side × logit(mid)`)
  + market offsets α_m. Chosen by the pre-registered rule of spec section 6 on development data only
  (`v2/output/selection.txt`: F3 had the lowest log-loss, F1 was within 1 SE of it, so the simplest set was taken).
  `FROZEN = {"set": "F1", "lam_beta": 25, "lam_alpha": 25, "refit_days": 7, "min_rows": 200}` in `v2/logit_v2.py`.
- **Model (section 4):** z = logit(mid) + b0 + β·x + α_m, p = sigmoid(z). logit(mid) is an offset with coefficient
  1. Loss = total log-loss + ½ λ_β ‖β‖² + ½ λ_α Σ α_m², b0 unpenalized, L-BFGS-B.
  - λ_β = 25, λ_α = 25
  - mid clipped to [0.001, 0.999] before logit; every scored probability (v2, v1 S, v1 P, mid) clipped to
    [0.001, 0.999]
  - features standardized with each fit's own training mean/std; constant columns dropped from that fit;
    warm start only when the column set is unchanged
  - refit before predicting expiry day d when ≥ 7 calendar days have passed since the last fit; training rows are
    the settled tail candidates of earlier expiry days
  - eligible to trade or score only when the fit has n ≥ 200 rows (`min_rows`) and the market is past v1's
    30-expiry-day warmup; before the first fit `estimate_row` returns (NaN, 0)
- **Rows (section 3):** v1 `tail_candidates`, H ∈ {6, 3}.
- **Features not in the frozen set** (built by `v2.features`, unused by F1): F-vol (realized σ_d over the last 20
  settlement log changes, missing if < 10; σ_H = σ_d · sqrt(min(H / T_m, 1)) with T_m = BTC 24, SPX 6.5, GOLD 23,
  WTI 23, NATGAS 23; isotonic ladder at `decision_ts`, implied median and interquartile σ_imp;
  `vol_prem` = ln(σ_imp / σ_H) with a `vol_missing` flag) and F-event (`event_day` from `events.yaml`).
- **Scoring (sections 6–7):** row-weighted mean log-loss; paired Δ = LL(a) − LL(b) on the same rows; moving-block
  bootstrap with 7-day blocks, 4,000 resamples, seed 11, row-weighted Δ inside each resample. Common rows: v2
  n ≥ 200 and v1 P pooled n ≥ 100; v1 S reported on the subset where its own cell n ≥ 100.
- **Sizer for the Sharpe secondary:** v1's frozen settings (c = 3%, k = 2, C = 4% of $1,000, `Q_MAX = 50`,
  `SLIP = 0.02`, `MIN_EDGE = 0.005`, fee 0.07 × P × (1 − P) rounded up per order).
- **Periods:** burn-in from 2024-07-05; development 2025-01-01 → 2026-06-30; holdout 2026-07-01 → 2026-11-11
  (seen period, not the verdict); forward 2026-11-12 → 2027-05-12, scored once on or after 2027-05-12. Primary
  scope BTC + SPX; the five-market results are exploratory.
- **Code state:** freeze commit `cfb5b9335db9370fa40fb5096533e0d706a65e90` ("v2: freeze configuration (F1)"); the evaluation code was added after it in `be14c12`, which also generalized `select._frame` (selection output unchanged); the model is untouched.

### Disclosure: an earlier holdout look
During Task 6 (the `multi_tail.run` V2 hook), a smoke run of `multi_tail.py --estimator V2` printed a holdout Sharpe
for v2 with the then-default set F3 (5.16, against 1.07 for BTC-only), before selection was run. Selection is a
mechanical rule on development data only (spec section 6 step 4) and nothing in the model, features or selection
was changed in response. The holdout was therefore already partly seen, which is one more reason it is reported as
"seen period, not the verdict"; the verdict is the forward test.

### Run
```
cd strategies/kalshi-tails/kalshi-multi-tail
python -m v2.features --markets BTC SPX GOLD WTI NATGAS   # v2/output/features/features.csv (git-ignored)
python -m v2.select                                       # development selection -> v2/output/selection.txt
python -m v2.evaluate --period holdout                    # -> v2/output/holdout.txt
python -m v2.evaluate --period forward                    # before 2027-05-12: data-quality counts only
python multi_tail.py --estimator V2 --markets BTC SPX     # sizer run with v2; the CLI passes the frozen set F1 (FROZEN["set"]), also the MultiConfig default
python -m pytest v2/tests -q && python -m pytest tests -q
```

## Holdout (2026-07-01 → 2026-11-11): seen period, not the verdict
Run once after the freeze; full report in `v2/output/holdout.txt`. BTC + SPX, 4,155 common rows over 134 days.
- Log-loss per row: v2 (F1) 0.08207, v1 P 0.08504, mid 0.08932; on the v1 S subset (3,937 rows) v2 0.07806,
  S 0.07985.
- Paired Δ (one-sided 95th percentile of the block bootstrap): v2 − P −0.00296 (−0.00025, passes);
  v2 − S −0.00180 (+0.00001, does not pass); v2 − mid −0.00724 (−0.00116, passes).
- Minimum detectable effect for the 181-day forward window: 0.0042 against P, 0.0026 against S. The holdout Δ
  against P (0.0030) and S (0.0018) are both below these, so the forward test may well come out "not shown".
- Sharpe through the frozen sizer: v2 2.51, P 1.24; difference +1.27, 95% CI [−4.09, +7.62].
- Five markets (exploratory): v2 − P −0.00166, v2 − S −0.00197; Sharpe v2 1.58, P 0.67.

## Forward refresh (spec section 9)
Before the forward verdict, refresh the data so it covers expiry days through 2027-05-12:
1. For each non-BTC market, re-run the pull for recent days, then rebuild the observations:
   ```
   python multi_pull.py --market SPX --days <N>     # ... GOLD, WTI, NATGAS; resumable, check failed_days.txt
   python multi_obs.py
   ```
2. For BTC, re-run its pull (this also refreshes `btc_spot_hourly.csv`, which v2 uses for BTC settlements) and its
   model backtest, from `strategies/kalshi-tails/btc-kalshi-digital/`:
   ```
   python btc_pm_data_pull.py --days <N>
   python btc_digital_model_backtest.py --data-dir output/btc_pm --out-dir output/btc_pm_results_long --no-plot
   ```
3. Rebuild the v2 features: `python -m v2.features --markets BTC SPX GOLD WTI NATGAS`.
4. On or after 2027-05-12: `python -m v2.evaluate --period forward` writes `v2/output/forward.txt`.

Warning: once the data covers forward days, `multi_tail.py` holds back rows after 2026-11-11 until the real date is on or after 2027-05-12 (it prints how many); do not work around this, and do not inspect forward results (per-market P&L, Sharpe, log-loss) before then. `v2.evaluate --today` can only move the date earlier than the real one.

Pulled Kalshi data, features, predictions and sim outputs are never committed.
