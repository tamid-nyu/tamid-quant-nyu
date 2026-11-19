# Predictor v3: fixed blend of v1 P and v2 F1

A fixed-weight average of two frozen estimators of the probability that a Kalshi tail contract settles YES: v1's
pooled cell rate P and v2's pooled logistic model F1. Nothing is refit or retuned beyond the components' own
walk-forward updates.

Spec: `strategies/kalshi-tails/docs/specs/2026-11-13-predictor-v3-blend-design.md`
Plan: `strategies/kalshi-tails/docs/plans/2026-11-13-predictor-v3-blend.md`

## Frozen v3 configuration (2026-11-13)
Fixed for the forward test. Nothing below changes without a new, pre-registered spec.

- **Weight: w = 0.5.** `FROZEN = {"w": 0.5}` in `v3/blend.py`; `estimators.make_estimator("V3", cfg)` reads it.
- **Blend:** f_v3 = w · f_P + (1 − w) · f_F1 = 0.5 · f_P + 0.5 · f_F1, clipped to [0.001, 0.999].
- **Components (frozen, code unchanged):**
  - v1 P = `PooledEstimator(n0=50, n1=100)`, eligible when its pooled cell n ≥ 100
  - v2 F1 = `LogitV2Estimator` built from `v2.logit_v2.FROZEN`
    (`{"set": "F1", "lam_beta": 25, "lam_alpha": 25, "refit_days": 7, "min_rows": 200}`), eligible when its fit
    has n ≥ 200 rows
- **Eligibility:** the blend is eligible only when both components are eligible and both estimates are finite;
  otherwise `estimate_row` returns (NaN, 0). There is no fallback to either component. When eligible, n = 1 and
  `min_rows = 1`, so the sizer's cell-count check passes on the blend's own eligibility.
- **Selection (spec section 6, `v3/output/selection.txt`):** development rows BTC + SPX, scored expiry days
  2025-01-01 → 2026-06-30 (walk-forward from 2024-07-05), 6,829 rows where both components are eligible. Mean
  log-loss by w: 0 → 0.09197, 0.25 → 0.09163, **0.5 → 0.09152**, 0.75 → 0.09169, 1 → 0.09232 (mid 0.09950).
  w = 0.5 had the outright lowest log-loss, so the rule (lowest log-loss; take 0.5 if within 1 SE of the best on
  the paired Δ, 7-day block bootstrap, 4,000 resamples, seed 11) chose w = 0.5 without needing the tie-break.
- **Periods:** development 2025-01-01 → 2026-06-30; holdout 2026-07-01 → 2026-11-11 (seen period, not the
  verdict); forward 2026-11-12 → 2027-05-12, scored once on or after 2027-05-12. Primary scope BTC + SPX; the
  five-market results are exploratory.
- **Forward verdict:** Δ_P = LL(v3) − LL(v1 P) and Δ_F1 = LL(v3) − LL(v2 F1) on common rows; WIN only if both
  one-sided tests pass (bootstrap 95th percentile < 0), else "not shown".
- **Multiplicity:** v2 and v3 verdicts are read on the same forward window, each at its own 5% level, with no multiplicity correction.
- **Sizer for the Sharpe secondary:** v1's frozen settings (c = 3%, k = 2, C = 4% of $1,000, `Q_MAX = 50`,
  `SLIP = 0.02`, `MIN_EDGE = 0.005`).
- **Code state:** freeze commit `abfcefaba29828ea3b10bde8028e2f0a7bf2d590` ("v3: freeze w = 0.5"); the evaluation code and the `--estimator V3` CLI option were added after it in `0fd5cc7`; the blend and its components are untouched.

### Run
```
cd strategies/kalshi-tails/kalshi-multi-tail
python -m v3.select_w                                     # development selection -> v3/output/selection.txt
python -m v3.evaluate --period holdout                    # -> v3/output/holdout.txt
python -m v3.evaluate --period forward                    # before 2027-05-12: data-quality counts only
python multi_tail.py --estimator V3 --markets BTC SPX     # sizer run with v3 -> v3/output/sim/ (git-ignored)
python -m pytest v3/tests -q
```

## Holdout (2026-07-01 → 2026-11-11): seen period, not the verdict
Run once after the freeze; full report in `v3/output/holdout.txt`. BTC + SPX, 4,155 common rows over 134 days.

- Log-loss: v3 0.08269, v1 P 0.08504, v2 F1 0.08207, mid 0.08932.
- Δ_P = LL(v3) − LL(P) = −0.00234 (95th pct −0.00061, passes); Δ_F1 = +0.00062 (95th pct +0.00197, fails).
  On the holdout v3 beats P but not F1, so it would be "not shown" by the forward rule.
- MDE for the 181-day forward window: 0.00264 vs P, 0.00159 vs F1.
- Sharpe through the frozen sizer: v3 3.62 (+14.14%, 288 trades), P 1.24 (+5.98%, 361), F1 2.51 (+9.83%, 158);
  Sharpe(v3) − Sharpe(P) = +2.37, 95% CI [−1.31, +8.23]; Sharpe(v3) − Sharpe(F1) = +1.11, 95% CI [−3.81, +6.77].

The verdict is the forward test (2026-11-12 → 2027-05-12, scored once on or after 2027-05-12).
