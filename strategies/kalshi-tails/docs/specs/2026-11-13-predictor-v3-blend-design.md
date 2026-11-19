# Predictor v3 (blend of v1 P and v2 F1): design

Date: 2026-11-13
Status: draft for review
Builds on:
- v1 (`strategies/kalshi-tails/docs/specs/2026-11-12-multi-market-tail-design.md`, frozen 2026-11-12)
- v2 (`strategies/kalshi-tails/docs/specs/2026-11-13-predictor-v2-design.md`, frozen set F1)
- the backtest dashboard (`strategies/kalshi-tails/docs/specs/2026-11-13-backtest-dashboard-design.md`)

## 1. Purpose

v1 P makes many small, steady trades, mostly at 1¢. v2 F1 makes fewer, larger-edge trades, mostly around 5¢. v3
combines them by blending their probabilities into one estimate, which then goes through the unchanged sizer:

    f_v3 = w · f_P + (1 − w) · f_F1

Averaging two partly independent estimates usually gives a more accurate probability than either alone. The
resulting trades fall naturally between the two styles.

**What is already known, stated before any v3 work:**
- On development data (BTC + S&P, 2025 to Jun 2026), v1 P alone beat a 50/50 split of the two books (Sharpe 1.63
  vs 1.46).
- The two models' daily returns correlate at 0.31.
- The holdout (Jul–Nov 2026) has been seen for both components.

v3 can therefore only be judged on forward data. Its development work is limited to choosing w by a fixed rule.

Success = on forward tail candidates in the BTC + S&P run, v3's log-loss is lower than both v1 P's and v2 F1's,
each by a one-sided 5% block-bootstrap test (section 7).

## 2. Scope

In scope:
- A blend estimator that runs both frozen components and combines their probabilities
- A pre-registered choice of w on development data
- A freeze, a holdout report labeled "seen", and a forward verdict on the same date as v2's
- A "v3 blend (frozen)" option in the dashboard

Out of scope:
- Changing either component. v1 P and v2 F1 keep their frozen settings, including v2's FROZEN dict.
- Blending in logit space, using more than two components, weights that vary by market, horizon or price, and
  learned or adaptive weights. Each would be a new spec.
- Changing the sizer, caps, fills, fees, tail filter or markets
- Changing the v1 or v2 forward verdicts. v3 is an additional arm.

## 3. Data periods

These are the same periods as v2. All dates are the expiry's New York date.

| Period | Dates | Use |
|---|---|---|
| Burn-in | 2024-07-05 → 2024-12-31 | training only |
| Development | 2025-01-01 → 2026-06-30 | choose w (section 6) |
| Holdout | 2026-07-01 → 2026-11-11 | scored once after the freeze, labeled "seen period, not the verdict" |
| Forward | 2026-11-12 → 2027-05-12 | the verdict, looked at once on or after 2027-05-12 |

- **Markets:** selection and the verdict use BTC + S&P 500. The five-market results are exploratory. Each run
  trains both components on the markets it trades, as in v2.
- **Rows:** v1's `tail_candidates`, H ∈ {6, 3}, and the 30-expiry-day warm-up per market.

## 4. Model

For each tail candidate row r:

    f_P(r), n_P   = v1 P estimate and its pooled cell count       (N0 = 50, N1 = 100)
    f_F1(r), n_F1 = v2 F1 estimate and its training-row count      (v2 FROZEN)
    eligible      = n_P ≥ 100 and n_F1 ≥ 200 and both estimates finite
    f_v3(r)       = w · f_P + (1 − w) · f_F1, clipped to [0.001, 0.999]

- **Eligibility:** v3 trades or scores a row only when both components are eligible. If one is missing, the row is
  skipped; v3 does not fall back to a single component. This keeps v3 a single, well-defined estimator and keeps
  its scored rows inside both components' common rows.
- **Learning:** both components learn walk-forward exactly as they do alone. `update(df)` passes the day's rows to
  both, after that day's decisions. Neither sees the other's output.
- **Estimator interface:** `estimate_row(row) -> (f_v3, n)`, where n = `min_rows` (eligibility flag) when
  eligible, else 0, with `min_rows = 1`. It plugs into the existing `multi_tail._candidates` hook with no change to
  `multi_tail.py`.

## 5. Inputs from earlier work

- v1 P: `estimators.PooledEstimator` with the frozen v1 constants
- v2 F1: `v2.logit_v2.LogitV2Estimator` built from `FROZEN`
- Walk-forward predictions: `v2.scoring.walk_forward_predict`, run separately for each component on the same rows,
  then blended. Blending after the fact is identical to running the blend estimator, because the components never
  interact. Section 10 has a test that confirms this.

## 6. Choosing w (pre-registered)

- **Candidate weights, fixed now:** w ∈ {0, 0.25, 0.5, 0.75, 1}. w = 1 is pure v1 P and w = 0 is pure v2 F1.
- **Rows:** development rows (BTC + S&P, 2025-01-01 → 2026-06-30) where both components are eligible.
- **Metric:** row-weighted mean log-loss of f_v3 for each w.
- **Uncertainty:** 7-day moving-block bootstrap, 4,000 resamples, seed 11, with the row-weighted Δ recomputed in
  each resample, as in v2.
- **Rule:** take the w with the lowest log-loss. If w = 0.5 is within 1 SE of the best on the paired Δ, take 0.5.
  Equal weighting is the default prior for a blend, so it wins near-ties.
- **If the rule picks w = 0 or w = 1,** v3 is identical to one of its components. In that case v3 is not frozen as
  a new arm, and the report says so; the existing forward test already covers that component.
- **No other weights, transformations or windows** are tried on development data.

**Freeze:** commit, record the hash, and write w and all of section 4 into `v3/README.md` under "Frozen v3
configuration".

## 7. Evaluation after the freeze

**Holdout (once).** On BTC + S&P common rows, report:
- log-loss for v3, v1 P, v2 F1 and the market mid
- paired Δ with one-sided 95th percentiles
- minimum detectable effect (MDE) for the 181-day window, using v2's formula (2.5 · SE_h · sqrt(D_h/181))
- the would-trade secondary
- Sharpe for v3 vs v1 P through the frozen sizer

It is labeled "seen period, not the verdict".

**Forward verdict (once, on or after 2027-05-12).** Same data and date as v2's verdict: expiry days 2026-11-12 →
2027-05-12, BTC + S&P.
- **Primary:** Δ_P = LL(v3) − LL(v1 P) and Δ_F1 = LL(v3) − LL(v2 F1), on common rows where all three are eligible.
  v3 wins only if both one-sided tests pass: the bootstrap 95th percentile is < 0 for each.
- **Secondary (reported, not required):**
  - Sharpe for v3 vs v1 P and vs v2 F1 through the frozen sizer, with `paired_diff_ci`
  - the would-trade log-loss and one-contract P&L
  - trades and winning trades per day
- **Exploratory:** five markets, no decision attached.
- **Guard:** the same forward guard as v2. Nothing is scored before 2027-05-12 (`hold_back_forward` and
  `evaluate`-style `--today` clamping).

## 8. Code

- **New folder** `strategies/kalshi-tails/kalshi-multi-tail/v3/`:
  - `blend.py`: `BlendEstimator(w, cfg)` wrapping a `PooledEstimator` and a `LogitV2Estimator(FROZEN)`, with
    `update(df)`, `estimate_row(row) -> (p, n)` and `min_rows = 1`
  - `select_w.py`: section 6 → `v3/output/selection.txt`. It reuses `v2.scoring` for walk-forward predictions, the
    bootstrap and the log-loss.
  - `evaluate.py`: section 7 → `v3/output/holdout.txt`, later `forward.txt`. It reuses `v2.evaluate`'s helpers and
    forward guard.
  - `README.md`, `tests/`
- **Hooks:** `estimators.make_estimator` gains kind `"V3"`, which reads the frozen w from `v3/blend.py`'s `FROZEN`.
  `MultiConfig` needs no new field. A regression test keeps v1 and v2 results unchanged.
- **Dashboard:** add the estimator label "v3 blend (frozen)" → ("V3", None). The accuracy tab works through the
  same `estimate_row` hook.
- **Data:** nothing is committed. Outputs go to git-ignored `v3/output/preds/`, and only the summary text files are
  committed.

## 9. What v3 is expected to do (stated before running)

- **Trades:** both components must be eligible and the blended edge must clear 0.5¢. v3 therefore trades only where
  the two models broadly agree on direction, which was every overlap on development. Its edge is the weighted
  average.
- **Trade count:** expect fewer trades than v1 P and somewhere around v2 F1's count or a bit more.
- **Price levels:** a mix of 1¢ and 5¢ contracts.
- **Risk:** expect drawdowns and cap breaches between the two components.

These are expectations to check against, not criteria.

## 10. Tests

- **`BlendEstimator`:**
  - with w = 1 its estimates equal v1 P's on every eligible row, and with w = 0 they equal v2 F1's
  - with w = 0.5 they are the mean of the two
  - a row where either component is ineligible returns (NaN, 0)
- **Same as blending after the fact:** on synthetic obs, running `walk_forward_predict` with `BlendEstimator(w)`
  equals blending the two components' walk-forward predictions.
- **No lookahead:** `update` after a predict does not change that prediction, and each component trains only on
  earlier days. This reuses v2's test pattern.
- **Selection rule:** on synthetic losses it picks 0.5 when within 1 SE of the best, and the best otherwise; w = 0
  or 1 reports "not a new arm".
- **v1 and v2 regression:** Model 0 (402 trades, +$52.61), the v1 P holdout (Sharpe 1.24, +5.98%, 361 trades), and
  the v2 holdout log-loss of 0.08207 all stay unchanged.
- **Forward guard:** `v3.evaluate --period forward` before 2027-05-12 prints only data-quality counts.
- **Dashboard:** "v3 blend (frozen)" runs without error and matches `multi_tail.run` with estimator V3.

## 11. Risks and limits

- **Development already favors v1 P.** The rule may well pick w = 0.75 or w = 1. w = 1 ends v3 as a separate arm,
  and that outcome is accepted in advance.
- **Statistical power.** v3 is built from its two comparators, so its paired Δs are small but also low-variance.
  The holdout report gives the MDE, and an inconclusive result is reported as inconclusive.
- **Three verdicts on one forward window.** v2 (vs S and P) and v3 (vs P and F1) are read on the same data. Each
  verdict is reported separately at its own 5% level, with no correction across them. The report says so.
- **Eligibility narrows coverage.** Requiring both components drops rows only one of them could price, mostly new
  markets early on and thin price bins for v1 P.
- **No weighting by conditions.** A single global w can't lean on v1 P at 1¢ and on v2 F1 around 5¢. That would
  need a per-price weight, which belongs in a later spec.
