# Predictor v2 for the multi-market tail strategy: design

Date: 2026-11-13 (revised after review, same day)
Status: draft for review
Builds on: `strategies/kalshi-tails/docs/specs/2026-11-12-multi-market-tail-design.md` (v1, frozen 2026-11-12)

## 1. Purpose

v1 estimates each tail contract's YES probability from a table of past hit rates per (hours to expiry, price bin).
That table cannot tell a calm day from a volatile one, or a quiet day from a Fed day. Predictor v2 replaces the
edge estimator with one that conditions on volatility regime and scheduled events. It plugs into the existing v1
sizer and simulator without changing them.

v2 is developed and selected only on data before 2026-07-01. It is frozen before it is scored on anything later.
Its verdict comes from forward data after 2026-11-11, at one pre-set look date.

Success = on forward tail candidates in the BTC + SPX spec run, v2's log-loss is lower than **both** v1 S and v1 P,
each by a one-sided test at the 5% level (section 7).

## 2. Scope

In scope:
- Feature construction from Kalshi data and a hand-made event calendar
- A pooled logistic bias estimator using the v1 `update` / `estimate` contract
- A pre-registered selection among three feature sets on development data
- Holdout and forward scoring scripts, and the frozen configuration

Out of scope:
- Changes to v1's sizer, caps, fills, fees, tail filter, or market list
- Automated daily data pulls. Forward data comes from re-running the existing pulls (section 9).
- New markets, maker orders, depth checks
- External implied-vol data (FRED VIX/GVZ/OVX). It was considered and cut: in the BTC + SPX runs it would vary
  only for SPX, so selection could barely weigh it.

## 3. Data periods

All dates are the expiry's New York calendar date (the `day` column in the obs files).

| Period | Dates | Use |
|---|---|---|
| Burn-in | 2024-07-05 → 2024-12-31 | training only, never scored |
| Development | 2025-01-01 → 2026-06-30 | walk-forward scoring and selection (section 6) |
| Holdout | 2026-07-01 → 2026-11-11 | already seen by v1. Scored once after the freeze, labeled "seen period, not the verdict" |
| Forward | 2026-11-12 → 2027-05-12 (181 days) | the verdict, looked at once on or after 2027-05-12 |

Training is always walk-forward: a fit used on expiry day d sees only rows with expiry day < d, in every period.
Holdout rows may be used as training data for forward predictions, but never to choose anything.

**Markets per run.** Each run trains v2 on the same markets it trades, as v1 P does:
- Selection and verdict: the BTC + SPX spec run.
- All five markets: exploratory and reported, with no decision attached.

**Rows.** Both scoring and trading use v1's own filters:
- tail candidates as `tail_candidates` in `tail_strategy.py` (including its spread and bid/ask filters)
- H ∈ {6, 3}
- v1's 30-expiry-day warmup per market

## 4. Model

For a tail candidate row r in market m:

    z = logit(mid_r) + b0 + β · x_r + α_m
    p_r = sigmoid(z), clipped to [0.001, 0.999]

mid is clipped to [0.001, 0.999] before logit (a safety guard only; tail mids in the data lie in
[0.015, 0.985]). When scoring, every estimator (v2, v1 S, v1 P, mid) is clipped to the same [0.001, 0.999].

- `logit(mid)` is an offset with its coefficient fixed at 1, so the model starts from "the market is right".
  `x` includes logit(mid) again, so a learned slope captures favorite–longshot bias.
- **Loss and penalties.** Minimize total log-loss + ½ λ_β ‖β‖² + ½ λ_α Σ_m α_m², with b0 unpenalized.
  Use L-BFGS-B, as in `LogitModel` (`tail_strategy.py`).
- **Penalty sizes.** λ_β = 25 and λ_α = 25. At tail probabilities (p(1−p) ≈ 0.05), a standardized feature over
  about 5,000 rows has log-loss curvature of about 250. λ = 25 therefore shrinks a well-supported weight by about
  10%. It shrinks an offset backed by a few hundred rows by more than half.
- **Standardization.** Before each fit, standardize features with the mean and std of that fit's training rows,
  and drop constant columns from that fit. Warm-start from the previous solution only when the column set is
  unchanged; otherwise start from zeros. The problem is convex, so warm-starting affects speed only.
- **Training rows.** Settled tail candidates (section 3) of the run's markets.
- **Refit timing.** Refit before predicting expiry day d whenever 7 or more calendar days have passed since the last
  fit.
- **Return value.** `estimate_row(row)` returns (p, n), where n = training rows in the current fit. A row is
  eligible to trade or score only if n ≥ 200 and its market is past v1's 30-day warmup. Before the first fit it
  returns (NaN, 0).
- **Condition number.** Every fit records the condition number of the standardized design matrix, and the selection
  report prints its maximum.

## 5. Features

Every feature uses only information available at the row's `decision_ts`.

Missing values:
- **Time-varying gaps** (e.g. too little settlement history, a ladder that doesn't span) are set
  to 0 after standardization. They get a 0/1 missing flag per feature group.
- **Structural gaps** (a market that never has the input) are set to 0 with no flag. The market offset α_m absorbs
  the level. A flag that equals a market dummy is not allowed.

**F-base**
- `logit(mid)` (slope)
- `h3` = 1[H = 3]
- `side` = 1[mid > 0.5]
- `side × logit(mid)`

**F-vol** (vol premium)
- **Settlement series per market.** The `expiration_value` of each past expiry (raw `markets.csv`), one per expiry
  day. BTC has no `expiration_value` in its pull, so it uses `btc_spot_hourly.csv`. Its `time` column is the
  candle's open, so the price at expiry T is the `open` of the candle with `time` = T (not that candle's
  `close`, which is the price at T + 1h). A value is usable for row r only if its expiry < r's `decision_ts`.
- **Realized vol.**
  - σ_d = std of log changes between consecutive usable settlements, over the last 20 changes.
  - Missing if there are fewer than 10 changes. A weekend or holiday gap counts as one change.
  - Session hours per interval T_m (set in `markets.yaml`): BTC 24, SPX 6.5, GOLD 23, WTI 23, NATGAS 23.
  - σ_H = σ_d · sqrt(H / T_m), with H / T_m capped at 1.
  - Close-to-close σ_d includes overnight moves, so σ_H is biased upward by a different amount per market. α_m and
    the standardization absorb the level. The feature carries the time variation.
- **Ladder at decision time.**
  - Built from the raw hourly quotes (`output/raw/<KEY>/hourly.csv`, BTC's `btc_kalshi_hourly.csv`) at exactly
    `decision_ts`, for all strikes of the event with a two-sided quote. It does not use the obs files, whose rows
    depend on a quote existing one hour later.
  - p(K) = (bid + ask) / 2, made non-increasing in K by isotonic fit.
  - Implied median M = K where p = 0.5 (linear interpolation).
  - Interquartile implied σ: σ_imp = (K₀.₂₅ − K₀.₇₅) / (2 · 0.6745 · M), where K_q is the K with p(K) = q.
  - Missing if the ladder does not span p = 0.25 to 0.75.
- `vol_prem` = ln(σ_imp / σ_H). This is the only vol-ratio feature in F-vol. A realized-vol gap feature is dropped
  because it is close to a function of `vol_prem` and moneyness.

**F-event**
- `events.yaml` lists release timestamps (UTC) from 2024-07-01 to 2027-06-30:
  - FOMC statement: 14:00 ET
  - CPI: 08:30 ET
  - Employment Situation (jobs report): 08:30 ET
  - EIA Weekly Petroleum Status Report: Wed 10:30 ET, with holiday shifts
  - EIA Weekly Natural Gas Storage Report: Thu 10:30 ET, with holiday shifts
- Which events apply to which markets:
  - FOMC, CPI, jobs → BTC, SPX, GOLD
  - EIA petroleum → WTI
  - EIA gas storage → NATGAS
- `event_day` = 1 if any applicable release is scheduled on the expiry's New York date, before or after
  `decision_ts`.
  - Release dates are published in advance, so this adds no lookahead.
  - A release before the decision affects the day's realized move and the price; a release after it is still
    pending. Both are an "event regime". A pending-only flag would almost never fire at these decision times.
- Check: each event family's dates are checked against the official published schedule, and the source URL is
  recorded in the file header.

## 6. Development and selection (pre-registered)

Candidate feature sets, fixed now:

| Set | Features |
|---|---|
| F1 | F-base + market offsets |
| F2 | F1 + F-vol |
| F3 | F2 + F-event |

Procedure (BTC + SPX run):
1. Run each set walk-forward over burn-in plus development. Record p for every development tail candidate.
2. **Common rows:** development rows where every set is eligible and v1 P returns an estimate with pooled n ≥ 100.
   - Report v1 P and the market mid (p = mid) on these rows.
   - Report v1 S only on the subset where its own cell n ≥ 100, labeled as that subset.
   - Report row counts per market.
3. **Metric:** row-weighted mean log-loss over the common rows. For a pair (a, b), Δ = LL(a) − LL(b) on the same
   rows.
   - Uncertainty comes from a moving-block bootstrap over calendar weeks: blocks of 7 consecutive days, 4,000
     resamples, seed 11.
   - The row-weighted Δ is recomputed inside each resample. SE = std of the resampled Δ.
4. **Pick** the set with the lowest log-loss. If a simpler set (lower index) is within 1 SE of the best on the
   paired Δ, take the simplest such set.
5. **Report only** (selection uses step 4 alone):
   - each set's log-loss by market and by H
   - the reliability table (10 bins over each tail)
   - the maximum condition number
   - the "would trade" secondary from section 7

No other feature sets, penalties, or windows are tried on development data. If a bug is found after selection, the
fix is ledgered and selection is rerun once with the same rule.

Freeze: commit the code, record the hash, and write the chosen set and all section 4–5 constants into the README
under "Frozen v2 configuration".

## 7. Evaluation after the freeze

**Holdout (once, right after the freeze).**
- Same scoring as section 6 steps 2–3: v2 against v1 S, v1 P and mid, plus Sharpe through the sizer.
- Labeled "seen period, not the verdict".
- Also computes the per-day standard deviation of v2's paired Δ against v1 P and v1 S. That gives the minimum
  detectable effect (below), recorded before any forward data is scored.

**Minimum detectable effect, stated now.** Holdout v1 numbers (134 days):
- the SE of a paired Δ is about 0.0013 per row for two similar estimators (v1 S vs P), and about 0.005 against mid
- scaled to the 181-day window, that is about 0.0009–0.0043
- at 80% power with a one-sided 5% test, MDE ≈ 2.5 × SE ≈ 0.002–0.011 log-loss per row
- for scale, v1 P's whole gain over mid on the holdout was 0.0048 per row

An improvement smaller than the MDE will probably read as inconclusive. That outcome is accepted in advance.

**Forward verdict (once, on or after 2027-05-12).** Data: expiry days 2026-11-12 → 2027-05-12, BTC + SPX run.
- **Primary.** Two tests on the common rows of section 6 step 2:
  - Δ_P = LL(v2) − LL(v1 P)
  - Δ_S = LL(v2) − LL(v1 S), on the subset where v1 S is eligible
  - v2 wins only if both one-sided tests pass: the 95th percentile of the block-bootstrap distribution (section 6
    step 3) is < 0 for each.
  - Δ against mid is reported.
- **Pre-registered secondary (reported, not required).** On rows where either v2 or v1 P would trade (net edge ≥
  `MIN_EDGE` after `SLIP` and fees, by `best_side`), report:
  - Δ_P log-loss
  - each model's hypothetical one-contract P&L
- **Secondary: Sharpe (reported, not required).** v2 vs v1 P through the frozen sizer (c = 3%, k = 2, C = 4%, with
  all v1 fill and fee settings), with the paired Sharpe-difference CI from `multi_tail.paired_diff_ci`.
- **Exploratory:** all-five-markets results are reported, with no decision attached.
- **No interim looks** at forward results before 2027-05-12. Data-quality checks (row counts, failed days) are
  allowed.

## 8. Code

New folder `strategies/kalshi-tails/kalshi-multi-tail/v2/`, importing from the parent folder and `btc-kalshi-digital/`:
- `features.py`:
  - `settlement_series`, `realized_sigma`, `ladder_implied` (from raw hourly quotes), `event_flags`
  - `build_features(obs, raw_quotes, events) -> DataFrame` runs on all obs rows before the tail filter, then
    joins onto tail candidates by (ticker, H)
- `events.yaml`: release timestamps (committed; public schedule data)
- `logit_v2.py`: `LogitV2Estimator(feature_set, lam_beta=25, lam_alpha=25, refit_days=7, min_rows=200)` with
  `update(df)` and `estimate_row(row) -> (p, n)`
- `select.py`: section 6 run and report → `v2/output/selection.txt`
- `evaluate.py`: section 7 run and report → `v2/output/holdout.txt`, later `v2/output/forward.txt`
- `tests/`

The one change to v1 code: `multi_tail.run` calls `est.estimate_row(r)` when the estimator has that method, and
otherwise calls `est.estimate(market, H, mid)` as now. `make_estimator` gains kind `"V2"`. A regression test checks
that v1 S and P trades and daily P&L are identical before and after the change. The existing BTC Model 0 check (402
trades, +$52.61) must be asserted by a test, not only stated in the README.

## 9. Data handling

- Derived features and predictions stay in the git-ignored `v2/output/features/`.
  Summary text files and charts are committed, as in v1.
- Pulled Kalshi data is never committed (the repo is public).
- **Forward refresh:**
  - re-run `multi_pull.py` per market for recent days, then `multi_obs.py`
  - for BTC, re-run its pull (including `btc_spot_hourly.csv`) and its model backtest
  - document these steps in the README

## 10. Tests

- **No lookahead.** For synthetic rows, changing any settlement, quote or event after `decision_ts` leaves that
  row's features unchanged. The exception is `event_day`, whose schedule is known in advance. Removing the quote
  at decision + 1h leaves the ladder features unchanged.
- **`realized_sigma`.** Matches a hand-computed std on a 25-value series, and is missing below 10 changes.
- **`ladder_implied`.**
  - A ladder generated from a known lognormal recovers M and σ_imp within 1%.
  - A non-monotone ladder is fixed by isotonic fit.
  - A ladder that doesn't span 0.25–0.75 returns missing.
  - Only quotes at exactly `decision_ts` are used.
- **`event_flags`.**
  - A release on the expiry's NY date → 1, whether before or after `decision_ts`.
  - A release on the previous NY date → 0.
  - Markets not mapped to the event → 0.
- **`LogitV2Estimator`.**
  - If y is drawn from mid itself, β and α converge near 0.
  - With a planted bias (y from sigmoid(logit(mid) + 0.5·event_day)), it recovers a positive event weight.
  - n < 200 → no trade.
  - It never trains on rows with expiry day ≥ the predicted row's expiry day.
  - Warm-start is used only when the columns are unchanged.
- **Selection rule.** On synthetic losses, it picks the simpler set when within 1 SE and the better set otherwise.
  The block bootstrap recomputes the row-weighted Δ inside each resample.
- **v1 regression** (section 8), including an assertion of 402 trades and +$52.61.

## 11. Risks and limits

- **The scoring set is mostly BTC.** The spec run is BTC + SPX, and SPX's share of eligible development rows is
  small, so selection is mostly a BTC result. The holdout report shows rows per market so this is visible.
- **GOLD, WTI and NATGAS are exploratory only.** They have about 6 weeks (WTI) to 3 months of development data and
  do not enter selection or the verdict.
- **Overlap with v1's check period.** The development window overlaps v1's check period (2026-01-01 → 06-30). v1's
  design choices were made on it, so v1's scores there are slightly flattered.
- **Statistical power.** Even over 181 days, an improvement below the MDE (section 7) will likely read as
  inconclusive. That result is reported as inconclusive, with no extension decided after looking.
- **Concentrated losses.** In the holdout, the top 5 days carried half of v1 P's total log-loss. One forward tail
  day can dominate the verdict, and the block bootstrap shows this as a wide interval.
- **Event calendar errors.** The calendar is hand-made and checked against sources; a wrong date blurs the event
  effect.
