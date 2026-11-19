# Predictor v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pooled logistic bias edge estimator (v2) for the multi-market tail strategy, select its feature set
on pre-July-2026 data by a pre-registered rule, freeze it, and ship the holdout and forward scoring scripts.

**Architecture:** A new `strategies/kalshi-tails/kalshi-multi-tail/v2/` package.
- `features.py` adds per-row features (vol premium, event day) to the v1 obs table, built from raw Kalshi quotes,
  settlement values and a committed event calendar.
- `logit_v2.py` is a walk-forward estimator with an `estimate_row` method.
- `scoring.py` replays v1's walk-forward loop to record predictions from any estimator, and holds the block
  bootstrap and the selection rule.
- `select.py` and `evaluate.py` are the CLIs.
- v1 changes in one place only: `multi_tail._candidates` dispatches to `estimate_row` when the estimator has it.

**Tech Stack:** Python 3 (`/opt/miniconda3/bin/python3`), numpy, pandas, scipy (L-BFGS-B), PyYAML, pytest. No
sklearn: isotonic regression is a hand-written pool-adjacent-violators.

**Spec:** `strategies/kalshi-tails/docs/specs/2026-11-13-predictor-v2-design.md`

## Global Constraints

- Development = expiry NY dates 2025-01-01 → 2026-06-30. Burn-in from 2024-07-05. Holdout 2026-07-01 → 2026-11-11.
  Forward 2026-11-12 → 2027-05-12, looked at once on or after 2027-05-12.
- Selection and verdict run on BTC + SPX only. Five markets are exploratory.
- Model: z = logit(mid) + b0 + β·x + α_m; λ_β = 25, λ_α = 25, b0 unpenalized; refit when ≥ 7 calendar days have
  passed since the last fit; min_rows = 200.
- Clip mid to [0.001, 0.999] before logit. Every scored probability (v2, v1 S, v1 P, mid) is clipped to
  [0.001, 0.999].
- Feature sets: F1 = base + market offsets; F2 = F1 + F-vol; F3 = F2 + F-event. Nothing else is tried.
- Block bootstrap: moving blocks of 7 consecutive days, 4,000 resamples, seed 11, row-weighted Δ inside each
  resample.
- Rows: v1 `tail_candidates`, H ∈ {6, 3}, and v1's 30-expiry-day warmup per market.
- v1 behavior must not change: the BTC regression (402 trades, +$52.61) and S/P outputs must stay identical.
- Pulled Kalshi data and derived features are never committed. `v2/output/features/` and `v2/output/preds/` are
  git-ignored.
- Commit only when the user has said to. Each "Commit" step means: stage these paths, and commit if commits are
  authorized; otherwise leave them staged and note this in the ledger.

## Review Focus

1. **Ladder quotes at exactly decision_ts.** If a strike has no candle at decision_ts, it is excluded (no
   forward-fill from earlier hours). A forward-fill would silently use stale prices. Pinned by
   `test_ladder_uses_only_decision_ts`.
2. **Off-schedule (holiday) expiries in the settlement series.** An early-closed event (e.g.
   KXINXU-26JAN19H1600) must not enter σ_d. Pinned by `test_settlement_series_drops_off_schedule`.
3. **Market with no training rows yet.** v2 must predict with α = 0 for a market unseen at fit time and not crash
   on a missing dummy column. Pinned by `test_unseen_market_offset_zero`.
4. **Bootstrap with fewer than 7 days of data.** It must fall back to one block of all days, not error or return
   NaN silently. Pinned by `test_block_bootstrap_short_series`.
5. **Forward evaluation run before 2027-05-12.** It must refuse to print forward results and print only
   data-quality counts. Pinned by `test_forward_refuses_early`.

---

### Task 1: v1 hook and stronger regression

**Files:**
- Modify: `strategies/kalshi-tails/kalshi-multi-tail/multi_tail.py` (`_candidates`, near line 67)
- Modify: `strategies/kalshi-tails/kalshi-multi-tail/estimators.py` (`make_estimator`)
- Modify: `strategies/kalshi-tails/kalshi-multi-tail/tests/test_multi_tail.py`

**Interfaces:**
- Produces:
  - `make_estimator("V2", cfg)` returns `v2.logit_v2.LogitV2Estimator(cfg.v2_set)`, imported lazily.
  - `MultiConfig.v2_set: str = "F3"`.
  - `_candidates` uses `est.estimate_row(r)` if `hasattr(est, "estimate_row")`, otherwise
    `est.estimate(r.market, r.H, r.mid)`.
  - `MultiConfig.min_cell_n` remains the eligibility threshold for S/P. For V2 the threshold is `est.min_rows`:
    `_candidates` uses `getattr(est, "min_rows", cfg.min_cell_n)`.

- [ ] **Step 1: Write failing tests** in `test_multi_tail.py`:
  - `test_btc_regression_counts`: the BTC-only run (`k=1, cap_k=0.03, estimator="S"`) gives `len(tr) == 402` and
    `round(tr.pnl.sum(), 2) == 52.61`.
  - `test_estimate_row_dispatch`: a stub estimator with `estimate_row` returning (0.99, 10**6), `min_rows = 0` and
    a no-op `update`, run on the existing `conftest` toy obs, places trades. A stub with only `estimate` returning
    the same also places trades. Both produce the same trades frame.
- [ ] **Step 2: Run** `cd strategies/kalshi-tails/kalshi-multi-tail && /opt/miniconda3/bin/python3 -m pytest tests/test_multi_tail.py -q`
  Expected: `test_btc_regression_counts` PASS (it describes current behavior); `test_estimate_row_dispatch` FAIL
  (`estimate_row` is never called, so its stub's trades differ, or there is an AttributeError on `min_rows`).
- [ ] **Step 3: Implement** the dispatch, the `min_rows` lookup, `v2_set` and the `"V2"` branch.
- [ ] **Step 4: Run the full suite** `python -m pytest tests -q`. Expected: 34 passed (32 old + 2 new).
- [ ] **Step 5: Commit** `multi_tail.py estimators.py tests/test_multi_tail.py` — "multi-tail: estimate_row hook for v2 estimators".

### Task 2: Settlement series and realized vol

**Files:**
- Create: `strategies/kalshi-tails/kalshi-multi-tail/v2/__init__.py` (empty), `v2/features.py`
- Create: `v2/tests/conftest.py` (put `v2/`, the parent folder and `btc-kalshi-digital/` on `sys.path`, as in
  the parent `tests/conftest.py`), `v2/tests/test_features.py`
- Modify: `strategies/kalshi-tails/kalshi-multi-tail/markets.yaml`: add `session_hours` per market (BTC 24, SPX 6.5, GOLD 23,
  WTI 23, NATGAS 23, NDX 6.5, SILVER 23)

**Interfaces:**
- Produces:
  - `settlement_series(markets: DataFrame) -> DataFrame[expiry (UTC ts), value]`, for non-BTC markets. Takes
    settled rows with non-null `expiration_value`, keeps only on-schedule expiries (minute ∈ {0, 30}, zero
    seconds and microseconds, the same rule as `multi_obs.build_market_obs`), one row per expiry (first value),
    sorted.
  - `btc_settlement_series(spot: DataFrame, expiries: Series) -> DataFrame[expiry, value]`. value = `open` of the
    spot candle whose `time` == expiry. Expiries without a candle are dropped.
  - `realized_sigma(series: DataFrame, t: Timestamp, n_changes=20, min_changes=10) -> float`. Uses values with
    expiry < t, takes the last `n_changes + 1` values, and returns the std (ddof=1) of the log diffs. NaN if fewer
    than `min_changes` diffs.

- [ ] **Step 1: Write failing tests:**
  - `test_realized_sigma_hand`: 25 values `100 * exp(cumsum(r))` with known r. With t after all of them, the
    result equals `np.std(r[-20:], ddof=1)` to 1e-12.
  - `test_realized_sigma_min_changes`: 10 values (9 changes) → NaN; 11 values → finite.
  - `test_realized_sigma_strictly_before`: a value whose expiry == t is excluded.
  - `test_settlement_series_drops_off_schedule`: an expiry at `2026-01-17T17:53:55.869664Z` is dropped, and one at
    `21:00:00` is kept.
  - `test_btc_settlement_uses_open`: a candle with `time = T`, `open = 1`, `close = 2` gives value 1.
- [ ] **Step 2: Run** `cd strategies/kalshi-tails/kalshi-multi-tail/v2 && python -m pytest tests -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** the three functions.
- [ ] **Step 4: Run.** Expected: 5 passed.
- [ ] **Step 5: Commit** `v2/__init__.py v2/features.py v2/tests/ markets.yaml` — "v2: settlement series and realized vol".

### Task 3: Ladder implied median and σ

**Files:** Modify `v2/features.py`, `v2/tests/test_features.py`

**Interfaces:**
- Produces:
  - `pav_decreasing(p: ndarray) -> ndarray`: pool-adjacent-violators fit, non-increasing.
  - `ladder_implied(strikes: ndarray, bid: ndarray, ask: ndarray, max_spread=0.10) -> (M, sigma_imp)`:
    - Uses quotes with `ask - bid <= max_spread`. **Ruling** (the spec says "two-sided quote"): a wider spread
      such as bid 0 / ask 1 carries no information.
    - Sorts by strike and takes p = PAV of mids.
    - Interpolates K at p = 0.5, 0.25 and 0.75. Where the fitted p has flat runs, collapse them to one point
      (mean K per p level) before `np.interp` on increasing p.
    - Returns (NaN, NaN) unless max p ≥ 0.75 and min p ≤ 0.25.
    - σ_imp = (K₀.₂₅ − K₀.₇₅) / (2 · 0.6745 · M).
  - `ladder_table(hourly: DataFrame, markets: DataFrame, decision_times: DataFrame[event_ticker, decision_ts]) -> DataFrame[event_ticker, decision_ts, M, sigma_imp]`.
    Uses only `hourly` rows with `ts == decision_ts` exactly, for that event's tickers (joined via
    `markets.ticker → event_ticker, floor`). BTC: only `strike_type == "greater"` markets.

- [ ] **Step 1: Write failing tests:**
  - `test_ladder_lognormal`: strikes `100 * exp(linspace(-0.1, 0.1, 30))`, p = 1 − Φ(ln(K/100)/0.03), bid = p −
    0.005, ask = p + 0.005. Recovers M within 1% of 100 and σ_imp within 1% of 0.03.
  - `test_ladder_nonmonotone`: the same ladder with one mid bumped up by 0.05 gives the same M within 1%.
  - `test_ladder_not_spanning`: all p in [0.3, 0.95] → (NaN, NaN).
  - `test_ladder_wide_spread_ignored`: adding a bid 0 / ask 1 strike changes nothing.
  - `test_ladder_uses_only_decision_ts`: in `ladder_table`, changing quotes at decision_ts + 1h, or deleting them,
    leaves the output unchanged. A strike with no candle at decision_ts is excluded.
- [ ] **Step 2: Run.** Expected: the 5 new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 10 passed.
- [ ] **Step 5: Commit** — "v2: ladder implied median and sigma from raw quotes".

### Task 4: Event calendar and event flags

**Files:** Create `v2/events.yaml`. Modify `v2/features.py`, `v2/tests/test_features.py`

**Interfaces:**
- Produces:
  - `events.yaml` format: `families: {fomc: {markets: [BTC, SPX, GOLD], time_et: "14:00", source: <url>, dates: [YYYY-MM-DD, ...]}, cpi: ..., jobs: ..., eia_petroleum: {markets: [WTI], time_et: "10:30", ...}, eia_gas: {markets: [NATGAS], time_et: "10:30", ...}}`.
    Dates span 2024-07-01 → 2027-06-30. FOMC dates are the statement day (the second day of the meeting).
  - `load_events(path) -> dict[market, set[date]]` (NY dates).
  - `event_day(market: str, day: date, events) -> int`.

- [ ] **Step 1: Build `events.yaml` from official sources.** Use web search or fetch; record each family's source
  URL in its `source` field:
  - FOMC: federalreserve.gov FOMC calendars
  - CPI and Employment Situation: bls.gov release schedules for 2024–2027
  - EIA Weekly Petroleum Status Report and Weekly Natural Gas Storage Report: EIA schedule and holiday-release
    pages

  Use **actual** release dates where releases were moved, e.g. the BLS releases delayed or cancelled by the
  Oct–Nov 2025 federal shutdown. A cancelled release gets no date. **Ruling:** a rescheduled date was published
  before the release, so using it is not lookahead. Ledger every moved date. If any family's official dates
  cannot be confirmed for some span, record the span in the file header and ledger it. Do not guess dates.
  EIA weeks with no listed exception use Wed (petroleum) and Thu (gas).
- [ ] **Step 2: Write failing tests:**
  - `test_event_day_on_date`: FOMC on 2025-12-10 → `event_day("SPX", 2025-12-10) == 1`.
  - `test_event_day_previous_date`: `event_day("SPX", 2025-12-11) == 0`.
  - `test_event_unmapped_market`: `event_day("WTI", 2025-12-10) == 0`.
  - `test_events_yaml_sane`: every family has a source and dates in range, with no duplicates. The petroleum and
    gas families have at least 140 dates; FOMC has 8 per full calendar year.
- [ ] **Step 3: Run.** Expected: FAIL.
- [ ] **Step 4: Implement** `load_events` and `event_day`.
- [ ] **Step 5: Run.** Expected: 14 passed.
- [ ] **Step 6: Commit** `v2/events.yaml v2/features.py v2/tests/` — "v2: event calendar and event_day flag".

### Task 5: Feature table builder

**Files:**
- Modify: `v2/features.py`, `v2/tests/test_features.py`
- Modify: repo `.gitignore` (add `strategies/kalshi-tails/kalshi-multi-tail/v2/output/features/` and
  `strategies/kalshi-tails/kalshi-multi-tail/v2/output/preds/`)

**Interfaces:**
- Consumes: Tasks 2–4. `multi_obs.load_all(keys)` for obs.
- Produces:
  - `build_features(obs: DataFrame, raw: dict[market, (markets_df, hourly_df)], btc_spot: DataFrame, events, session_hours: dict) -> DataFrame`:
    - Returns obs plus columns `sigma_H`, `M`, `sigma_imp`, `vol_prem` (NaN when missing), `vol_missing` (0/1)
      and `event_day` (0/1).
    - σ_H = σ_d · sqrt(min(H / T_m, 1)), and vol_prem = ln(σ_imp / σ_H).
    - The ladder join key is (event_ticker, decision_ts). The obs ticker maps to its event via `markets_df`. BTC's
      raw files are `btc_kalshi_markets.csv` and `btc_kalshi_hourly.csv`; its settlement series uses the BTC obs
      expiries.
  - `load_raw(keys) -> (raw, btc_spot)`: reads `output/raw/<KEY>/markets.csv`, `hourly.csv` and BTC's
    `../btc-kalshi-digital/output/btc_pm/{btc_kalshi_markets,btc_kalshi_hourly,btc_spot_hourly}.csv`.
  - CLI `python -m v2.features --markets BTC SPX GOLD WTI NATGAS` writes
    `v2/output/features/features.csv` and prints, per market, the row count and the share with `vol_missing == 0`.

- [ ] **Step 1: Write a failing test,** `test_no_lookahead_scramble`. A synthetic 2-market obs with raw quotes,
  settlements and events, built twice:
  - once as is
  - once after changing every settlement value with expiry ≥ a cutoff day, every quote with ts > decision_ts of
    that day's rows, and deleting all quotes at decision_ts + 1h

  Features for rows on the cutoff day are identical.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `build_features`, `load_raw`, the CLI and the `.gitignore` lines.
- [ ] **Step 4: Run** the tests (expected: 15 passed), then the CLI on real data. Expected: completes, with
  per-market spanning shares near the review's figures (dev-period shares at 0.16–0.84 were BTC 91%, SPX 93%,
  GOLD 77%, WTI 88%, NATGAS 14%; at 0.25–0.75 they should be higher). Record the printed table in the ledger.
- [ ] **Step 5: Commit** `v2/features.py v2/tests/ ../../.gitignore` — "v2: feature table builder".

### Task 6: LogitV2Estimator

**Files:** Create `v2/logit_v2.py`, `v2/tests/test_logit_v2.py`

**Interfaces:**
- Consumes: feature columns from Task 5.
- Produces: `LogitV2Estimator(feature_set="F3", lam_beta=25.0, lam_alpha=25.0, refit_days=7, min_rows=200)`:
  - `FEATURES = {"F1": ["lm", "h3", "side", "side_lm"], "F2": F1 + ["vol_prem", "vol_missing"], "F3": F2 + ["event_day"]}`,
    where `lm = logit(clip(mid))` and `side_lm = side * lm`.
  - `update(df)` appends that day's settled rows: features, offset lm, market, y and day.
  - `estimate_row(row) -> (p, n)`:
    - Refits before predicting if (no fit yet and rows ≥ min_rows) or (fitted and row.day − last_fit_day ≥ 7 days).
    - n = rows in the current fit. Returns (NaN, 0) before the first fit.
  - `estimate(market, H, mid)` raises `TypeError("V2 needs estimate_row")`.
  - Fit details:
    - Standardize the feature columns (NaN `vol_prem` filled with the training mean first). Drop zero-variance
      columns.
    - Market dummies are not standardized and are penalized with λ_α. b0 is unpenalized.
    - Warm-start only if the column set is unchanged.
    - Store `self.cond` = condition number of the standardized design matrix, and append it to `self.cond_log`.
    - Prediction uses the stored means and stds, and α = 0 for a market without a dummy.

- [ ] **Step 1: Write failing tests:**
  - `test_null_bias_near_zero`: 5,000 synthetic rows with y ~ Bernoulli(mid) over 60 days. After the fit, |β| <
    0.15 for every feature and |α| < 0.15.
  - `test_planted_event_bias`: y ~ Bernoulli(sigmoid(logit(mid) + 0.5·event_day)). The F3 event weight, divided
    by the event column's std, is > 0.25.
  - `test_min_rows_gate`: with 150 rows, it returns (NaN, 0).
  - `test_never_trains_on_same_or_future_day`: after `update` with days ≤ d − 1, predicting day d uses n equal to
    the row count of days < d. Calling `update` with day d after the predict does not change that prediction.
  - `test_refit_cadence`: predictions on days d and d + 3 use the same fit; d + 7 triggers a refit (check
    `last_fit_day`).
  - `test_unseen_market_offset_zero`: train on BTC only; predicting an SPX row returns a finite p with no error.
  - `test_warm_start_only_same_columns`: when a column turns constant between fits, the next fit starts from zeros
    (the warm-start flag is recorded on the estimator).
- [ ] **Step 2: Run** `python -m pytest tests/test_logit_v2.py -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the whole v2 suite. Expected: 22 passed. Then the parent suite with
  `test_estimate_row_dispatch`, plus a smoke run `python multi_tail.py --markets BTC SPX --estimator V2` on the
  real features. **Ruling to record:** `multi_tail.run` needs feature columns on obs, so `main` loads
  `v2/output/features/features.csv` when `--estimator V2` is given. Expected: completes and writes outputs.
- [ ] **Step 5: Commit** `v2/logit_v2.py v2/tests/ multi_tail.py` — "v2: pooled logistic bias estimator".

### Task 7: Scoring harness and selection

**Files:** Create `v2/scoring.py`, `v2/select.py`, `v2/tests/test_scoring.py`

**Interfaces:**
- Consumes: `make_estimator`, `ts.tail_candidates`, `ts.best_side`, `ts.fee`, `MultiConfig`.
- Produces:
  - `walk_forward_predict(obs, est, cfg) -> DataFrame[day, market, ticker, H, mid, bid, ask, bid_f, ask_f, y, p, n]`.
    This is the same day loop as `multi_tail.run`: tails → warm markets → predict → `est.update(tails)` → bump
    days_seen. It records every live tail row, with p = NaN when the estimate is missing.
  - `clip_p(p)` clips to [0.001, 0.999]. `logloss(p, y)` gives per-row values.
  - `block_bootstrap_delta(day, d, n=4000, seed=11, block=7) -> ndarray`:
    - Takes per-row day labels and per-row Δ, and returns 4,000 resampled row-weighted mean Δ.
    - Moving blocks of 7 consecutive distinct days (in sorted day order), with start indices uniform on
      [0, D − 7]. Concatenate ceil(D/7) blocks and truncate to D days.
    - If D < 7, use a single block of all D days.
  - `select_set(ll: dict[set_name, per-row ll], day) -> (chosen, table)`: implements spec section 6 step 4, with
    SE = std of the bootstrap Δ.
  - `would_trade(df, cfg) -> bool mask`: `best_side` edge ≥ `cfg.min_edge` with a valid fill quote, as in
    `_candidates`.
  - `select.py` CLI:
    - For S, P, F1, F2 and F3 on BTC + SPX, it runs `walk_forward_predict` on features from 2024-07-05 through
      2026-06-30 only (it drops later rows before running).
    - It builds the common rows: dev period, every v2 set n ≥ 200 and v1 P n ≥ 100. v1 S is reported on its own
      n ≥ 100 subset.
    - It writes `v2/output/selection.txt` with:
      - LL per estimator and mid, overall and by market and H, plus row counts per market
      - paired Δ with SE
      - the chosen set
      - the reliability tables (10 bins over each tail)
      - the max condition number
      - the would-trade secondary
    - Predictions go to `v2/output/preds/dev_*.csv`.

- [ ] **Step 1: Write failing tests:**
  - `test_walk_forward_matches_run_estimates`: on `conftest` toy obs, the S predictions from
    `walk_forward_predict` equal the f values recorded in `multi_tail.run`'s trades for the traded rows.
  - `test_block_bootstrap_row_weighted`: day A has 1 row with Δ = 1, day B has 9 rows with Δ = 0, and 8 more days
    have one Δ = 0 row each. The mean of the resamples is close to the row-weighted mean 1/18, not the day-mean
    0.1.
  - `test_block_bootstrap_short_series`: 3 days gives 4,000 finite values.
  - `test_select_prefers_simpler_within_se` and `test_select_takes_better_outside_se`, on synthetic per-row losses.
  - `test_select_ignores_post_dev_rows`: rows dated 2026-07-01 or later in the input do not change selection.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the v2 suite (expected: 28 passed). Then `python -m v2.select`. Expected: it writes
  `selection.txt`. Copy the chosen set and the Δ table into the ledger.
- [ ] **Step 5: Commit** `v2/scoring.py v2/select.py v2/tests/ v2/output/selection.txt` — "v2: walk-forward scoring and pre-registered selection".

### Task 8: Freeze, holdout evaluation, forward script

**Files:**
- Create: `v2/evaluate.py`, `v2/tests/test_evaluate.py`, `v2/README.md`
- Modify: `strategies/kalshi-tails/kalshi-multi-tail/README.md` (link to v2)

**Interfaces:**
- Consumes: Task 7 (`walk_forward_predict`, `block_bootstrap_delta`, `would_trade`), `multi_tail.run`,
  `paired_diff_ci`.
- Produces:
  - `FROZEN = {"set": <chosen>, "lam_beta": 25, "lam_alpha": 25, "refit_days": 7, "min_rows": 200}`, in
    `v2/logit_v2.py`.
  - `evaluate.py --period holdout|forward [--today YYYY-MM-DD]`:
    - holdout → `v2/output/holdout.txt`:
      - v2 vs S, P and mid, on the BTC + SPX common rows
      - one-sided 95th-percentile Δ
      - the per-day SD of paired Δ and the MDE = 2.5 × SE scaled to 181 days
      - the would-trade secondary
      - Sharpe v2 vs P via `multi_tail.run` and `paired_diff_ci`
      - an exploratory five-market block
    - forward: if `today < 2027-05-12` (default `today` = the real date), it prints only data-quality counts
      (rows and days per market, failed-day files) and exits 0. Otherwise it writes `v2/output/forward.txt` with
      the spec section 7 verdict: WIN only if both one-sided tests pass, otherwise "not shown".

- [ ] **Step 1: Write the freeze.** Set `FROZEN` from `selection.txt`. In `v2/README.md`, write a "Frozen v2
  configuration" section with the set, all spec section 4–5 constants, the run commands, and a placeholder line
  for the commit hash (filled in after the commit in Step 6). Add a "Forward refresh" section (spec section 9):
  - re-run `multi_pull.py` per market, then `multi_obs.py`
  - for BTC, re-run `btc_pm_data_pull.py` (including spot) and `btc_digital_model_backtest.py`
  - then `python -m v2.features`

  The holdout report's header line reads "HOLDOUT: seen period, not the verdict".
- [ ] **Step 2: Write failing tests:**
  - `test_forward_refuses_early`: `--today 2027-05-11` prints no "Δ" or "WIN" text and does not create
    `forward.txt`.
  - `test_verdict_requires_both`: synthetic predictions where v2 beats P but not S → "not shown"; beats both →
    "WIN".
  - `test_mde_formula`: a known per-day SD gives the expected MDE.
- [ ] **Step 3: Run.** Expected: FAIL.
- [ ] **Step 4: Implement** `evaluate.py`.
- [ ] **Step 5: Run** the v2 suite (expected: 31 passed) and the parent suite (expected: 34 passed). Then
  `python -m v2.evaluate --period holdout` and read `holdout.txt`.
- [ ] **Step 6: Commit** `v2/ README.md` — "v2: freeze configuration, holdout evaluation, forward script". Then
  write the commit hash into `v2/README.md` and commit that as "v2: record frozen hash".
